#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm> 

#include "UtilityFunctions.cuh"

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
#endif

template <typename MemoryManagerType, bool warp_based>
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(threadIdx.x % 32 != 0)
			return;
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
	}
	if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size));
}

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType mm, int** verification_ptr, int num_allocations)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	mm.free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
	// Usage: num_allocations size_of_allocation_in_byte print_output
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte{16};
	int num_iterations {25};
	bool warp_based{false};
	bool print_output{true};
	bool free_memory{true};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_byte = atoi(argv[2]);
			if(argc >= 4)
			{
				num_iterations = atoi(argv[3]);
				if(argc >= 5)
				{
					warp_based = static_cast<bool>(atoi(argv[4]));
					if(argc >= 6)
					{
						print_output = static_cast<bool>(atoi(argv[5]));
						if(argc >= 7)
							free_memory = static_cast<bool>(atoi(argv[6]));
					}
				}
			}
		}
	}

	allocation_size_byte = alignment(allocation_size_byte, sizeof(int));
	if(print_output)
		std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << std::endl;

	int device{0};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	
	#ifdef TEST_CUDA
	if(print_output)
		std::cout << "--- CUDA ---\n";
	MemoryManagerCUDA memory_manager;
	std::string mem_name("CUDA");
	#elif TEST_HALLOC
	if(print_output)
		std::cout << "--- Halloc ---\n";
	MemoryManagerHalloc memory_manager;
	std::string mem_name("Halloc");
	#elif TEST_SCATTERALLOC
	if(print_output)
		std::cout << "--- ScatterAlloc ---\n";
	MemoryManagerScatterAlloc memory_manager;
	std::string mem_name("ScatterAlloc");
	#elif TEST_OUROBOROS
	if(print_output)
		std::cout << "--- Ouroboros ---\n";
	MemoryManagerOuroboros memory_manager;
	std::string mem_name("Ouroboros");
	#endif

	memory_manager.init();

	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	std::ofstream results_frag;
	results_frag.open((std::string("../results/frag_") + prop.name  + "_" + mem_name + "_" + std::to_string(num_allocations) + ".csv").c_str(), std::ios_base::app);
	results_frag << "\n" << allocation_size_byte << ",";

	int blockSize {256};
	int gridSize {divup<int>(num_allocations, blockSize)};
	if (warp_based)
		gridSize *= 32;

	for(auto i = 0; i < num_iterations; ++i)
	{
		if(warp_based)
			d_testAllocation <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
		else
			d_testAllocation <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations, allocation_size_byte);
		CHECK_ERROR(cudaDeviceSynchronize());

		// Look at address range
		static int* static_min_ptr{reinterpret_cast<int*>(0xFFFFFFFFFFFFFFFFULL)};
		static int* static_max_ptr{nullptr};
		std::vector<int*> verification_pointers(num_allocations);
		CHECK_ERROR(cudaMemcpy(verification_pointers.data(), d_memory, sizeof(int*) * verification_pointers.size(), cudaMemcpyDeviceToHost));
		auto min_ptr = *min_element(verification_pointers.begin(), verification_pointers.end());
		auto max_ptr = *max_element(verification_pointers.begin(), verification_pointers.end());
		static_min_ptr = std::min(static_min_ptr, min_ptr);
		static_max_ptr = std::max(static_max_ptr, max_ptr);
		printf("%llu | %llu | %llu MB | %llu | %llu | %llu B\n", 
		reinterpret_cast<unsigned long long>(min_ptr), 
		reinterpret_cast<unsigned long long>(max_ptr), 
		(reinterpret_cast<unsigned long long>(max_ptr) - reinterpret_cast<unsigned long long>(min_ptr)) / (1024*1024),
		reinterpret_cast<unsigned long long>(static_min_ptr), 
		reinterpret_cast<unsigned long long>(static_max_ptr), 
		(reinterpret_cast<unsigned long long>(static_max_ptr) - reinterpret_cast<unsigned long long>(static_min_ptr)));
		results_frag << (reinterpret_cast<unsigned long long>(max_ptr) - reinterpret_cast<unsigned long long>(min_ptr)) 
			<< "," 
			<<(reinterpret_cast<unsigned long long>(static_max_ptr) - reinterpret_cast<unsigned long long>(static_min_ptr));
		if(num_iterations != 1)
			results_frag << ",";

		if(free_memory)
		{
			d_testFree <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
			CHECK_ERROR(cudaDeviceSynchronize());
		}
	}
	
	return 0;
}