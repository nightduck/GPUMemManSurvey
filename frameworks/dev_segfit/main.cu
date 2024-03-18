#include <chrono>
#include <random>
#include <iostream>
#include <memory>
#include <stdio.h>

#include "BasicInstance.cuh"
#include "Instance.cuh"

__global__ void fill_values(int * test_array) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  test_array[tx] = tx;
}

typedef struct blob {
  size_t size;
  MemoryManager *mm;
  void* addr;

  blob(MemoryManager *mm, size_t size) : size(size), mm(mm) {
    addr = mm->malloc(size);
  }
  ~blob() {
    mm->free(addr);
  }
} blob_t;

void testFunctions(DeviceMemoryManager memory_manager)
{
	// int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// if(tid > 0)
	// 	return;

	int* d_test_array = reinterpret_cast<int*>(memory_manager.malloc(sizeof(int) * 16));

  printf("Allocated memory\n");

	fill_values<<<1,16>>>(d_test_array);

  printf("Filled values\n");

	memory_manager.free(d_test_array);
  
  printf("Freed memory\n");

	return;
}

int main(int argc, char* argv[])
{
	std::cout << "Simple CUDA Testcase\n";

	DeviceMemoryManager memory_manager(1024*1024*1024);

  // std::cout << "Memory manager initialized" << std::endl;
	// // testFunctions (memory_manager);
  // std::cout << "Test functions done" << std::endl;
  static constexpr int kNumBuffers = 20;
  static constexpr size_t kMinBufferSize = 1024;
  static constexpr size_t kMaxBufferSize = 16 * 1024 * 1024;
  std::unique_ptr<blob_t> buffers[kNumBuffers];

  std::mt19937 gen(42); //rd());
  std::uniform_int_distribution<> size_distribution(kMinBufferSize, kMaxBufferSize);
  std::uniform_int_distribution<> buf_number_distribution(0, kNumBuffers - 1);
  
  static constexpr int kNumIterations = 2000;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kNumIterations; ++i) {
    int buffer_idx = buf_number_distribution(gen);
    size_t new_size = size_distribution(gen);
    // std::cout << buffer_idx << ":" << new_size << std::endl;
    buffers[buffer_idx] = std::make_unique<blob_t>(&memory_manager, new_size);
  }

	cudaDeviceSynchronize();

	printf("Testcase done!\n");

	return 0;
}