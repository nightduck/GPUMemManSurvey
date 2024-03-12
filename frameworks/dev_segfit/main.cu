#include <iostream>
#include <stdio.h>

#include "Instance.cuh"

__global__ void fill_values(int * test_array) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  test_array[tx] = tx;
}

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

  std::cout << "Memory manager initialized" << std::endl;

	testFunctions (memory_manager);

  std::cout << "Test functions done" << std::endl;

	cudaDeviceSynchronize();

	printf("Testcase done!\n");

	return 0;
}