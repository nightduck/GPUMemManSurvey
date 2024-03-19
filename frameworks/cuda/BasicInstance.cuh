#pragma once
#include "TestInstance.cuh"

struct MemoryManagerCUDA
{
	explicit MemoryManagerCUDA(size_t instantiation_size)
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		initialized = true;
	}
	~MemoryManagerCUDA(){};

	static constexpr size_t alignment{16ULL};

	virtual __forceinline__ void* malloc(size_t size) override
	{
		return cudaMalloc(size);
	}

	virtual __forceinline__ void free(void* ptr) override
	{
		cudaFree(ptr);
	};

	static bool initialized;
};

bool MemoryManagerCUDA::initialized = false;