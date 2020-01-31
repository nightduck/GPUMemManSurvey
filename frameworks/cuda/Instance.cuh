#pragma once
#include "TestInstance.cuh"

struct MemoryManagerCUDA : public MemoryManagerBase
{
	explicit MemoryManagerCUDA(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size){}
	~MemoryManagerCUDA(){};

	virtual void init() override
	{
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	}

	virtual __device__ __forceinline__ void* malloc(size_t size)
	{
		return malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr)
	{
		free(ptr);
	};
};