#pragma once

struct MemoryManager
{
  bool initialized;
  MemoryManager() : initialized(false) {};
  virtual __forceinline__ void* malloc(size_t size) = 0;
  virtual __forceinline__ void free(void* ptr) = 0;
};

struct MemoryManagerCUDA : MemoryManager
{
	explicit MemoryManagerCUDA(size_t instantiation_size) : MemoryManager()
	{
		if(initialized)
			return;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, instantiation_size);
		initialized = true;
	}
	~MemoryManagerCUDA(){};

	static constexpr size_t alignment{16ULL};

	virtual __forceinline__ void* malloc(size_t size) override
	{
    void * ptr;
		cudaMalloc(&ptr, size);
    return ptr;
	}

	virtual __forceinline__ void free(void* ptr) override
	{
		cudaFree(ptr);
	};

	static bool initialized;
};

bool MemoryManagerCUDA::initialized = false;