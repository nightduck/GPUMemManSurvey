#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>
#include <list>
#include <vector>

#ifndef __CUDA_SEGFIT_HPP__
#define __CUDA_SEGFIT_HPP__
#define PAGESIZE 4096
#define CU_PTR(x) (CUdeviceptr)(uintptr_t)x;
#define VOID_PTR(x) (void*)(uintptr_t)x;

typedef struct block {
  size_t size;
  bool free;
  void* addr;
  void* prev_adj;
  void* next;
  void* prev;
} block_t;

typedef struct tlsf {
  size_t avail_bitmask;         // Used to indicate which bins have non-empty free lists
  std::vector<void*> bins;      // List of free blocks that form the head of each free list
  std::map<void*, block_t> ht;  // Hashtable to store block information
  block_t dummy_node;           // Dummy node to simplify list operations
} tlsf_t;

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

struct DeviceMemoryManager
{
  CUdevice dev;             // CUDA device
  CUcontext pctx;           // CUDA context
	bool initialized;         // Whether this instance has been initialized
  CUdeviceptr heap;         // Pointer to the start of the memory region
  tlsf_t tlsf;              // Overhead 
  void * final_block_addr;  // Pointer to start of block that is adjacent to heap
  size_t granularity;       // Minimum size when calling cuMemCreate
	static constexpr size_t alignment{16ULL};
  // static constexpr size_t HEAP_MASK{1UL<<(sizeof(size_t)*8-1)};  // Largest power of 2
  static constexpr size_t HEAP_BIN{sizeof(size_t)*8-1};          // Set last bit

	explicit DeviceMemoryManager(size_t instantiation_size)
	{
		if(initialized)
			return;
		// cudaDeviceSetLimit(cudaLimitMallocHeapSize, instantiation_size);
    
    // Initialize CUDA and virtual heap
    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuMemAddressReserve(&heap, instantiation_size, 0, 0, 0));

    // Initialize tlsf table
    tlsf.bins.resize(sizeof(size_t)*8);
    tlsf.dummy_node = {0, false, nullptr, nullptr, &tlsf.dummy_node, &tlsf.dummy_node};

    // Allocate first block (size set by granularity requirements)
    CUmemGenericAllocationHandle allocHandle;
    CUmemAccessDesc accessDesc;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_DRV(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CHECK_DRV(cuMemCreate(&allocHandle, granularity, &prop, 0));
    CHECK_DRV(cuMemMap(heap, granularity, 0ULL, allocHandle, 0ULL));
    CHECK_DRV(cuMemSetAccess(heap, granularity, &accessDesc, 1ULL));
    final_block_addr = VOID_PTR(heap);
    block_t first_block = {granularity, true, final_block_addr, 0, &tlsf.dummy_node, &tlsf.dummy_node};
    tlsf.ht[final_block_addr] = first_block;
    tlsf.bins[get_order_rounddown(granularity)] = final_block_addr;
    tlsf.avail_bitmask = 1 << get_order_rounddown(granularity);

		initialized = true;
	}
	~DeviceMemoryManager(){
    // TODO: Free all memory
  };

  void inline remove(tlsf_t tlsf, block_t it) {
    tlsf.ht[it.prev].next = it.next;
    tlsf.ht[it.next].prev = it.prev;
    tlsf.ht.erase(it.addr);
  }

  size_t inline get_order_roundup(size_t size) {
    return 8*sizeof(size_t) - __builtin_clzl(size - 1);
  }

  size_t inline get_order_rounddown(size_t size) {
    return 8*sizeof(size_t) - __builtin_clzl(size) - 1;
  }

  void * request_from_heap(size_t &size) {
    CUmemGenericAllocationHandle allocHandle;
    CUmemAccessDesc accessDesc;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    size = ((size + granularity - 1) / granularity) * granularity;

    void * heap_start = final_block_addr + tlsf.ht[final_block_addr].size;
    CHECK_DRV(cuMemCreate(&allocHandle, size, &prop, 0));
    CHECK_DRV(cuMemMap((CUdeviceptr)heap_start, size, 0ULL, allocHandle, 0ULL));
    CHECK_DRV(cuMemSetAccess((CUdeviceptr)heap_start, size, &accessDesc, 1ULL));
    return heap_start;
  }

	virtual __forceinline__ void* malloc(size_t size)
	{
    uint64_t requested_bins = (1L << (8*sizeof(size_t)-1) >> __builtin_clzl(size - 1) - 1);
    requested_bins = tlsf.avail_bitmask & requested_bins;
    size_t bin_idx = __builtin_ctzl(requested_bins);

    block_t it;
    if (requested_bins == 0) {  // If no bin can service the request, allocate a new page at heap
      // Remove final block from its bin
      it = tlsf.ht[final_block_addr];
      bin_idx = get_order_rounddown(it.size);
      tlsf.bins[bin_idx] = it.next;

      // Add the newly allocate page(s) to the final block
      request_from_heap(size);
      it.size += size;
      it.free = false;
    } else {
      it = tlsf.ht[tlsf.bins[bin_idx]];
      tlsf.bins[bin_idx] = it.next;
      it.free = false;
    }

    if (size < it.size) {
      block_t new_block;
      void* next_addr = it.addr + it.size;
      tlsf.ht[next_addr].prev_adj = it.addr + size;
      new_block.size = it.size - size;
      new_block.free = true;
      new_block.addr = it.addr + size;
      new_block.prev_adj = it.addr;

      // Insert split off portion into the appropriate bin
      bin_idx = get_order_rounddown(new_block.size);
      new_block.next = tlsf.bins[bin_idx];
      tlsf.ht[new_block.next].prev = new_block.addr;
      tlsf.bins[bin_idx] = new_block.addr;

      tlsf.ht[new_block.addr] = new_block;

      // Update the final block, if relevant
      if (it.addr == final_block_addr) {
        final_block_addr = new_block.addr;
      }
    }

    return it.addr;
	}

	virtual __forceinline__ void free(void* ptr)
	{
    const size_t top_idx = 1 << sizeof(size_t);
    block_t it = tlsf.ht[ptr];
    block_t left = tlsf.ht[it.prev_adj];
    block_t right = tlsf.ht[it.addr + it.size];
    it.free = true;

    // Coalesce
    if (left.free) {
      if (right.free) {
        left.size += it.size + right.size;
        tlsf.ht[right.next].prev_adj = left.addr;

        if (right.prev != right.next) {
          size_t unset_bit = top_idx >> __builtin_clzl(right.size - 1) - 1;
          tlsf.avail_bitmask &= ~unset_bit;
        }
        remove(tlsf, right);
      } else {
        left.size += it.size;
        right.prev_adj = left.addr;
      }
      if (it.prev != it.next) {
        size_t unset_bit = top_idx >> __builtin_clzl(it.size - 1) - 1;
        tlsf.avail_bitmask &= ~unset_bit;
      }
      remove(tlsf, it);
      it = left;
    } else if (right.free) {
      it.size += right.size;
      tlsf.ht[right.addr + right.size].prev_adj = it.addr;
      if (right.prev != right.next) {
        size_t unset_bit = top_idx >> __builtin_clzl(right.size - 1) - 1;
        tlsf.avail_bitmask &= ~unset_bit;
      }
      remove(tlsf, right);
    }

    // Insert coalesced block into the appropriate bin
    tlsf.ht[it.prev].next = it.next;
    tlsf.ht[it.next].prev = it.prev;
    size_t bin_idx = __builtin_clzl(it.size - 1);
    it.next = tlsf.bins[bin_idx];
    it.prev = &tlsf.dummy_node;
    tlsf.bins[bin_idx] = it.addr;
    tlsf.avail_bitmask |= top_idx >> bin_idx;
	};
};

#endif
