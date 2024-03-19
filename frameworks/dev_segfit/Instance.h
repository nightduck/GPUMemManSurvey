#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>
#include <list>
#include <vector>

#include "BasicInstance.cuh"

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
  block_t * dummy_node;         // Dummy node to simplify list operations
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

struct DeviceMemoryManager : MemoryManager
{
  CUdevice dev;             // CUDA device
  CUcontext pctx;           // CUDA context
  CUdeviceptr heap;         // Pointer to the start of the memory region
  tlsf_t tlsf;              // Overhead 
  void * final_block_addr;  // Pointer to start of block that is adjacent to heap
  size_t granularity;       // Minimum size when calling cuMemCreate
	static constexpr size_t alignment{16ULL};
  // static constexpr size_t HEAP_MASK{1UL<<(sizeof(size_t)*8-1)};  // Largest power of 2
  static constexpr size_t HEAP_BIN{sizeof(size_t)*8-1};          // Set last bit

	explicit DeviceMemoryManager(size_t instantiation_size) : MemoryManager()
	{
		if(initialized)
			return;
		// cudaDeviceSetLimit(cudaLimitMallocHeapSize, instantiation_size);
    
    // Initialize CUDA and virtual heap
    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuMemAddressReserve(&heap, instantiation_size, 0, 0, 0));

    // Initialize tlsf table
    tlsf.bins.resize(sizeof(size_t)*8);

    // Allocate first block (size set by granularity requirements)
    // CUmemGenericAllocationHandle allocHandle;
    // CUmemAccessDesc accessDesc;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    // accessDesc.location = prop.location;
    // accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_DRV(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // CHECK_DRV(cuMemCreate(&allocHandle, granularity, &prop, 0));
    // CHECK_DRV(cuMemMap(heap, granularity, 0ULL, allocHandle, 0ULL));
    // CHECK_DRV(cuMemSetAccess(heap, granularity, &accessDesc, 1ULL));
    final_block_addr = 0;
    block_t first_block = {(size_t)(char*)(uintptr_t)(heap), false, final_block_addr, 0, 0, 0};  // Final two values are irrelevant
    tlsf.ht[final_block_addr] = first_block;
    tlsf.avail_bitmask = 0;

		initialized = true;
	}
	~DeviceMemoryManager(){
    // TODO: Free all memory
  };

  void inline remove(tlsf_t &tlsf, block_t it) {
    tlsf.ht[it.prev].next = it.next;
    tlsf.ht[it.next].prev = it.prev;
    if (tlsf.bins[get_order_rounddown(it.size)] == it.addr) {
      tlsf.bins[get_order_rounddown(it.size)] = it.next;
    }
    tlsf.ht.erase(it.addr);
    return;
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
    std::cout << "Requesting " << size << " bytes from heap at " << heap_start << std::endl;
    CHECK_DRV(cuMemCreate(&allocHandle, size, &prop, 0));
    CHECK_DRV(cuMemMap((CUdeviceptr)heap_start, size, 0ULL, allocHandle, 0ULL));
    CHECK_DRV(cuMemSetAccess((CUdeviceptr)heap_start, size, &accessDesc, 1ULL));
    final_block_addr = heap_start;
    return heap_start;
  }

	virtual __forceinline__ void* malloc(size_t size) override
	{
    dump_blocks();
    sanity_check();
    std::cout << "Mallocing " << std::hex << size << std::endl;
    uint64_t requested_bins = (1L << (8*sizeof(size_t)-1) >> __builtin_clzl(size - 1) - 1);
    requested_bins = tlsf.avail_bitmask & requested_bins;
    size_t bin_idx = __builtin_ctzl(requested_bins);

    block_t it;
    if (requested_bins == 0) {  // If no bin can service the request, allocate a new page at heap
      // // Remove final block from its bin
      // it = tlsf.ht[final_block_addr];
      // bin_idx = get_order_rounddown(it.size);
      // tlsf.bins[bin_idx] = it.next;

      // Add the newly allocate page(s) to the final block
      size_t actual_size = size;
      it = block_t{0, false, 0, final_block_addr, 0, 0};
      it.addr = request_from_heap(actual_size);
      it.size = actual_size;
    } else {
      it = tlsf.ht[tlsf.bins[bin_idx]];
      tlsf.bins[bin_idx] = it.next;
      tlsf.ht[it.next].prev = 0;
      tlsf.ht[it.prev].next = 0;
      if (it.next == 0) {
        size_t unset_bit = 1L << get_order_rounddown(it.size);
        tlsf.avail_bitmask &= ~unset_bit;
      }
      it.prev = 0;
      it.next = 0;
      it.free = false;
    }

    // it.size is what was requested, and size is the amount that was obtained from the heap
    if (size < it.size) {
      block_t new_block;
      new_block.size = it.size - size;
      new_block.free = true;
      new_block.addr = it.addr + size;
      new_block.prev_adj = it.addr;
      new_block.prev = 0;
      it.size = size;

      // Insert split off portion into the appropriate bin
      bin_idx = get_order_rounddown(new_block.size);
      new_block.next = tlsf.bins[bin_idx];
      tlsf.ht[new_block.next].prev = new_block.addr;
      tlsf.bins[bin_idx] = new_block.addr;
      tlsf.avail_bitmask |= 1L << bin_idx;

      tlsf.ht[new_block.addr] = new_block;

      // Update the final block, if relevant
      if (it.addr == final_block_addr) {
        final_block_addr = new_block.addr;
      } else {
        // Update prev_adj of the block after the split
        tlsf.ht[new_block.addr + new_block.size].prev_adj = new_block.addr;
      }
    }
    tlsf.ht[it.addr] = it;

    return it.addr;
	}

	virtual __forceinline__ void free(void* ptr) override
	{
    dump_blocks();
    sanity_check();
    std::cout << "Freeing " << ptr << std::endl;
    const size_t top_idx = 1 << sizeof(size_t);
    auto& it = tlsf.ht[ptr];
    auto& left = tlsf.ht[it.prev_adj];
    auto& right = tlsf.ht[it.addr + it.size];

    // Coalesce
    if (left.free) {
      if (tlsf.bins[get_order_rounddown(left.size)] == left.addr) {
        tlsf.bins[get_order_rounddown(left.size)] = left.next;
        if (left.prev == left.next) {
          size_t unset_bit = 1L << get_order_rounddown(left.size);
          tlsf.avail_bitmask &= ~unset_bit;
        }
      }
      if (right.free) {
        left.size += it.size + right.size;
        
        // Update the final block, if relevant
        if (right.addr == final_block_addr) {
          final_block_addr = left.addr;
        } else {
          // Update prev_adj of the block after the split
          tlsf.ht[right.addr + right.size].prev_adj = right.addr;
        }

        if (right.prev == right.next) {
          size_t unset_bit = 1L << get_order_rounddown(right.size);
          tlsf.avail_bitmask &= ~unset_bit;
        }
        remove(tlsf, right);
      } else {
        left.size += it.size;
        right.prev_adj = left.addr;
      }
      remove(tlsf, it);
      it = left;
    } else if (right.free) {
      if (tlsf.bins[get_order_rounddown(it.size)] == it.addr) {
        tlsf.bins[get_order_rounddown(it.size)] = it.next;
      }
      it.size += right.size;
      tlsf.ht[right.addr + right.size].prev_adj = it.addr;
      if (right.prev == right.next) {
        size_t unset_bit = 1L << get_order_rounddown(right.size);
        tlsf.avail_bitmask &= ~unset_bit;
      }
      remove(tlsf, right);
    }

    // Insert coalesced block into the appropriate bin
    tlsf.ht[it.prev].next = it.next;
    tlsf.ht[it.next].prev = it.prev;
    size_t bin_idx = get_order_rounddown(it.size);
    it.next = tlsf.bins[bin_idx];
    tlsf.ht[it.next].prev = it.addr;
    it.prev = 0;
    tlsf.bins[bin_idx] = it.addr;
    it.free = true;
    tlsf.ht[it.addr] = it;
    tlsf.avail_bitmask |= 1L << bin_idx;
    return;
	}

  void dump_blocks() {
    bool failure = false;
    std::cout << std::endl;
    void * expected_next = 0;
    std::cout << "Availability " << std::hex << tlsf.avail_bitmask << std::dec << std::endl;
    for (auto it = tlsf.ht.begin(); it != tlsf.ht.end(); it++) {
      std::cout << "Block at " << it->first << " with size " << std::hex << it->second.size << " and free " << std::dec << it->second.free << " ptrs {" <<
        it->second.prev_adj << ", " << it->second.next << ", " << it->second.prev << "}" << std::endl;

      if (it->first != it->second.addr) {
        std::cout << "!!!!!!!!!!!!!!!!Block at " << it->first << " has address (" << it->second.addr << ") that doesn't match hash" << std::endl;
        failure = true;
      }
      if (it->second.free && !(tlsf.avail_bitmask & (1UL << get_order_rounddown(it->second.size)))) {
        std::cout << "!!!!!!!!!!!!!!!!Found block of size " << it->second.size << ", but bin is not set" << std::endl;
        failure = true;
      }
      if (it->second.size == 0) {
        std::cout << "!!!!!!!!!!!!!!!!Found block of size 0" << std::endl;
        failure = true;
      }
      if (it->second.addr != expected_next) {
        std::cout << "!!!!!!!!!!!!!!!!Block at " << it->first << " is not adjacent to previous block" << std::endl;
        failure = true;
      }
      block_t prev_block = tlsf.ht[it->second.prev_adj];
      if (prev_block.addr != 0 && it->first != prev_block.addr + prev_block.size) {
        std::cout << "!!!!!!!!!!!!!!!!Block at " << it->first << " has prev_adj that doesn't match next" << std::endl;
        failure = true;
      }
      expected_next = it->first + it->second.size;
    }
    for (int i = 0; i < tlsf.bins.size(); i++) {
      if (tlsf.bins[i] == 0) {
        continue;
      }
      std::cout << "Bin " << i << " has " << tlsf.bins[i] << std::endl;
    }
    if (failure) {
      throw;
    }
  }

  void sanity_check() {
    for (int i = 0; i < tlsf.bins.size(); i++) {
      if (tlsf.bins[i] == 0) {
        if (tlsf.avail_bitmask & (1UL << i)) {
          std::cout << "!!!Bin " << i << " is empty, but bit is set in avail_bitmask" << std::endl;
          throw;
        }
        continue;
      }
      auto block = tlsf.ht[tlsf.bins[i]];
      if (block.prev != 0) {
        std::cout << "!!!Block " << block.addr << " in bin " << i << " is at head of list with prev elements" << std::endl;
        throw;
      }
      do {
        if (!block.free) {
          std::cout << "!!!Block " << block.addr << " in bin " << i << " is not free" << std::endl;
          throw;
        }
        if (get_order_rounddown(block.size) != i) {
          std::cout << "!!!Block " << block.addr << " in bin " << i << " is in the wrong bin" << std::endl;
          throw;
        }
        block = tlsf.ht[block.next];
      } while(block.addr != 0);
      // Check bit is set in avail_bitmask
      if (!(tlsf.avail_bitmask & (1UL << i))) {
        std::cout << "!!!Block << " << tlsf.bins[i] << " in bin " << i << " is not in avail_bitmask" << std::endl;
        throw;
      }
    }
    std::cout << std::endl;
  }
};

#endif
