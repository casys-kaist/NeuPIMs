#pragma once
#include "../Common.h"

// Used in NPU + PIM to allocate weights, and in NPU only to allocate all tensors.
class WgtAlloc : public Singleton<WgtAlloc> {
   private:
    friend class Singleton;
    WgtAlloc();
    ~WgtAlloc() = default;

   public:
    addr_type _base_addr;
    uint64_t _top_addr;

    addr_type allocate(uint64_t size);
    addr_type get_next_aligned_addr();
};

class ActAlloc : public Singleton<ActAlloc> {
   private:
    friend class Singleton;
    ActAlloc();
    ~ActAlloc() = default;

   public:
    addr_type _base_addr;
    addr_type _top_addr;
    uint64_t _act_buf_size;   // fixed.
    uint64_t _act_buf_limit;  // _base_addr + _act_buf_size

    void init(addr_type base_addr);
    addr_type allocate(uint64_t size);
    addr_type get_next_aligned_addr();  // aligned limit addr + alignment of ActAlloc buf
    void flush();
};

class KVCacheAlloc : public Singleton<KVCacheAlloc> {
   private:
    friend class Singleton;
    KVCacheAlloc();
    ~KVCacheAlloc() = default;

   public:
    RunMode _mode;
    addr_type _base_addr;

    // for NPU layout
    uint64_t _kv_cache_size;          // fixed.
    uint64_t _kv_cache_limit;         // _base_addr + _kv_cache_size
    uint64_t _kv_cache_entry_size;    // 32 (bank per ch)
    std::deque<addr_type> _kv_cache;  // base_addr of each entry

    // for PIM layout
    uint64_t _dram_channels;
    uint64_t _base_row;         // _base_addr >> row_offset
    uint32_t _dram_row_size;    // DRAM row size (1024KB)
    uint32_t _num_ele_per_row;  // DRAM row size / precision
    uint32_t _bank_per_ch;
    std::vector<Ptr<std::deque<uint64_t>>> _rows;  // channel -> free rows base index

    void init(addr_type base_addr);
    void init_npu_layout(addr_type base_addr);
    void init_pim_layout(addr_type base_addr);
    addr_type allocate();
    addr_type allocate(uint64_t ch);
    void free(addr_type addr);
    void free(uint32_t ch, uint64_t row);
};