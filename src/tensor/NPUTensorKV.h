#pragma once

#include "NPUTensorInner.h"

enum class NPUTensorKVType { KEY, VALUE };

class NPUTensorKV : public NPUTensorInner {
   public:
    NPUTensorKV() = default;
    NPUTensorKV(std::vector<uint32_t> dims, NPUTensorKVType kv_type);
    virtual addr_type get_addr(std::vector<uint32_t> indexes);
    virtual std::vector<addr_type> get_all_addrs();
    uint32_t get_allocated_seq_len();
    void add_token();  // automatically allocates buffer each time a token is added during iteration

    NPUTensorKVType _kv_type;
    std::vector<addr_type> _bases;  // store row index allocated from KVCache
    uint32_t _kv_cache_entry_size;  // 32
    uint32_t _seq_len;              // seq_len of current KV cache 
};