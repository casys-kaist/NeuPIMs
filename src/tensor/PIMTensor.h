#pragma once

#include "BTensor.h"

enum class PIMTensorKVType { KEY, VALUE };

class PIMTensor : public BTensor {
   public:
    PIMTensor() = default;
    PIMTensor(std::string name, uint32_t ch, std::vector<uint32_t> dims, PIMTensorKVType kv_type,
              bool produced);
    ~PIMTensor() = default;

    virtual addr_type get_addr(std::vector<uint32_t> indexes) override;
    virtual std::vector<addr_type> get_all_addrs() override;
    virtual void add_token()
        override;  // automatically allocates buffer each time a token is added during iteration.

    uint32_t get_allocated_seq_len();
    uint32_t get_num_rows();
    uint32_t get_channel();
    std::vector<uint64_t> get_rows();

    PIMTensorKVType _kv_type;
    uint32_t _bank_per_ch;
    uint32_t _E;
    uint32_t _num_ele_per_row;
    // for here, row means DRAM row
    // how many rows to allocate at once when additional allocation is needed due to increased seq_len.
    uint32_t _num_rows_per_alloc;

    uint32_t _ch;                 // DRAM channel
    std::vector<uint64_t> _rows;  // store the row index allocated from KVCache.
    uint32_t _seq_len;
};