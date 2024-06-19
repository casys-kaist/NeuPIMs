#include "AddressAllocator.h"

KVCacheAlloc::KVCacheAlloc()
    : _kv_cache_size(0), _kv_cache_limit(0), _kv_cache_entry_size(0), _base_addr(0), _base_row(0) {}

void KVCacheAlloc::init(addr_type base_addr) {
    _mode = Config::global_config.run_mode;
    if (_mode == RunMode::NPU_ONLY) {  // NPU only mode
        init_npu_layout(base_addr);
    } else if (_mode == RunMode::NPU_PIM) {
        init_pim_layout(base_addr);
    } else {
        ast(0);
    }
}

/**
 * Initialize NPU memory layout.
 * Firt, allocate the whole memory considering predetermined cache size.
 * Cache entry is consisted of 32 key and value, both sizes (d_k).
 *  Note that cache is saved as d_k granularity, not E
 * the memory layout of saved cache is (h,l,d_k).
 * this is because it is because consecutive memory access is faster,
 * so adjacent latent vector at certain head should be loaded faster.
 */
void KVCacheAlloc::init_npu_layout(addr_type base_addr) {
    uint32_t max_active_reqs = Config::global_config.max_active_reqs;
    uint32_t max_seq_len = Config::global_config.max_seq_len;
    uint32_t h = Config::global_config.model_n_head;
    uint32_t d_k = Config::global_config.model_n_embd / h;
    uint32_t precision = Config::global_config.precision;

    _base_addr = base_addr;
    _kv_cache_entry_size = 32;  // allocate once per seq_len 32
    _kv_cache_size = max_active_reqs * max_seq_len * h * d_k * precision;
    ast(_base_addr + _kv_cache_size < Config::global_config.HBM_size);

    addr_type next_addr = _base_addr;
    // The number of sequence lengths that can be stored per block / sequence length per block
    // (a block consists of 32 * d_k elements)
    // = Number of KV cache blocks in HBM
    uint64_t num_kv_cache_entries = max_active_reqs * max_seq_len * h / _kv_cache_entry_size;

    for (int i = 0; i < num_kv_cache_entries; ++i) {
        _kv_cache.push_back(next_addr);
        next_addr += _kv_cache_entry_size * d_k * precision;  // 32 seq_len * d_k * precision
    }
}

void KVCacheAlloc::init_pim_layout(addr_type base_addr) {
    // =rows of matrix in a DRAM PIM row
    constexpr uint32_t row_per_bank = 32768;
    constexpr uint32_t row_offset = 21;
    constexpr uint64_t mask = ~((1 << row_offset) - 1);  // 0x1111(64-21)0000(21)
    _dram_col_size = 1024;
    _num_ele_per_row = _dram_col_size / Config::global_config.precision;  // 512
    _bank_per_ch = Config::global_config.dram_banks_per_ch;
    _dram_channels = Config::global_config.dram_channels;

    base_addr = base_addr & mask;  // get last row index using
    base_addr = base_addr + (1 << row_offset);  // move to next row index

    _base_addr = base_addr;
    _base_row = base_addr >> row_offset;  // get only row index

    // _rows: channel -> row idx
    uint32_t free_rows_size = row_per_bank - _base_row;
    for (int i = 0; i < _dram_channels; ++i) {
        _rows.push_back(std::make_shared<std::deque<uint64_t>>());
        for (int j = 0; j < free_rows_size; ++j) {
            if (_base_row + j < row_per_bank) _rows[i]->push_back(_base_row + j);
        }
    }
}

// allocate space [bank per ch, d_k], and return
// when repeat this h times, we have allocated space [h, bank per ch, d_k]
addr_type KVCacheAlloc::allocate() {
    ast(_mode == RunMode::NPU_ONLY);
    ast(_kv_cache.size() > 0);
    addr_type addr = _kv_cache.front();
    _kv_cache.pop_front();
    return addr;
}

addr_type KVCacheAlloc::allocate(uint64_t ch) {
    ast(_mode == RunMode::NPU_PIM);
    ast(_rows[ch]->size() > 0);
    addr_type row = _rows[ch]->front();
    _rows[ch]->pop_front();
    return row;  // return free row
}

void KVCacheAlloc::free(addr_type addr) {
    ast(_mode == RunMode::NPU_ONLY);
    _kv_cache.push_back(addr);
}

void KVCacheAlloc::free(uint32_t ch, uint64_t row) {
    ast(_mode == RunMode::NPU_PIM);
    _rows[ch]->push_back(row);
}