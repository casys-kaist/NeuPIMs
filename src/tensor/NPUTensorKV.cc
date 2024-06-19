#include "NPUTensorKV.h"

#include "../allocator/AddressAllocator.h"
#include "spdlog/spdlog.h"

/**
 * NPUTensorKV
 *  allocate memory address with KVCacheAlloc.
 *  The size of each allocated block is _kv_cache_entry_size(32) * d_k * precision
 *  Memory layout for Key and Value is (d_k,n) and (n,d_k) each.
 *      each n in Key and Value is divided into 32.
 *      so, (d_k,n) values become ceil(n,32) of (d_k,32)
 */
NPUTensorKV::NPUTensorKV(std::vector<uint32_t> dims, NPUTensorKVType kv_type)
    : NPUTensorInner(dims, NPUTensorBufType::KV), _kv_type(kv_type) {
    auto alloc = KVCacheAlloc::GetInstance();
    _kv_cache_entry_size = alloc->_kv_cache_entry_size;

    // K: [h, d_k, n], V: [h, n, d_k]
    // NPUTensorKV has shape of [d_k, n] or [n, d_k]
    // NPUTensor has h NPUTensorKV
    _seq_len = kv_type == NPUTensorKVType::KEY ? dims[1] : dims[0];
    uint32_t num_required_alloc = ceil((double)_seq_len / (double)_kv_cache_entry_size);
    // spdlog::info("num_required_alloc: {}, seq_len: {}, kv_cache_entry_size: {}",
    // num_required_alloc,
    //  seq_len, kv_cache_entry_size);
    for (int i = 0; i < num_required_alloc; ++i) {
        _bases.push_back(alloc->allocate());
    }
}

addr_type NPUTensorKV::get_addr(std::vector<uint32_t> indexes) {
    // key: [dk, seq_len]
    // value: [seq_len, dk]
    auto alloc = KVCacheAlloc::GetInstance();

    uint32_t seq_idx = _kv_type == NPUTensorKVType::KEY ? indexes[1] : indexes[0];
    uint32_t byte_idx = _kv_type == NPUTensorKVType::KEY ? indexes[0] : indexes[1];
    uint32_t dk = _kv_type == NPUTensorKVType::KEY ? _dims[0] : _dims[1];

    uint32_t idx = floor((double)seq_idx / (double)_kv_cache_entry_size);
    addr_type base_addr = _bases[idx];
    uint32_t offset = ((seq_idx % _kv_cache_entry_size) * dk + byte_idx) * _precision;
    return AddressConfig::switch_co_ch(base_addr + offset);
}

std::vector<addr_type> NPUTensorKV::get_all_addrs() {
    std::vector<addr_type> ret;
    uint32_t d_k = _kv_type == NPUTensorKVType::KEY ? _dims[0] : _dims[1];

    uint32_t cnt = 0;  // counter for (32, d_k)
    uint32_t idx = 0;  // which pointer is used in _bases
    for (int i = 0; i < _seq_len * d_k; ++i) {
        // spdlog::info("i: {}, cnt: {}, idx: {}", i, cnt, idx);
        ret.push_back(_bases[idx] + cnt * _precision);

        cnt++;
        if (cnt == _kv_cache_entry_size * d_k) {
            cnt = 0;
            idx++;
        }
    }
    return ret;
}

uint32_t NPUTensorKV::get_allocated_seq_len() { return _bases.size() * _kv_cache_entry_size; }

/**
 * After each iteration, call this function for inference.
 */
void NPUTensorKV::add_token() {
    _seq_len++;
    if (_kv_type == NPUTensorKVType::KEY)
        _dims[1]++;
    else
        _dims[0]++;

    if (_seq_len <= get_allocated_seq_len()) return;

    _bases.push_back(KVCacheAlloc::GetInstance()->allocate());
}