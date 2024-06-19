#include "AddressAllocator.h"

ActAlloc::ActAlloc() : _base_addr(0), _top_addr(0), _act_buf_size(0), _act_buf_limit(0) {}

void ActAlloc::init(addr_type base_addr) {
    _base_addr = base_addr;
    _top_addr = base_addr;
    _act_buf_size = Config::global_config.HBM_act_buf_size;
    _act_buf_limit = _base_addr + _act_buf_size;
}

addr_type ActAlloc::allocate(uint64_t size) {
    ast(_top_addr + size < _act_buf_limit);
    uint32_t alignment = AddressConfig::alignment;

    addr_type result = _top_addr;
    _top_addr += size;
    if (_top_addr & (alignment - 1)) {
        _top_addr += alignment - (_top_addr & (alignment - 1));
    }
    return result;
}

addr_type ActAlloc::get_next_aligned_addr() {
    ast(_base_addr > 0);
    ast(_act_buf_size > 0);

    return AddressConfig::align(_act_buf_limit) + AddressConfig::alignment;
}

void ActAlloc::flush() { _top_addr = _base_addr; }