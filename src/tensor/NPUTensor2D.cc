#include "NPUTensor2D.h"

#include "../allocator/AddressAllocator.h"

NPUTensor2D::NPUTensor2D(std::vector<uint32_t> dims, NPUTensorBufType buf_type)
    : NPUTensorInner(dims, buf_type) {
    _size = Config::global_config.precision;
    for (auto dim : dims) {
        _size *= dim;
    }

    if (buf_type == NPUTensorBufType::WGT)
        _base_addr = WgtAlloc::GetInstance()->allocate(_size);
    else if (buf_type == NPUTensorBufType::ACT)
        _base_addr = ActAlloc::GetInstance()->allocate(_size);
}

addr_type NPUTensor2D::get_addr(std::vector<uint32_t> indexes) {
    assert(indexes.size() == _dims.size());

    if (indexes.size() == 1)  // bias
        return _base_addr + indexes[0] * _precision;

    // return _base_addr + (indexes[0] * _dims[1] + indexes[1]) * _precision;
    return AddressConfig::switch_co_ch(_base_addr +
                                       (indexes[0] * _dims[1] + indexes[1]) * _precision);
}

std::vector<addr_type> NPUTensor2D::get_all_addrs() {
    std::vector<addr_type> ret;

    if (_dims.size() == 1) {
        for (uint32_t i = 0; i < _dims[0]; i++) {
            ret.push_back(_base_addr + i * _precision);
        }
    } else {
        for (uint32_t i = 0; i < _dims[0]; i++) {
            for (uint32_t j = 0; j < _dims[1]; j++) {
                ret.push_back(_base_addr + (i * _dims[1] + j) * _precision);
            }
        }
    }
    return ret;
}

std::vector<addr_type> NPUTensor2D::get_row_addrs(uint32_t row_idx) {
    std::vector<addr_type> ret;
    // _dims: [row, column]
    uint32_t col_size = _dims[1];
    for (uint32_t j = 0; j < col_size; j++) {
        ret.push_back(_base_addr + (row_idx * col_size + j) * _precision);
    }
    return ret;
}

std::vector<Ptr<NPUTensor2D>> NPUTensor2D::split_by_row(std::vector<uint32_t> row_dims) {
    ast(_dims.size() == 2);
    ast(std::accumulate(row_dims.begin(), row_dims.end(), 0) == _dims[0]);

    std::vector<Ptr<NPUTensor2D>> ret;
    uint32_t base_idx = 0;
    uint32_t column_size = _dims[1];

    for (auto row_dim : row_dims) {
        auto tensor = std::make_shared<NPUTensor2D>();
        tensor->_base_addr = get_addr({base_idx, 0});
        tensor->_dims = {row_dim, column_size};
        tensor->_size = _precision * row_dim * column_size;
        tensor->_buf_type = _buf_type;
        tensor->_precision = _precision;
        ret.push_back(tensor);
        base_idx += row_dim;
    }

    return ret;
}