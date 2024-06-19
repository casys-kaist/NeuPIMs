#pragma once
#include "Common.h"

// TensorBufType indicates which buffer the tensor is stored in.
// Weights might also be stored in the ACT (activation) buffer.
enum class TensorBufType { WGT, ACT, KV };  // weight, activation, key/value

class Tensor2D {
   public:
    Tensor2D() = default;
    Tensor2D(std::vector<uint32_t> dims, TensorBufType buf_type);
    addr_type get_addr(std::vector<uint32_t> indexes);
    std::vector<std::shared_ptr<Tensor2D>> split_by_row(std::vector<uint32_t> row_dims);

    addr_type _base_addr;
    std::vector<uint32_t> _dims;
    uint64_t _size;
    TensorBufType _buf_type;
};