#pragma once

#include "NPUTensorInner.h"

class NPUTensor2D : public NPUTensorInner {
   public:
    NPUTensor2D() = default;
    NPUTensor2D(std::vector<uint32_t> dims, NPUTensorBufType buf_type);
    virtual addr_type get_addr(std::vector<uint32_t> indexes);
    virtual std::vector<addr_type> get_all_addrs();
    std::vector<addr_type> get_row_addrs(uint32_t row_idx);
    std::vector<Ptr<NPUTensor2D>> split_by_row(std::vector<uint32_t> row_dims);
};