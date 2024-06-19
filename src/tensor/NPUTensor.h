#pragma once

#include "BTensor.h"
#include "NPUTensor2D.h"
#include "NPUTensorKV.h"

class NPUTensor : public BTensor {
   public:
    NPUTensor() = default;
    NPUTensor(std::string name, std::vector<uint32_t> dims, NPUTensorBufType buf_type,
              bool produced);
    NPUTensor(std::string name, std::vector<uint32_t> dims, NPUTensorKVType kv_type, bool produced);
    NPUTensor(std::string name, Ptr<NPUTensor2D> tensor, bool produced);
    ~NPUTensor() = default;

    std::vector<uint32_t> get_dims();

    virtual addr_type get_addr(std::vector<uint32_t> indexes);
    virtual std::vector<addr_type> get_all_addrs();
    virtual void set_transposed();
    virtual void unset_transposed();
    virtual void add_token() override;  // for KV
    std::vector<addr_type> get_row_addrs(uint32_t row_idx);

    std::vector<Ptr<NPUTensor>> split_by_row(std::vector<uint32_t> row_dims);  // for 2D

    std::vector<Ptr<NPUTensorInner>> _inners;

    bool _is_transposed;
};