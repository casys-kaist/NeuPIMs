#pragma once
#include "Common.h"

class BatchedRequest {
   public:
    BatchedRequest() = default;
    BatchedRequest(std::vector<std::shared_ptr<InferRequest>> reqs);
    uint32_t get_num_reqs();
    uint32_t get_num_rows();
    std::vector<uint32_t> get_num_rows_breakdown();

    bool is_initiated(uint32_t index);
    std::pair<Ptr<BTensor>, Ptr<BTensor>> get_cache(uint32_t layer, uint32_t index);

    // todo: switch from Tensor to VirtualTensor
    int _batch_size;
    std::vector<std::shared_ptr<InferRequest>> _reqs;
};