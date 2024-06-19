#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

// split input tensor to three
class SplitDecoding : public Operation {
   public:
    SplitDecoding(std::string name, std::pair<Ptr<BTensor>, Ptr<BTensor>> kv_cache,
                  bool is_initiated);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);
    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(uint32_t B, uint32_t N, uint32_t K, uint32_t M, bool should_store);
    uint32_t sram_size_needed();

   private:
    bool _is_initiated;
};