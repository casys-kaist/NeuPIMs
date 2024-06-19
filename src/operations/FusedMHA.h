#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

class FusedMHA : public Operation {
   public:
    FusedMHA(std::string name);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    uint32_t _batch_size;
    std::vector<Ptr<NPUTensor>> _query;
    std::vector<Ptr<NPUTensor>> _key;
    std::vector<Ptr<NPUTensor>> _value;

    uint32_t _nh;
    uint32_t _dk;

    std::vector<uint32_t> _heads_per_tile;

    void calculate_loops();
    void initialize_tiles();
    void initialize_instructions(Tile &tile, int req_idx, int head_idx, int num_heads);
    uint32_t sram_size_needed();
};