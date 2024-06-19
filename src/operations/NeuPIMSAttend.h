
#pragma once
#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"
#include "Operation.h"

class NeuPIMSAttend : public Operation {
   public:
    NeuPIMSAttend(std::string name);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    uint32_t _batch_size;
    std::vector<Ptr<NPUTensor>> _logits;
    std::vector<Ptr<PIMTensor>> _vs;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    // model spec
    uint32_t _nh;
    uint32_t _dk;

    // memory spec
    uint32_t _page_size;
    uint32_t _banks_per_channel;

    uint32_t _tiles_per_chunk;
    uint32_t _datas_per_comp_cmd;

    // loop
    std::vector<int> _req_idxs;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(int start, int end);
    uint32_t sram_size_needed();
};