#pragma once
#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"
#include "Operation.h"

class NeuPIMSLogitSoftmax : public Operation {
   public:
    NeuPIMSLogitSoftmax(std::string name);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    uint32_t _batch_size;
    std::vector<Ptr<NPUTensor>> _qs;
    std::vector<Ptr<PIMTensor>> _ks;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    uint32_t _nh;
    uint32_t _dk;
    uint32_t _E;
    uint32_t _chunks;
    uint32_t _heads_per_tile;
    uint32_t _heads_in_last_chunk;
    uint32_t _comps_per_head;

    // new
    std::vector<int> _req_idxs;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(int start, int end);
    uint32_t sram_size_needed();

    //    private:
    //     addr_type _spad_addr;
    //     addr_type _acc_spad_addr;

    //     std::pair<addr_type, uint32_t> allocate_sram_addr(uint32_t size, bool accum);
};