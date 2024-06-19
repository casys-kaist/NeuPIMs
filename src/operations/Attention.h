#pragma once
#include "../BatchedRequest.h"
#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"
#include "Operation.h"

class Attention : public Operation {
   public:
    Attention(std::string name, std::shared_ptr<BatchedRequest> breq);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    std::shared_ptr<BatchedRequest> _breq;
    uint32_t _N;
    std::vector<uint32_t> _seq_lens;
    uint32_t _batch_size;

    uint32_t _k;  // max num_heads per tile

    std::vector<Ptr<NPUTensor>> _qs;
    std::vector<Ptr<PIMTensor>> _ks;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    uint32_t _E;
    uint32_t _nh;
    uint32_t _dk;

    uint32_t _chunks;
    uint32_t _heads_per_tile;
    uint32_t _heads_in_last_chunk;
    uint32_t _comps_per_head;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(int k);
    uint32_t sram_size_needed();

    //    private:
    //     addr_type _spad_addr;
    //     addr_type _acc_spad_addr;

    //     std::pair<addr_type, uint32_t> allocate_sram_addr(uint32_t size, bool accum);
};