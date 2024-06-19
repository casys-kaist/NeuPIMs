#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

class Microbench : public Operation {
   public:
    Microbench(std::string name);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);

   private:
    uint32_t _prod_batches;

    std::vector<uint32_t> _input_dim;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions();
    uint32_t sram_size_needed();
};