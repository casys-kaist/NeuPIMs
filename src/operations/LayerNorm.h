#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

class LayerNorm : public Operation {
   public:
    // LayerNorm(std::string name, std::vector<uint32_t> weight_dim);
    LayerNorm(std::string name, std::vector<Ptr<NPUTensor>> weights);
    // LayerNorm(SimulationConfig config,
    //                      std::string name,
    //                      std::vector<uint32_t> weight_tensors);
    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

   private:
    std::vector<uint32_t> _weight_dim;

    uint32_t _prod_batches;
    uint32_t _prod_weight_dim;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(uint32_t N);
    uint32_t sram_size_needed();
};