#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

class MatMul : public Operation {
   public:
    // MatMul(std::string name, std::vector<uint32_t> weight_dim);
    MatMul(std::string name, std::vector<Ptr<NPUTensor>> weights);
    MatMul(std::string name);

    // MatMul(std::string name,
    //                std::vector<uint32_t> weight_tensors);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);
    void set_transposed() { _is_transposed = true; }

    // todo: add attributes
    // currently, values below are dummy.
   private:
    uint32_t _alpha;
    uint32_t _beta;
    bool _transA;
    bool _transB;

    bool _is_transposed;

    bool _is_gemv;

    // b0, b1, ...
    uint32_t _prod_batches;
    // 3 dimensions, only for n, k, m
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(uint32_t B, uint32_t N, uint32_t K, uint32_t M, bool should_store);
    uint32_t sram_size_needed();
};