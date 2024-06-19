#pragma once

#include <vector>

#include "BatchedRequest.h"
#include "Common.h"
#include "Logger.h"
#include "Model.h"
#include "Stat.h"
#include "operations/Operation.h"
#include "tensor/BTensor.h"

class ModelProgram {
   public:
    ModelProgram(std::shared_ptr<Model> model, Ptr<BatchedRequest> batched_request);
    void init_program();
    Ptr<Operation> add_op(Ptr<Operation> op);
    std::vector<Ptr<BTensor>> get_outputs(Ptr<Operation> op, std::vector<Ptr<BTensor>> inputs);

    bool check_exist_in_executable(uint32_t op_id);
    void finish_operation(uint32_t id);
    void find_executable_node(Ptr<BTensor> tensor);
    std::vector<std::shared_ptr<Operation>> get_executable_operations() {
        return _executable_operations;
    }
    bool check_finish();
    std::vector<OperationStat> list_operation_stat();
    void finish_operation_tile(Tile& tile);
    void log();

    // todo: from BatchedRequest
    std::shared_ptr<Model> _model;
    std::shared_ptr<BatchedRequest> _breq;
    robin_hood::unordered_map<uint32_t, Ptr<Operation>> _op_map;

    // todo: Constructor
    // std::map<uint32_t, Ptr<Operation>> _operation_map;
    std::map<uint32_t, Ptr<BTensor>> _tensor_map;
    std::vector<std::shared_ptr<Operation>> _executable_operations;

    // bool check_exist_in_executable(uint32_t id);

    std::vector<Ptr<BTensor>> attn_block(uint32_t layer, std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> fused_attn_block(uint32_t layer, std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> fused_pim_attn_block(uint32_t layer,
                                                   std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> pim_attn_block(uint32_t layer, std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> ffn_block(uint32_t layer, std::vector<Ptr<BTensor>> inputs);

    Ptr<Operation> block_layer_norm(uint32_t layer);
    Ptr<Operation> block_QKV_gen(uint32_t layer);
    Ptr<Operation> block_QKV_split(uint32_t layer, uint32_t unit, uint32_t dim_idx);
    Ptr<Operation> test_block_gelu(uint32_t layer);
    Ptr<Operation> test_block_add(uint32_t layer);
    Ptr<Operation> test_block_softmax(uint32_t layer);
    // std::shared_ptr<Operation> block_QKV_gen(uint32_t layer);
    // std::shared_ptr<Operation> block_QKV_split(uint32_t layer, uint32_t unit, uint32_t dim);
    Ptr<Operation> block_gemv_softmax(std::string prefix);
    Ptr<Operation> block_gemv_add(std::string prefix);
    Ptr<Operation> block_fused_mha(std::string prefix);
};