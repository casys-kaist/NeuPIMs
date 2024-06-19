#include "ModelProgram.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "Common.h"
#include "Model.h"
#include "SimulationConfig.h"
#include "Stat.h"
#include "tensor/BTensor.h"
#include "tensor/NPUTensor.h"
#include "tensor/NPUTensorInner.h"
#include "tensor/PIMTensor.h"

ModelProgram::ModelProgram(Ptr<Model> model, Ptr<BatchedRequest> batched_request)
    : _model(model), _breq(batched_request) {
    this->init_program();
}

void ModelProgram::init_program() {
    bool end_to_end = false;

    /* end-to-end GPT program */
    if (end_to_end) {
        bool npu_program = Config::global_config.run_mode == RunMode::NPU_ONLY;
        bool fused = Config::global_config.kernel_fusion;

        auto N = _breq->get_num_rows();
        auto E = Config::global_config.model_n_embd;

        std::vector<uint32_t> input_dim{N, E};
        auto input = std::make_shared<NPUTensor>("input", input_dim, NPUTensorBufType::ACT, true);
        std::vector<Ptr<BTensor>> inputs{input};
        for (uint32_t layer_idx = 0; layer_idx < Config::global_config.model_n_layer; ++layer_idx) {
            if (npu_program) {
                if (fused)
                    inputs = fused_attn_block(layer_idx, inputs);
                else
                    inputs = attn_block(layer_idx, inputs);
            } else {
                if (fused)
                    inputs = fused_pim_attn_block(layer_idx, inputs);
                else
                    inputs = pim_attn_block(layer_idx, inputs);
            }
            inputs = ffn_block(layer_idx, inputs);
        }
        find_executable_node(input);
    }

    /* only Multi-head attention layer */
    else {
        // all Q, K, V are batched

        spdlog::info(">>> Initialize Model Program <<<");
        Ptr<NPUTensor> query;
        std::vector<Ptr<BTensor>> inputs;

        bool npu_program = Config::global_config.run_mode == RunMode::NPU_ONLY;  //
        int batch_size = _breq->_reqs.size();
        spdlog::info("----------");
        spdlog::info(">>>logging<<<");
        spdlog::info("*** batch size: {}", batch_size);
        spdlog::info("*** K_cache.size(num_layers): {}", _breq->_reqs[0]->K_cache.size());

        for (int i = 0; i < Config::global_config.model_n_layer; ++i) {
            uint32_t num_heads = Config::global_config.model_n_head;
            uint32_t dk = Config::global_config.model_n_embd / num_heads;  // 64;

            std::vector<Ptr<BTensor>> querys;
            std::vector<Ptr<BTensor>> keys;
            std::vector<Ptr<BTensor>> values;

            // >>> gsheo
            // batch of q,k,vs
            // fixme: query size is always 1, while it can be l at initiation phase.
            for (int j = 0; j < batch_size; j++) {
                /* - [] todo: change query to real query from gkv gen */
                Ptr<InferRequest> request = _breq->_reqs[j];
                int lj = request->input_size;
                lj = 1;
                // if (j < 8 || j >= 16) {
                //     request->is_initiated = true;
                //     lj = 1;
                // }

                query = std::make_shared<NPUTensor>(
                    "query", std::vector<uint32_t>{num_heads, lj, dk}, NPUTensorBufType::ACT, true);
                querys.push_back(query);

                /* key/value cache */
                keys.push_back(request->K_cache[i]);
                values.push_back(request->V_cache[i]);
            }

            auto prefix = name_gen(LAYER(i), BlockType::Attention);
            // this process is the same as applying attn_block()
            if (npu_program) {
                /* Fused multi-head attention */
                std::vector<Ptr<BTensor>> mha_npu_inputs = querys;
                keys.insert(keys.end(), values.begin(), values.end());
                mha_npu_inputs.insert(mha_npu_inputs.end(), keys.begin(),
                                      keys.end());  // querys, keys, values

                auto fused_mha = block_fused_mha(prefix);
                inputs = get_outputs(fused_mha, mha_npu_inputs);
            } else {
                /* gemv + softmax */
                std::vector<Ptr<BTensor>> mha_pim_inputs = querys;
                mha_pim_inputs.insert(mha_pim_inputs.end(), keys.begin(),
                                      keys.end());  // querys, keys
                // auto gemv_softmax = block_gemv_softmax(prefix);
                // inputs = get_outputs(gemv_softmax, mha_pim_inputs);

                /* gemv + add */
                // auto gemv_add = block_gemv_add(prefix);
                // inputs = get_outputs(gemv_add, inputs);

                auto logit_softmax = add_op(std::make_shared<NeuPIMSLogitSoftmax>(
                    name_gen(prefix, BlockType::Attention, OperationType::NeuPIMSLogitSoftmax)));
                inputs = get_outputs(logit_softmax, mha_pim_inputs);

                /* pim_gemv + add */
                inputs.insert(inputs.end(), values.begin(), values.end());  // logits, values

                auto attend = add_op(std::make_shared<NeuPIMSAttend>(
                    name_gen(prefix, BlockType::Attention, OperationType::NeuPIMSAttend)));
                inputs = get_outputs(attend, inputs);
            }
        }
        find_executable_node(query);
    }
}

Ptr<Operation> ModelProgram::add_op(std::shared_ptr<Operation> op) {
    // spdlog::info("operation {} added. add_op", op->get_name());
    _op_map[op->get_id()] = op;
    return op;
}

std::vector<Ptr<BTensor>> ModelProgram::get_outputs(Ptr<Operation> op,
                                                    std::vector<Ptr<BTensor>> inputs) {
    return op->get_outputs(inputs);
}

void ModelProgram::find_executable_node(Ptr<BTensor> tensor) {
    for (auto op : tensor->get_child_nodes()) {
        // spdlog::info("initializing operation {} ...", op->get_name());
        if (op->check_executable()) {
            _executable_operations.push_back(op);
        }
    }
}

bool ModelProgram::check_exist_in_executable(uint32_t op_id) {
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        if (op_id == (*iter)->get_id()) {
            return true;
        }
    }
    return false;
}

void ModelProgram::finish_operation(uint32_t id) {
    _op_map[id]->set_finish();
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        spdlog::info("iterating operation: {}", (*iter)->get_name());
        if (id == (*iter)->get_id()) {
            spdlog::info("erasing operation: {}", (*iter)->get_name());
            _executable_operations.erase(iter);
            break;
        }
    }

    for (auto op : _op_map[id]->get_child_nodes()) {
        if (op->check_executable() && !check_exist_in_executable(op->get_id())) {
            _executable_operations.push_back(op);
        }
    }
}

bool ModelProgram::check_finish() {
    bool finish = true;
    for (auto const &[key, val] : _op_map) {
        finish = finish && val->check_finish();
    }

    return finish;
}

std::vector<OperationStat> ModelProgram::list_operation_stat() {
    std::vector<OperationStat> ret;
    for (auto &[key, val] : _op_map) {
        ret.push_back(val->get_stat());
    }

    return ret;
}

void ModelProgram::finish_operation_tile(Tile &tile) {
    _op_map[tile.operation_id]->reduce_tile(tile);
}

/**
 * logger function for ModelProgram
 * TODO: log file name is tentative. think of fname rule
 */
void ModelProgram::log() {
    std::string fname = Config::global_config.log_dir + "/tmp_log_file";
    Logger::log(list_operation_stat(), fname);
}

/* PIMgemvSoftmax, PIMgemvAdd */
std::vector<Ptr<BTensor>> ModelProgram::fused_pim_attn_block(uint32_t layer,
                                                             std::vector<Ptr<BTensor>> inputs) {
    auto prefix = name_gen(LAYER(layer), BlockType::Attention);
    auto res_buf = inputs[0];

    // (N,E) -> (N,E)
    auto ln1 = block_layer_norm(layer);
    inputs = get_outputs(ln1, inputs);

    // (N,E) x (E,3E)
    auto qkv_gen = block_QKV_gen(layer);
    inputs = get_outputs(qkv_gen, inputs);

    // divide Batches into predetermined length
    // if not initialized
    //  returns 1
    // else,
    //  returns initial token length
    // (N,E) -> (t1,E),(t2,E), ..., (tn,E)
    auto num_rows_breakdown = _breq->get_num_rows_breakdown();
    auto split = add_op(std::make_shared<Split>(name_gen(prefix, OperationType::BatchSplit),
                                                num_rows_breakdown, 0));
    auto qkv_per_reqs = get_outputs(split, inputs);

    // std::vector<Ptr<NPUTensor>> qkv_per_reqs =
    //     std::static_pointer_cast<NPUTensor>(inputs[0])->split_by_row(num_rows_breakdown);
    auto num_requests = qkv_per_reqs.size();

    std::vector<Ptr<BTensor>> querys;
    std::vector<Ptr<BTensor>> keys;
    std::vector<Ptr<BTensor>> values;

    //  iterate by batch
    std::vector<Ptr<BTensor>> qkv_results;
    for (int request_index = 0; request_index < num_requests; ++request_index) {
        std::vector<Ptr<BTensor>> qkv(qkv_per_reqs.begin() + request_index,
                                      qkv_per_reqs.begin() + request_index + 1);
        std::vector<Ptr<BTensor>> qkv_cached;

        // (1,3E),KeyCache(nh,dk,T),ValueCache(nh,T,dk) ->
        //  (nh,1,dk)[NPUTensor2D],(nh,dk,T+1)[NPUTensorKV],(nh,T+1,dk)[NPUTensorKV]
        auto split = add_op(std::make_shared<SplitDecoding>(
            name_gen(prefix, OperationType::QKVSplit, std::to_string(request_index)),
            _breq->get_cache(layer, request_index), _breq->is_initiated(request_index)));
        qkv_cached = get_outputs(split, qkv);

        querys.push_back(qkv_cached[0]);
        keys.push_back(qkv_cached[1]);
        values.push_back(qkv_cached[2]);
    }

    /* pim_gemv + softmax */
    querys.insert(querys.end(), keys.begin(),
                  keys.end());  // querys,keys
    // auto gemv_softmax = block_gemv_softmax(prefix);
    // auto ls = get_outputs(gemv_softmax, querys);

    auto logit_softmax = add_op(std::make_shared<NeuPIMSLogitSoftmax>(
        name_gen(prefix, BlockType::Attention, OperationType::NeuPIMSLogitSoftmax)));
    auto ls = get_outputs(logit_softmax, querys);

    /* pim_gemv + add */
    ls.insert(ls.end(), values.begin(), values.end());  // logits, values
    // auto gemv_add = block_gemv_add(prefix);
    // auto a = get_outputs(gemv_add, ls);
    auto attend = add_op(std::make_shared<NeuPIMSAttend>(
        name_gen(prefix, BlockType::Attention, OperationType::NeuPIMSAttend)));
    auto a = get_outputs(attend, ls);

    for (int request_index = 0; request_index < num_requests; ++request_index) {
        // (nh,{1,T},dk) -> ({1,T},E)
        std::vector<uint32_t> reshape_result = {a[request_index]->get_dims()[1],
                                                Config::global_config.model_n_embd};
        auto reshape = add_op(std::make_shared<Reshape>(
            name_gen(prefix, OperationType::AReshape, std::to_string(request_index)),
            reshape_result));
        auto result = get_outputs(reshape, {a[request_index]});

        qkv_results.push_back(result[0]);
    }

    // collect batches
    // (T1,E),(T2,E), ... ,(Tn,E) -> (N,E)
    auto concat = add_op(std::make_shared<Concat>(name_gen(prefix, OperationType::BatchConcat), 0));
    inputs = get_outputs(concat, qkv_results);

    auto projection = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::Projection),
        _model->get_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = get_outputs(projection, inputs);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);

    return inputs;
}

/* gemv, softmax, PIMgemvAdd */
std::vector<Ptr<BTensor>> ModelProgram::pim_attn_block(uint32_t layer,
                                                       std::vector<Ptr<BTensor>> inputs) {
    auto prefix = name_gen(LAYER(layer), BlockType::Attention);
    auto res_buf = inputs[0];

    // (N,E) -> (N,E)
    auto ln1 = block_layer_norm(layer);
    inputs = get_outputs(ln1, inputs);

    // (N,E) x (E,3E)
    auto qkv_gen = block_QKV_gen(layer);
    inputs = get_outputs(qkv_gen, inputs);

    // divide Batches into predetermined length
    // if not initialized
    //  returns 1
    // else,
    //  returns initial token length
    // (N,E) -> (t1,E),(t2,E), ..., (tn,E)
    auto num_rows_breakdown = _breq->get_num_rows_breakdown();
    auto split = add_op(std::make_shared<Split>(name_gen(prefix, OperationType::BatchSplit),
                                                num_rows_breakdown, 0));
    auto qkv_per_reqs = get_outputs(split, inputs);

    // std::vector<Ptr<NPUTensor>> qkv_per_reqs =
    //     std::static_pointer_cast<NPUTensor>(inputs[0])->split_by_row(num_rows_breakdown);
    auto num_requests = qkv_per_reqs.size();

    std::vector<Ptr<BTensor>> querys;
    std::vector<Ptr<BTensor>> keys;
    std::vector<Ptr<BTensor>> values;

    //  iterate by batch
    std::vector<Ptr<BTensor>> qkv_results;
    for (int request_index = 0; request_index < num_requests; ++request_index) {
        std::vector<Ptr<BTensor>> qkv(qkv_per_reqs.begin() + request_index,
                                      qkv_per_reqs.begin() + request_index + 1);
        std::vector<Ptr<BTensor>> qkv_cached;

        // (1,3E),KeyCache(nh,dk,T),ValueCache(nh,T,dk) ->
        //  (nh,1,dk)[NPUTensor2D],(nh,dk,T+1)[NPUTensorKV],(nh,T+1,dk)[NPUTensorKV]
        auto split = add_op(std::make_shared<SplitDecoding>(
            name_gen(prefix, OperationType::QKVSplit, std::to_string(request_index)),
            _breq->get_cache(layer, request_index), _breq->is_initiated(request_index)));
        qkv_cached = get_outputs(split, qkv);

        querys.push_back(qkv_cached[0]);
        keys.push_back(qkv_cached[1]);
        values.push_back(qkv_cached[2]);
    }

    /* pim_gemv (QxK) */
    querys.insert(querys.end(), keys.begin(),
                  keys.end());  // querys,keys

    auto pim_gemv = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::PIMGEMV)));
    // auto pim_gemv = block_gemv_softmax(prefix);
    auto l = get_outputs(pim_gemv, querys);

    /* softmax */
    auto logit_softmax =
        add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
    auto ls = get_outputs(logit_softmax, l);

    /* pim_gemv + add */
    ls.insert(ls.end(), values.begin(), values.end());  // logits, values
    auto gemv_add = block_gemv_add(prefix);
    auto a = get_outputs(gemv_add, ls);

    for (int request_index = 0; request_index < num_requests; ++request_index) {
        // (nh,{1,T},dk) -> ({1,T},E)
        std::vector<uint32_t> reshape_result = {a[request_index]->get_dims()[1],
                                                Config::global_config.model_n_embd};
        auto reshape = add_op(std::make_shared<Reshape>(
            name_gen(prefix, OperationType::AReshape, std::to_string(request_index)),
            reshape_result));
        auto result = get_outputs(reshape, {a[request_index]});

        qkv_results.push_back(result[0]);
    }

    // collect batches
    // (T1,E),(T2,E), ... ,(Tn,E) -> (N,E)
    auto concat = add_op(std::make_shared<Concat>(name_gen(prefix, OperationType::BatchConcat), 0));
    inputs = get_outputs(concat, qkv_results);

    auto projection = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::Projection),
        _model->get_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = get_outputs(projection, inputs);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);

    return inputs;
}

/**
 * attn_block
 *  input:
 *      (N,E) 2d matrix.
 *      Note that N is not the number of batch size, but sum of n batches token len.
 *      This means, N = T1 + T2 + ... + Tn
 *  output:
 *      (N,E) 2d matrix.
 */
std::vector<Ptr<BTensor>> ModelProgram::attn_block(uint32_t layer,
                                                   std::vector<Ptr<BTensor>> inputs) {
    auto prefix = name_gen(LAYER(layer), BlockType::Attention);
    auto res_buf = inputs[0];

    // (N,E) -> (N,E)
    auto ln1 = block_layer_norm(layer);
    inputs = get_outputs(ln1, inputs);

    // (N,E) x (E,3E)
    auto qkv_gen = block_QKV_gen(layer);
    inputs = get_outputs(qkv_gen, inputs);

    // divide Batches into predetermined length
    // if not initialized
    //  returns 1
    // else,
    //  returns initial token length
    // (N,E) -> (t1,E),(t2,E), ..., (tn,E)
    auto num_rows_breakdown = _breq->get_num_rows_breakdown();
    auto split = add_op(std::make_shared<Split>(name_gen(prefix, OperationType::BatchSplit),
                                                num_rows_breakdown, 0));
    auto qkv_per_reqs = get_outputs(split, inputs);

    // std::vector<Ptr<NPUTensor>> qkv_per_reqs =
    //     std::static_pointer_cast<NPUTensor>(inputs[0])->split_by_row(num_rows_breakdown);
    auto num_requests = qkv_per_reqs.size();
    //  iterate by batch
    std::vector<Ptr<BTensor>> qkv_results;
    for (int request_index = 0; request_index < num_requests; ++request_index) {
        std::vector<Ptr<BTensor>> qkv(qkv_per_reqs.begin() + request_index,
                                      qkv_per_reqs.begin() + request_index + 1);
        std::vector<Ptr<BTensor>> qkv_cached;

        // (1,3E),KeyCache(nh,dk,T),ValueCache(nh,T,dk) ->
        //  (nh,1,dk)[NPUTensor2D],(nh,dk,T+1)[NPUTensorKV],(nh,T+1,dk)[NPUTensorKV]
        auto split = add_op(std::make_shared<SplitDecoding>(
            name_gen(prefix, OperationType::QKVSplit, std::to_string(request_index)),
            _breq->get_cache(layer, request_index), _breq->is_initiated(request_index)));
        qkv_cached = get_outputs(split, qkv);

        // (nh,{1,T},dk)@(nh,dk,{T+1,T}) -> (nh,{1,T},{T+1,T})
        auto l_mm = add_op(std::make_shared<MatMul>(
            name_gen(prefix, OperationType::QKMatMul, std::to_string(request_index))));
        auto l = get_outputs(l_mm, std::vector<Ptr<BTensor>>{qkv_cached[0], qkv_cached[1]});

        auto ls_sm = add_op(std::make_shared<Softmax>(
            name_gen(prefix, OperationType::SoftMax, std::to_string(request_index))));
        auto ls = get_outputs(ls_sm, l);

        // (nh,{1,T},{T+1,T})@(nh,{T+1,T},dk) -> (nh,{1,T},dk)
        auto a_mm = add_op(std::make_shared<MatMul>(
            name_gen(prefix, OperationType::LsVMatMul, std::to_string(request_index))));
        auto a = get_outputs(a_mm, std::vector<Ptr<BTensor>>{ls[0], qkv_cached[2]});

        // (nh,{1,T},dk) -> ({1,T},E)
        std::vector<uint32_t> reshape_result = {a[0]->get_dims()[1],
                                                Config::global_config.model_n_embd};
        auto reshape = add_op(std::make_shared<Reshape>(
            name_gen(prefix, OperationType::AReshape, std::to_string(request_index)),
            reshape_result));
        auto result = get_outputs(reshape, a);

        qkv_results.push_back(result[0]);
    }

    // collect batches
    // (T1,E),(T2,E), ... ,(Tn,E) -> (N,E)
    auto concat = add_op(std::make_shared<Concat>(name_gen(prefix, OperationType::BatchConcat), 0));
    inputs = get_outputs(concat, qkv_results);

    auto projection = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::Projection),
        _model->get_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = get_outputs(projection, inputs);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);

    return inputs;
}

/**
 * attn_block
 *  input:
 *      (N,E) 2d matrix.
 *      Note that N is not the number of batch size, but sum of n batches token len.
 *      This means, N = T1 + T2 + ... + Tn
 *  output:
 *      (N,E) 2d matrix.
 */
std::vector<Ptr<BTensor>> ModelProgram::fused_attn_block(uint32_t layer,
                                                         std::vector<Ptr<BTensor>> inputs) {
    auto prefix = name_gen(LAYER(layer), BlockType::Attention);
    auto res_buf = inputs[0];

    // (N,E) -> (N,E)
    auto ln1 = block_layer_norm(layer);
    inputs = get_outputs(ln1, inputs);

    // (N,E) x (E,3E)
    auto qkv_gen = block_QKV_gen(layer);
    inputs = get_outputs(qkv_gen, inputs);

    // divide Batches into predetermined length
    // if not initialized
    //  returns 1
    // else,
    //  returns initial token length
    // (N,E) -> (t1,E),(t2,E), ..., (tn,E)
    // auto num_rows_breakdown = _breq->get_num_rows_breakdown();
    // auto split = add_op(std::make_shared<Split>(name_gen(prefix, OperationType::BatchSplit),
    //                                             num_rows_breakdown, 0));
    // auto qkv_per_reqs = get_outputs(split, inputs);

    // std::vector<Ptr<BTensor>> querys;
    // std::vector<Ptr<BTensor>> keys;
    // std::vector<Ptr<BTensor>> values;

    // for (int request_index = 0; request_index < _breq->get_num_reqs(); ++request_index) {
    //     std::vector<Ptr<BTensor>> qkv(qkv_per_reqs.begin() + request_index,
    //                                   qkv_per_reqs.begin() + request_index + 1);
    //     std::vector<Ptr<BTensor>> qkv_cached;

    //     // (1,3E),KeyCache(nh,dk,T),ValueCache(nh,T,dk) ->
    //     //  (nh,1,dk)[NPUTensor2D],(nh,dk,T+1)[NPUTensorKV],(nh,T+1,dk)[NPUTensorKV]
    //     auto split = add_op(std::make_shared<SplitDecoding>(
    //         name_gen(prefix, OperationType::QKVSplit, std::to_string(request_index)),
    //         _breq->get_cache(layer, request_index), _breq->is_initiated(request_index)));
    //     qkv_cached = get_outputs(split, qkv);

    //     querys.push_back(qkv_cached[0]);
    //     keys.push_back(qkv_cached[1]);
    //     values.push_back(qkv_cached[2]);
    // }

    // /* Fused multi-head attention */
    // std::vector<Ptr<BTensor>> mha_npu_inputs = querys;
    // keys.insert(keys.end(), values.begin(), values.end());
    // mha_npu_inputs.insert(mha_npu_inputs.end(), keys.begin(),
    //                       keys.end());  // querys, keys, values

    // auto fused_mha = block_fused_mha(prefix);
    // inputs = get_outputs(fused_mha, mha_npu_inputs);

    // std::vector<Ptr<BTensor>> qkv_results;
    // for (int request_index = 0; request_index < _breq->get_num_reqs(); ++request_index) {
    //     std::vector<uint32_t> reshape_result = {inputs[request_index]->get_dims()[1],
    //                                             Config::global_config.model_n_embd};
    //     auto reshape = add_op(std::make_shared<Reshape>(
    //         name_gen(prefix, OperationType::AReshape, std::to_string(request_index)),
    //         reshape_result));
    //     auto result = get_outputs(reshape, {inputs[request_index]});

    //     qkv_results.push_back(result[0]);
    // }

    // // collect batches
    // // (T1,E),(T2,E), ... ,(Tn,E) -> (N,E)
    // auto concat = add_op(std::make_shared<Concat>(name_gen(prefix, OperationType::BatchConcat),
    // 0)); inputs = get_outputs(concat, qkv_results);

    // auto projection = add_op(std::make_shared<MatMul>(
    //     name_gen(prefix, OperationType::Projection),
    //     _model->get_params(layer, BlockType::Attention, OperationType::Projection)));
    // inputs = get_outputs(projection, inputs);

    // auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    // inputs.push_back(res_buf);
    // inputs = get_outputs(residual, inputs);

    return inputs;
}

/**
 * ffn_block
 *  input: n,E batched input
 *  output: n,E batched output
 */
std::vector<Ptr<BTensor>> ModelProgram::ffn_block(uint32_t layer,
                                                  std::vector<Ptr<BTensor>> inputs) {
    auto res_buf = inputs[0];
    std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
    // create operations
    auto ln = add_op(std::make_shared<LayerNorm>(
        name_gen(prefix, OperationType::LayerNorm),
        _model->get_params(layer, BlockType::FeedForward, OperationType::LayerNorm)));
    inputs = get_outputs(ln, inputs);

    auto fc1 = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::FullyConnected1),
        _model->get_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
    inputs = get_outputs(fc1, inputs);

    auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
    inputs = get_outputs(gelu, inputs);

    auto fc2 = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::FullyConnected2),
        _model->get_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
    inputs = get_outputs(fc2, inputs);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);

    return inputs;
}

Ptr<Operation> ModelProgram::block_layer_norm(uint32_t layer) {
    return add_op(std::make_shared<LayerNorm>(
        name_gen(LAYER(layer), BlockType::Attention, OperationType::LayerNorm),
        _model->get_params(layer, BlockType::Attention, OperationType::LayerNorm)));
}

Ptr<Operation> ModelProgram::block_QKV_gen(uint32_t layer) {
    return add_op(std::make_shared<MatMul>(
        name_gen(LAYER(layer), BlockType::Attention, OperationType::QKVGen),
        _model->get_params(layer, BlockType::Attention, OperationType::QKVGen)));
}

Ptr<Operation> ModelProgram::block_QKV_split(uint32_t layer, uint32_t unit, uint32_t dim_idx) {
    // return add_op(std::make_shared<Split>(
    //     name_gen(LAYER(layer), BlockType::Attention, OperationType::QKVSplit), unit,
    //     dim_idx));
}

Ptr<Operation> ModelProgram::test_block_gelu(uint32_t layer) {
    return add_op(
        std::make_shared<Gelu>(name_gen(LAYER(layer), BlockType::Attention, OperationType::Gelu)));
}

Ptr<Operation> ModelProgram::test_block_add(uint32_t layer) {
    return add_op(std::make_shared<Add>(
        name_gen(LAYER(layer), BlockType::Attention, OperationType::Residual)));
}

Ptr<Operation> ModelProgram::test_block_softmax(uint32_t layer) {
    return add_op(std::make_shared<Softmax>(
        name_gen(LAYER(layer), BlockType::Attention, OperationType::SoftMax)));
}

// std::shared_ptr<Operation> ModelProgram::block_QKV_gen(uint32_t layer) {
//     return add_op(std::make_shared<MatMul>(
//         name_gen(LAYER(layer), BlockType::Attention, OperationType::QKVGen),
//         _model->get_param(layer, BlockType::Attention, OperationType::QKVGen)));
// }

// std::shared_ptr<Operation> ModelProgram::block_QKV_split(uint32_t layer, uint32_t unit,
//                                                          uint32_t dim) {
//     return add_op(std::make_shared<Split>(
//         name_gen(LAYER(layer), BlockType::Attention, OperationType::QKVSplit), unit, dim));
// }

Ptr<Operation> ModelProgram::block_gemv_softmax(std::string prefix) {
    return add_op(std::make_shared<PIMGEMVSoftmax>(
        name_gen(prefix, BlockType::Attention, OperationType::PIMGEMVSoftmax)));
}

Ptr<Operation> ModelProgram::block_gemv_add(std::string prefix) {
    return add_op(std::make_shared<PIMGEMVAdd>(
        name_gen(prefix, BlockType::Attention, OperationType::PIMGEMVAdd)));
}

Ptr<Operation> ModelProgram::block_fused_mha(std::string prefix) {
    return add_op(std::make_shared<FusedMHA>(
        name_gen(prefix, BlockType::Attention, OperationType::FusedMHA)));
}
