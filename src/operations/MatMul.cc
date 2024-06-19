#include "MatMul.h"

// MatMul::MatMul(std::string name, std::vector<uint32_t> weight_dim) : Operation(name) {
//     assert(weight_dim.size() == 2);
//     _inputs.resize(3);
//     // initialize weight and bias tensors
//     // wait for a single input
//     _inputs[1] = std::make_shared<BTensor>(_name + "_weight", weight_dim, true);
//     auto bias_dim = std::vector<uint32_t>(weight_dim.begin() + 1, weight_dim.end());
//     _inputs[2] = std::make_shared<BTensor>(_name + "_bias", bias_dim, true);
// }

/**
 * Initialize MatMul
 *  - if it has bias, weights has two pointer -> resize input to 2
 *  - else if one -> resize input to 2 but not the case in GPT2
 *  - Case where MatMul doesn't have weight, it is initialized only with name. (Below Constructor)
 */
MatMul::MatMul(std::string name, std::vector<Ptr<NPUTensor>> weights) : Operation(name) {
    ast(weights.size() == 2 || weights.size() == 1);
    if (weights.size() == 2) {
        // assert(weights.size() == 2);
        _inputs.resize(3);

        _inputs[1] = weights[0];
        _inputs[2] = weights[1];
    } else if (weights.size() == 1) {
        _inputs.resize(2);
        _inputs[1] = weights[0];
    }

    // xxx: currently, it always shows better performance if _is_transposed is true.
    _is_transposed = true;
}

MatMul::MatMul(std::string name) : Operation(name) { _inputs.resize(2); }

/**
 * function executing MatMul
 *  inputs:
 *      If weight exist, only has a single tensor. else, has two tensors
 *      Case where MatMul getting a single tensor input is calculating QKV, proj, MLP layer
 *      It takes batched input, (T1+T2+...+Tn,E) calculated with weight(E,3E) and bias(3E).
 *      else, it gets two tensor inputs when calculating q*k, s*v.
 *      It gets 3D tensor, (n,T,dk) * (n,dk,T) resulting (n,T,T)
 *
 *  output: output tensor not produced
 */
std::vector<Ptr<BTensor>> MatMul::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _outputs.resize(1);

    assert((inputs.size() == 2 && _inputs.size() == 2) ||
           (inputs.size() == 1 && _inputs.size() == 3));

    for (size_t i = 0; i < inputs.size(); ++i) {
        _inputs[i] = inputs[i];
        spdlog::info("MatMul input idx: {} / input sz: {}", i, inputs[i]->get_dims());
    }

    // validate input dimension.
    auto input0_dims = _inputs[0]->get_dims();
    auto input1_dims = _inputs[1]->get_dims();
    assert(*input0_dims.rbegin() == *(input1_dims.rbegin() + 1));

    auto larger_dim = input0_dims.size() > input1_dims.size() ? input0_dims : input1_dims;
    std::vector<uint32_t> output_dims(larger_dim.begin(), larger_dim.end());
    *(output_dims.rbegin() + 1) =
        *(input0_dims.rbegin() + 1);                // Set (M, x) in matmul (M, K) x (K, N).
    *output_dims.rbegin() = *input1_dims.rbegin();  // Set (x, N) in matmul (M, K) x (K, N)
    spdlog::info("MatMul output sz: {}", output_dims);

    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", output_dims, NPUTensorBufType::ACT, false);

    // spdlog::info("[{}] input0 : {}  / input1: {}", _name, input0_dims, input1_dims);

    calculate_loops();
    initialize_tiles();

    spdlog::info("input0 : {}  / input1: {} / output0 : {}", input0_dims, input1_dims, output_dims);
    spdlog::info("outer loop : {} / inner loop : {}", _outer_loop, _inner_loop);

    return _outputs;
}

void MatMul::initialize_tiles() {
    // Here, B does not refer to batch_size,
    // but rather to the outer dimensions of the tensor in matmul,
    // such as b*h in [b, h, l, d_k].
    for (uint32_t B = 0; B < _prod_batches; ++B) {
        for (uint32_t M = 0; M < _outer_loop[0]; ++M) {
            for (uint32_t N = 0; N < _outer_loop[2]; ++N) {
                for (uint32_t K = 0; K < _outer_loop[1]; ++K) {
                    _tiles.push_back(initialize_instructions(B, M, K, N, K + 1 == _outer_loop[1]));
                }
            }
        }
    }
}

Tile MatMul::initialize_instructions(uint32_t B, uint32_t M, uint32_t K, uint32_t N,
                                     bool should_store) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = B,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,
    };

    addr_type act_addr = 0;
    addr_type wgt_addr = 0;

    // base on the inner loop, initialize instructions
    // _inner_loop means L2 tile size
    auto m_inner = _inner_loop[0];  // M-axis L2 tile size
    auto k_inner = _inner_loop[1];  // K-axis L2 tile size
    auto n_inner = _inner_loop[2];  // N-axis L2 tile size

    auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
    auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
    auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

    // load tile to SPM in order ACT / WGT
    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_weight_base = SPAD_BASE + m_inner * k_inner * _config.precision;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.core_width;

    auto activation_tensor = std::static_pointer_cast<NPUTensor>(_inputs[0]);
    auto weight_tensor = std::static_pointer_cast<NPUTensor>(_inputs[1]);
    auto output_tensor = std::static_pointer_cast<NPUTensor>(_outputs[0]);

    if (_is_transposed) {
        std::swap(activation_tensor, weight_tensor);
        activation_tensor->set_transposed();
        weight_tensor->set_transposed();
    }

    // In MHA, calculating logit score or a uses 3D * 3D matrix multiplications.
    //  for exmaple, (n,t,dk)@(n,dk,t)
    // in this case, batch index is needed for memory access.
    auto batch_index = std::vector<uint32_t>();
    if (_inputs[0]->get_dims().size() == 3) {
        batch_index.push_back(B);
    }

    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;

    // -- bias --
    // if      input size is 2, no need for bias initialization
    //         (_inputs[2] x)
    // else if input size is 3, and is not accumulation tile, create activation
    // region using bias load
    if (_inputs.size() == 3 && K == 0) {
        auto bias_tensor = std::static_pointer_cast<NPUTensor>(_inputs[2]);
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
            // n_inner_offset: L1 tile start index in each L2 tile
            std::vector<addr_type> bias_addrs;
            for (uint32_t n_loop = 0; n_loop < loop_size; ++n_loop) {
                // get address by get_addr
                auto bias_addr = bias_tensor->get_addr({n_outer_offset + n_inner_offset + n_loop});
                if (bias_addr != GARBAGE_ADDR) {
                    bias_addrs.push_back(bias_addr);
                }
            }
            if (bias_addrs.size() == 0) {
                spdlog::info("zero load for activation n: {} {} / bias tensor dim: {}",
                             n_outer_offset, n_inner_offset, bias_tensor->get_dims());
                assert(0);
            } else {
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_accumulation_base + n_inner_offset * _config.precision,
                    .size = (uint32_t)bias_addrs.size() * _config.precision,  // assume broadcasting
                                                                              // bias is
                    // available inside the npu
                    .src_addrs = std::move(bias_addrs),
                    .operand_id = _INPUT_OPERAND + 2,
                });
            }
        }
    }

    // ex: [2, 12, seq_len, d_k] B=14, emb_dim_size=2 -> return [1, 1]
    // inner_offset is the index from the tensor base address to the start point of the L1 tile.
    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
        for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
            for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner;
                 m_inner_offset += loop_size) {
                // SRAM act L1 tile offset
                addr_type sram_activation_offset =
                    sram_activation_base +
                    (m_inner_offset * k_inner + k_inner_offset) * _config.precision;
                // SRAM wgt L1 tile offset
                addr_type sram_weight_offset =
                    sram_weight_base +
                    (k_inner_offset * n_inner + n_inner_offset) * _config.precision;
                // SRAM out L1 tile offset
                addr_type sram_accumulation_offset =
                    sram_accumulation_base +
                    (m_inner_offset * n_inner + n_inner_offset) * _config.precision;

                // -- activation --
                if (n_inner_offset == 0) {
                    tile_m = 0;
                    tile_k = 0;
                    // During the n_inner tile iterations (to prevent duplication),
                    // add the MOVIN instruction only in the first inner loop.
                    std::vector<addr_type> activation_addrs;
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                        for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                            std::vector<uint32_t> activation_indexes(batch_index);
                            activation_indexes.push_back(m_outer_offset + m_inner_offset + m_loop);
                            activation_indexes.push_back(k_outer_offset + k_inner_offset + k_loop);
                            auto activation_addr = activation_tensor->get_addr(activation_indexes);
                            // xxx is it ok to validate with garbage_addr?
                            if (activation_addr != GARBAGE_ADDR) {
                                // save maximum tile_m, tile_k value
                                tile_m = m_loop + 1;
                                tile_k = k_loop + 1;
                                // activation_addrs.push_back(AddressConfig::switch_co_ch(act_addr));
                                // act_addr += 2;  // precision
                                activation_addrs.push_back(activation_addr);
                                // spdlog::info("index in range. target tensor dimension: {}, "
                                //              "access index {}",
                                //              activation_tensor->get_dims(), activation_indexes);
                            } else {
                                // spdlog::info("index out of range. target tensor dimension: {}, "
                                //              "access index {}",
                                //              activation_tensor->get_dims(), activation_indexes);
                            }
                        }
                    }
                    if (activation_addrs.size() == 0) {
                        spdlog::info(
                            "zero load for activation m: {} {} / k: "
                            "{} {} / activation tensor dim: {}",
                            m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset,
                            activation_tensor->get_dims());
                        assert(0);
                    } else {
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_activation_offset,
                            .size = (uint32_t)activation_addrs.size() * _config.precision,
                            .src_addrs = std::move(activation_addrs),
                            .operand_id = _INPUT_OPERAND});
                    }
                }
                // -- weight --
                if (m_inner_offset == 0) {
                    tile_n = 0;
                    // During the m_inner tile iterations (to prevent duplication),
                    // add the MOVIN instruction only in the first inner loop.
                    std::vector<addr_type> weight_addrs;
                    for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                            std::vector<uint32_t> weight_indexes(batch_index);
                            weight_indexes.push_back(k_outer_offset + k_inner_offset + k_loop);
                            weight_indexes.push_back(n_outer_offset + n_inner_offset + n_loop);
                            auto weight_addr = weight_tensor->get_addr(weight_indexes);
                            if (weight_addr != GARBAGE_ADDR) {
                                tile_n = n_loop + 1;
                                weight_addrs.push_back(weight_addr);
                                // spdlog::info(
                                //     "index in range. target tensor dimension: {}, "
                                //     "access index {}",
                                //     weight_tensor->get_dims(), weight_indexes);
                            } else {
                                // spdlog::info(
                                //     "index out of range. target tensor dimension: {}, "
                                //     "access index {}",
                                //     weight_tensor->get_dims(), weight_indexes);
                            }
                        }
                    }
                    if (weight_addrs.size() == 0) {
                        spdlog::info(
                            "operation name : {} / "
                            "zero load for weight k: {} {} / n: {} {} "
                            "/ weight tensor dim: {} / is transposed: {}",
                            get_name(), k_outer_offset, k_inner_offset, n_outer_offset,
                            n_inner_offset, weight_tensor->get_dims(),
                            weight_tensor->_is_transposed);
                        spdlog::info("inner loop {}, outer loop {}, act_size {}, wgt_size {}",
                                     _inner_loop, _outer_loop, activation_tensor->get_dims(),
                                     weight_tensor->get_dims());
                        assert(0);
                    } else {
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_weight_offset,
                            .size = (uint32_t)weight_addrs.size() * _config.precision,
                            .src_addrs = std::move(weight_addrs),
                            .operand_id = _INPUT_OPERAND + 1,
                        });
                    }
                }
                // spdlog::info("{} {} {}", activation_tensor->get_dims(),
                // weight_tensor->get_dims(),
                //              output_tensor->get_dims());
                // std::cout << "tile " << m_inner_offset << " " << k_inner_offset << " "
                //           << n_inner_offset << std::endl;
                // std::cout << "tile " << tile_m << " " << tile_n << " " << tile_k << std::endl;
                // -- compute --
                // in case of 1st L1 tile execution, execute GEMM_PRELOAD instruction
                tile.instructions.push_back(Instruction{
                    .opcode = (m_inner_offset == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM),
                    .dest_addr = sram_accumulation_offset,
                    // xxx : fixed to systolic array size 8
                    .size = loop_size / 8,
                    // what does src_addrs do in computation instructions?
                    // read Core::can_issue_compute.
                    // checks if it's loaded to sram.
                    .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_weight_offset},

                    .tile_m = tile_m,
                    .tile_k = tile_k,
                    .tile_n = tile_n,
                });
                // -- store --
                // when iterating inner_loop k times,
                // store L1 tile to output
                if (should_store && (k_inner_offset + loop_size >= k_inner)) {
                    std::vector<addr_type> output_addrs;
                    for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                        for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                            std::vector<uint32_t> output_indexes(batch_index);
                            output_indexes.push_back(m_outer_offset + m_inner_offset + m_loop);
                            output_indexes.push_back(n_outer_offset + n_inner_offset + n_loop);
                            auto output_addr = output_tensor->get_addr(output_indexes);
                            if (output_addr != GARBAGE_ADDR) {
                                output_addrs.push_back(output_addr);
                            }
                        }
                    }
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::MOVOUT,
                        .dest_addr = sram_accumulation_offset,
                        .size = (uint32_t)output_addrs.size() * _config.precision,
                        .src_addrs = std::move(output_addrs),
                        .operand_id = _OUTPUT_OPERAND,
                    });
                }
            }
        }
    }

    if (_is_transposed) {
        activation_tensor->unset_transposed();
        weight_tensor->unset_transposed();
    }

    // spdlog::info("{} instructions generated from tile {}",
    // t.instructions.size(), t.optype); spdlog::info("outer loop {}, inner loop
    // {}", _outer_loop, _inner_loop);

    return tile;
}

// Initialize _inner_loop, _outer_loop
// _inner_loop represents the L2 tile size for each axis M, N, K.
// _outer_loop indicates how many L2 tiles the matmul is divided into across each axis M, N, K.
void MatMul::calculate_loops() {
    std::vector<uint32_t> input0_dims(_inputs[0]->get_dims());
    std::vector<uint32_t> input1_dims(_inputs[1]->get_dims());

    // m,k @ k,n
    _inner_loop.resize(3);  // M, K, N

    // _inner_loop[0]: [b, l, E] -> l. [b, h, l, d_k] -> l.
    _inner_loop[0] = input0_dims[input0_dims.size() - 2];
    _inner_loop[1] = input0_dims.back();
    _inner_loop[2] = input1_dims.back();

    _outer_loop.assign(3, 1);

    // todo: future work, consider broadcasting.
    // currently, it just assumes that the feature size of the smaller
    // dimensions are included to larger dimensions.

    // larger_dims: [768, 2304] => _prod_batches: 1
    // larger_dims: [1, 12, 64, 15] => _prod_batches: 12
    // Calculate the number of iterations for matmul by multiplying all dimensions except the last
    // two.
    _prod_batches = 1;
    auto larger_dims = input0_dims.size() > input1_dims.size() ? input0_dims : input1_dims;
    for (uint32_t i = 0; i + 2 < larger_dims.size(); i++) {
        _prod_batches *= larger_dims[i];
    }

    while (sram_size_needed() > _config.spad_size KB / 2)  // double buffer
    {
        // max_element return iterator
        // divide max_element dimension to 1/2,
        // increment outer_loop to 1
        auto max_el = max_element(_inner_loop.begin(), _inner_loop.end());
        _outer_loop[max_el - _inner_loop.begin()] *= 2;
        *max_el = ((*max_el) & 1) + ((*max_el) >> 1);  // ceil(*max_el / 2)
    }

    if (_is_transposed) {
        std::reverse(_inner_loop.begin(), _inner_loop.end());
        std::reverse(_outer_loop.begin(), _outer_loop.end());
    }
    spdlog::info("MatMul inner loop: {}, outer loop: {}", _inner_loop, _outer_loop);
    // todo: if _inner_loop cannot fill the sram, extra batching is needed for
    // more utilization

    // spdlog::info("sram utilization of tile {}: {}", get_name(),
    //              (float)sram_size_needed() / (float)_config.spad_size);
}

// bias is loaded to the accumulation space
uint32_t MatMul::sram_size_needed() {
    // If performing the inner loop [130, 130, 130] on a 128x128 SA,
    // align to [256, 256, 256] by adding [128 - 2, 128 - 2, 128 - 2] to each dimension,
    // and load into SRAM.

    auto n = _inner_loop[0];
    if (n % _config.core_width != 0) {
        n += _config.core_width - n % _config.core_width;
    }
    auto k = _inner_loop[1];
    if (k % _config.core_width != 0) {
        k += _config.core_width - k % _config.core_width;
    }
    auto m = _inner_loop[2];
    if (m % _config.core_width != 0) {
        m += _config.core_width - m % _config.core_width;
    }

    return (n * k + k * m + m * n) * _config.precision;
}