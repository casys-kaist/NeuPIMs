#include "LayerNorm.h"

#include "../tensor/NPUTensor.h"

LayerNorm::LayerNorm(std::string name, std::vector<Ptr<NPUTensor>> weights) : Operation(name) {
    assert(weights.size() == 2);
    _inputs.resize(3);

    assert(weights[0]->get_dims() == weights[1]->get_dims());
    _weight_dim = weights[0]->get_dims();
    _prod_weight_dim = 1;
    for (auto weight : _weight_dim) {
        _prod_weight_dim *= weight;
    }

    _inputs[1] = weights[0];
    _inputs[2] = weights[1];
}

// LayerNorm does not change shapes.
std::vector<Ptr<BTensor>> LayerNorm::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _outputs.resize(1);

    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];

    auto input_dims = _inputs[0]->get_dims();
    auto input_dim_riter = input_dims.rbegin();
    // last dimensions of weight and input matches
    for (auto weight_dim_riter = _weight_dim.rbegin(); weight_dim_riter != _weight_dim.rend();
         ++weight_dim_riter) {
        assert((*weight_dim_riter) == (*input_dim_riter));
        assert(input_dim_riter != input_dims.rend());
        input_dim_riter++;
    }
    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", input_dims, NPUTensorBufType::ACT, false);

    calculate_loops();
    initialize_tiles();

    spdlog::info("input dims : {} {} {}", _inputs[0]->get_dims(), _inputs[1]->get_dims(),
                 _inputs[2]->get_dims());
    spdlog::info("output dim : {}", input_dims);

    return _outputs;
}

// a, b : 2C
// in   : N * C
// else : N * C (multiplication^2)
// out  : N * C
// sum  : C * (2 + 3 * N)
void LayerNorm::initialize_tiles() {
    for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
        _tiles.push_back(initialize_instructions(N));
    }
}

// layernorm:
//  load affine gamma, beta, input

// -- instructions
//  input           : C
//      addtree             : 1 (C+1)
//      divide n            : 1 (C+1)
//  scalar sub      : C
//      multiplication ^2   : C (2*C)
//      addtree             : 1 (C+1)
//      divide n            : 1 (C+1)
//      sqrt                : 1 (C+1)
//  scalar div      : C
//  scalar mul * a  : C
//  scalar add + b  : C
// -- memory map
//  ---- sram base
//      ---- _affine_gamma, _affine_beta
//           0                          2, _weight_dim
//      ---- inputs(1), scalar sub(4), scalar div(9), scalar mul(10), scalar
//      add(11)
//           2*prod(_weight_dim)        N, _weight_dim
//      ---- addtree(2,6), divide n(3,7), sqrt(8)
//           (2+N)*prod(_weight_dim)    N
//      ---- multiplication(5)
//           (2+N)*prod(_weight_dim)    N, _weight_dim
Tile LayerNorm::initialize_instructions(uint32_t N) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    uint32_t weight_count = 1;
    for (auto weight : _weight_dim) {
        weight_count *= weight;
    }

    // base on the inner loop, initialize instructions
    auto n_inner = _inner_loop[0];
    auto n_outer_offset = n_inner * N;

    addr_type sram_gamma_base = SPAD_BASE;
    addr_type sram_beta_base = sram_gamma_base + weight_count * _config.precision;
    addr_type sram_activation_base = sram_beta_base + weight_count * _config.precision;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    // xxx not used
    const uint32_t loop_size = _config.vector_core_width;

    auto activation_tensor = std::static_pointer_cast<NPUTensor>(_inputs[0]);
    auto gamma_tensor = std::static_pointer_cast<NPUTensor>(_inputs[1]);
    auto output_tensor = std::static_pointer_cast<NPUTensor>(_outputs[0]);

    // -- bias --
    // if      input size is 2, no need for bias initialization, but should
    // create an sram for activation region else,   create activation region
    // using bias load
    if (_inputs.size() == 3) {
        auto beta_tensor = std::static_pointer_cast<NPUTensor>(_inputs[2]);
        // std::set<addr_type> beta_addrs =
        // beta_tensor->calculate_dram_addresses({});
        std::vector<addr_type> beta_addrs = beta_tensor->get_all_addrs();
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_beta_base,
            // assume broadcasting bias is available inside the npu
            .size = (uint32_t)beta_addrs.size() * _config.precision,
            .src_addrs = std::move(beta_addrs),
            .operand_id = _INPUT_OPERAND + 2,
        });
    }

    // std::set<addr_type> gamma_addrs =
    // gamma_tensor->calculate_dram_addresses({});
    std::vector<addr_type> gamma_addrs = gamma_tensor->get_all_addrs();
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_gamma_base,
        // assume broadcasting bias is available inside the npu
        .size = (uint32_t)gamma_addrs.size() * _config.precision,
        .src_addrs = std::move(gamma_addrs),
        .operand_id = _INPUT_OPERAND + 1,
    });

    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
        addr_type sram_activation_offset =
            sram_activation_base + n_inner_offset * weight_count * _config.precision;
        addr_type sram_accumulation_offset =
            sram_accumulation_base + n_inner_offset * weight_count * _config.precision;
        // addr_type sram_activation_tmp_offset = sram_activation_tmp_base +
        // n_inner_offset * weight_count * _config.precision;

        // -- activation --
        uint32_t row_idx = n_outer_offset + n_inner_offset;
        std::vector<addr_type> activation_addrs = activation_tensor->get_row_addrs(row_idx);

        if (activation_addrs.size() == 0)
            spdlog::info(
                "zero load for activation m: {} {} / k: {} {} / "
                "activation tensor dim: {}",
                activation_tensor->get_dims());
        else
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_activation_offset,
                .size = (uint32_t)activation_addrs.size() * _config.precision,
                .src_addrs = std::move(activation_addrs),
                .operand_id = _INPUT_OPERAND,
            });

        // -- compute --
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::LAYERNORM,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)activation_addrs.size(),
            .src_addrs =
                std::vector<addr_type>{sram_activation_offset, sram_gamma_base, sram_beta_base},
        });
        // -- save outputs --
        std::vector<addr_type> output_addrs =
            output_tensor->get_row_addrs(n_outer_offset + n_inner_offset);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)output_addrs.size() * _config.precision,
            .src_addrs = std::move(output_addrs),
            .operand_id = _OUTPUT_OPERAND,
        });
    }

    // spdlog::info("{} instructions generated from tile {}", tile.instructions.size(),
    // tile.optype); spdlog::info("outer loop {}, inner loop {}", _outer_loop, _inner_loop);

    return tile;
}

void LayerNorm::calculate_loops() {
    std::vector<uint32_t> input_dim(_inputs[0]->get_dims());

    // vector processing unit needs only batching
    _inner_loop.resize(1);

    _outer_loop.assign(1, 1);

    _prod_batches = 1;
    for (size_t i = 0; i + _weight_dim.size() < input_dim.size(); i++) {
        _prod_batches *= input_dim[i];
    }
    _inner_loop[0] = _prod_batches;

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
    // spdlog::info("sram utilization of tile {}: {}", get_name(),
    //              (float)sram_size_needed() / (float)(_config.spad_size KB / 2));
}

uint32_t LayerNorm::sram_size_needed() {
    auto n = _inner_loop[0];
    auto k = _prod_weight_dim;
    if (k % _config.core_width != 0) {
        k += _config.core_width - k % _config.core_width;
    }

    return k * (n * 3 + 2) * _config.precision;
}