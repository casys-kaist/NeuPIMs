#include "Softmax.h"

Softmax::Softmax(std::string name) : Operation(name) {
    // assume as dim = -1
    // _inputs.resize(1);
}

// Softmax does not change shapes.
std::vector<Ptr<BTensor>> Softmax::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _batch_size = inputs.size();
    _outputs.resize(_batch_size);
    // assert(inputs.size() == 1);
    _inputs = inputs;

    for (int i = 0; i < _batch_size; i++) {
        std::vector<uint32_t> input_dim = inputs[i]->get_dims();

        _outputs[i] =
            std::make_shared<NPUTensor>(_name + "_output", input_dim, NPUTensorBufType::ACT, false);
    }
    spdlog::info("softmax batch_size: {}", _batch_size);

    initialize_tiles();

    return _outputs;
}

void Softmax::initialize_tiles() {
    for (int i = 0; i < _batch_size; i++) {
        calculate_loops(i);

        for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
            _tiles.push_back(initialize_instructions(N, i));
        }
    }
}

//  input           : C
//      reduce max  : 1 (C+1)
//  scalar sub      : C
//  exponentiation  : C
//      add tree    : 1 (C+1)
//  scalar div      : C
// -- memory map
//  ---- sram base
//      ---- inputs(1), scalar sub(3), exponentiation(4), scalar div(6)
//           0                          N, _weight_dim
//      ---- reduce max(2), add tree(5)
//           N*prod(_weight_dim)        N
Tile Softmax::initialize_instructions(uint32_t N, uint32_t req_idx) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    uint32_t weight_count = _input_dim.back();

    // base on the inner loop, initialize instructions
    auto n_inner = _inner_loop[0];
    auto n_outer_offset = n_inner * N;

    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.vector_core_width;

    auto activation_tensor = std::static_pointer_cast<NPUTensor>(_inputs[req_idx]);
    auto output_tensor = std::static_pointer_cast<NPUTensor>(_outputs[req_idx]);

    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
        addr_type sram_activation_offset =
            sram_activation_base + n_inner_offset * weight_count * _config.precision;
        addr_type sram_accumulation_offset =
            sram_accumulation_base + n_inner_offset * weight_count * _config.precision;
        // addr_type sram_activation_tmp_offset = sram_activation_tmp_base +
        // n_inner_offset * weight_count * _config.precision;

        // -- activation --
        auto activation_addrs = activation_tensor->get_row_addrs(n_outer_offset + n_inner_offset);

        spdlog::info("Softmax act_addrs.size(): {}", activation_addrs.size());

        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_activation_offset,
            .size = (uint32_t)activation_addrs.size() * _config.precision,
            .src_addrs = std::move(activation_addrs),
            .operand_id = _INPUT_OPERAND,
        });

        // -- compute --
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::SOFTMAX,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)activation_addrs.size(),
            .src_addrs = std::vector<addr_type>{sram_activation_offset},
        });
        // -- save outputs --
        auto output_addrs = output_tensor->get_row_addrs(n_outer_offset + n_inner_offset);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)output_addrs.size() * _config.precision,
            .src_addrs = std::move(output_addrs),
            .operand_id = _OUTPUT_OPERAND,
        });
    }

    // spdlog::info("{} instructions generated from tile {}",
    // tile.instructions.size(), tile.optype); spdlog::info("outer loop {},
    // inner loop {}", _outer_loop, _inner_loop);

    return tile;
}

void Softmax::calculate_loops(uint32_t req_idx) {
    std::vector<uint32_t> input_dims(_inputs[req_idx]->get_dims());

    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);
    _input_dim = _inputs[req_idx]->get_dims();

    _prod_batches = 1;
    for (size_t i = 0; i + 1 < input_dims.size(); i++) {
        _prod_batches *= input_dims[i];
    }
    _inner_loop[0] = _prod_batches;

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
        spdlog::info("SoftMax inner loop: {}, outer loop: {}", _inner_loop, _outer_loop);
    }
}

// in   : N * C
// else : N * 1 (reduce max)
// out  : N * C
// sum  : N * (2*C + 1)
uint32_t Softmax::sram_size_needed() {
    auto n = _inner_loop[0];
    auto k = _input_dim.back();
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }

    return n * (2 * k + 1) * _config.precision;
}