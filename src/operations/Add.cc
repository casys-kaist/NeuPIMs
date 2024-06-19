#include "Add.h"

Add::Add(std::string name) : Operation(name) { _inputs.resize(2); }

std::vector<Ptr<BTensor>> Add::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _outputs.resize(1);

    assert(inputs.size() == 2);
    _inputs.assign(inputs.begin(), inputs.end());

    // See more at https:numpy.org/doc/stable/user/basics.broadcasting.html
    // currently assert two input dimensions are the same.
    _input_dim = _inputs[0]->get_dims();

    spdlog::info("_input_dim:{}, _inputs[1]->get_dims():{}", _input_dim, _inputs[1]->get_dims());
    assert(_inputs[1]->get_dims() == _input_dim);

    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", _input_dim, NPUTensorBufType::ACT, false);

    calculate_loops();
    initialize_tiles();

    return _outputs;
}

void Add::initialize_tiles() {
    for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
        _tiles.push_back(initialize_instructions(N));
    }
}

// add : load two inputs
//  input   : 2*C
//  add     : C

// -- memory map
//  ---- sram base
//      ---- input(1)
//           0                          N, 2, _weight_dim
//      ---- add(2)
//           N*2*prod(_weight_dim)        N, _weight_dim
//      ---- addtree(2,6), divide n(3,7), sqrt(8)
//           (2+N)*prod(_weight_dim)    N
//      ---- multiplication(5)
//           (2+N)*prod(_weight_dim)    N, _weight_dim
Tile Add::initialize_instructions(uint32_t N) {
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

    auto activation_tensor0 = std::static_pointer_cast<NPUTensor>(_inputs[0]);
    auto activation_tensor1 = std::static_pointer_cast<NPUTensor>(_inputs[1]);
    auto output_tensor = std::static_pointer_cast<NPUTensor>(_outputs[0]);

    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
        addr_type sram_activation0_offset =
            sram_activation_base + 2 * n_inner_offset * weight_count * _config.precision;
        addr_type sram_activation1_offset =
            sram_activation_base + (2 * n_inner_offset + 1) * weight_count * _config.precision;
        addr_type sram_accumulation_offset =
            sram_accumulation_base + n_inner_offset * weight_count * _config.precision;
        // addr_type sram_activation_tmp_offset =
        //     sram_activation_tmp_base + n_inner_offset * weight_count * _config.precision;

        // -- activation --
        auto activation_addrs = activation_tensor0->get_row_addrs(n_outer_offset + n_inner_offset);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_activation0_offset,
            .size = (uint32_t)activation_addrs.size() * _config.precision,
            .src_addrs = std::move(activation_addrs),
            .operand_id = _INPUT_OPERAND,
        });

        activation_addrs = activation_tensor1->get_row_addrs(n_outer_offset + n_inner_offset);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_activation1_offset,
            .size = (uint32_t)activation_addrs.size() * _config.precision,
            .src_addrs = std::move(activation_addrs),
            .operand_id = _INPUT_OPERAND,
        });

        // -- compute --
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::ADD,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)activation_addrs.size(),
            .src_addrs = std::vector<addr_type>{sram_activation0_offset, sram_activation1_offset},
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

void Add::calculate_loops() {
    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);

    _prod_batches = 1;
    for (size_t i = 0; i + 1 < _input_dim.size(); i++) {
        _prod_batches *= _input_dim[i];
    }
    _inner_loop[0] = _prod_batches;

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
}

uint32_t Add::sram_size_needed() {
    auto n = _inner_loop[0];
    auto k = _input_dim.back();
    if (k % _config.core_width != 0) {
        k += _config.core_width - k % _config.core_width;
    }

    return 3 * n * k * _config.precision;
}
