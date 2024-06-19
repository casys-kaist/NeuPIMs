#include "Microbench.h"

Microbench::Microbench(std::string name) : Operation(name) { _inputs.resize(1); }

// Microbench does not change shapes.
std::vector<Ptr<BTensor>> Microbench::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _outputs.resize(1);

    _inputs[0] = inputs[0];

    _input_dim = inputs[0]->get_dims();
    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", _input_dim, NPUTensorBufType::ACT, false);

    calculate_loops();
    initialize_tiles();

    return _outputs;
}

void Microbench::initialize_tiles() { _tiles.push_back(initialize_instructions()); }

// table lookup
Tile Microbench::initialize_instructions() {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .K = 0,
        .accum = false,
    };

    uint32_t N = 10;
    int num_gemvs = 1;
    int load_iteration = 50;

    int ch = 0;

    int num_comps = 32;
    int num_readres = 8;
    int num_comps_per_readres = num_comps / num_readres;
    int banks_per_channel = 32;

    for (uint32_t i = 0; i < N; ++i) {
        std::vector<addr_type> sram_load_addrs;
        std::vector<addr_type> sram_gemv_addrs;
        // PIM GEMV
        for (int gemv_idx = 0; gemv_idx < num_gemvs; gemv_idx++) {
            // if (i == 0) break;
            int pim_row = 100 + i * num_gemvs + gemv_idx;
            uint32_t p_header_addr =
                AddressConfig::encode_pim_header(ch, pim_row, false, num_comps, num_readres);
            auto sram_header_entry = allocate_sram_addr(0, false);

            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_HEADER,
                .dest_addr = sram_header_entry.first,
                .size = 0,
                .src_addrs = std::vector<addr_type>{p_header_addr},
                .operand_id = _INPUT_OPERAND,
            });

            for (int r_id = 0; r_id < num_readres; r_id++) {
                uint64_t dram_addr = AddressConfig::encode_pim_comps_readres(
                    ch, pim_row, num_comps_per_readres, false);
                auto sram_gemv_entry = allocate_sram_addr(banks_per_channel, false);
                sram_gemv_addrs.push_back(sram_gemv_entry.first);
                if (_config.dram_type == DramType::NEWTON) {
                    for (int c_id = 0; c_id < num_comps_per_readres; c_id++) {
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::PIM_COMP,
                            .dest_addr = sram_gemv_entry.first,
                            .size = 0,
                            .src_addrs = std::vector<addr_type>{dram_addr},
                            .operand_id = _INPUT_OPERAND,
                        });
                    }
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::PIM_READRES,
                        .dest_addr = sram_gemv_entry.first,
                        .size = sram_gemv_entry.second,
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    });
                } else if (_config.dram_type == DramType::NEUPIMS) {
                    if (r_id == num_readres - 1)
                        dram_addr = AddressConfig::encode_pim_comps_readres(
                            ch, pim_row, num_comps_per_readres, true);
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::PIM_COMPS_READRES,
                        .dest_addr = sram_gemv_entry.first,
                        .size = sram_gemv_entry.second,
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    });
                }
            }
        }

        // LOAD
        std::vector<uint64_t> activation_addrs;
        for (int load_idx = 0; load_idx < load_iteration; load_idx++) {
            int banks_per_channel = 16;

            for (int j = 0; j < banks_per_channel; j++) {
                int dram_row = 10 + load_idx;  // rand() % 100;
                int rank = j >> 4;

                int bankgroup = (j & 15) >> 2;
                int bank = j & 3;
                int col = j + load_idx;

                uint32_t dram_addr =
                    AddressConfig::make_address(ch, rank, bankgroup, bank, dram_row, col);
                activation_addrs.push_back(dram_addr);
            }
            auto sram_load_entry = allocate_sram_addr(activation_addrs.size(), false);
            sram_load_addrs.push_back(sram_load_entry.first);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_load_entry.first,
                .size = sram_load_entry.second,
                .src_addrs = activation_addrs,
                .operand_id = _INPUT_OPERAND,
            });
        }
        // spdlog::info("sram_load_addrs.size:{}", sram_load_addrs.size());
        // spdlog::info("sram_gemv_addrs.size:{}", sram_gemv_addrs.size());

        // COMPUTE with load data
        auto sram_accum_entry = allocate_sram_addr(8, true);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::DUMMY,
            .dest_addr = sram_accum_entry.first,
            .size = sram_accum_entry.second,
            .src_addrs = sram_load_addrs,
            .tile_m = 12,
            .tile_k = 32,
            .tile_n = 32,
        });

        uint32_t movout_addr = AddressConfig::make_address(ch, 0, 0, 0, 0, 0);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_accum_entry.first,
            .size = sram_accum_entry.second,
            .src_addrs = std::vector<addr_type>{movout_addr},
        });

        // COMPUTE with gemv data
        auto sram_gemv_accum_entry = allocate_sram_addr(8, true);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::DUMMY,
            .dest_addr = sram_gemv_accum_entry.first,
            .size = sram_gemv_accum_entry.second,
            .src_addrs = sram_gemv_addrs,
            .tile_m = 12,
            .tile_k = 32,
            .tile_n = 32,
        });

        uint32_t gemv_movout_addr = AddressConfig::make_address(ch, 1, 0, 0, 0, 0);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_gemv_accum_entry.first,
            .size = sram_gemv_accum_entry.second,
            .src_addrs = std::vector<addr_type>{gemv_movout_addr},
        });
    }

    return tile;
}

void Microbench::calculate_loops() {}

uint32_t Microbench::sram_size_needed() { return 0; }