#include "PIMGEMVAdd.h"

PIMGEMVAdd::PIMGEMVAdd(std::string name) : Operation(name) {}

std::vector<Ptr<BTensor>> PIMGEMVAdd::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _inputs = inputs;

    _batch_size = inputs.size() / 2;
    uint32_t i = 0;

    for (auto tensor : inputs) {
        if (i < _batch_size) {
            _logits.push_back(std::static_pointer_cast<NPUTensor>(tensor));
        } else {
            _vs.push_back(std::static_pointer_cast<PIMTensor>(tensor));
        }
        i++;
    }

    _outputs.resize(_batch_size);

    _nh = _vs[0]->get_dims()[0];
    _dk = _vs[0]->get_dims()[2];

    // assert(inputs.size() == 2);
    for (int i = 0; i < _batch_size; ++i) {
        auto L = _logits[i];  // [h, 1, seq_len]
        auto V = _vs[i];      // [h, seq_len, dk]

        spdlog::info("L: {}, V: {}", L->get_dims(), V->get_dims());
        // seq_len of L == seq_len of V
        assert(L->get_dims()[2] == V->get_dims()[1]);
        std::vector<uint32_t> gemv_output_dim{_nh, 1, _dk};

        _outputs[i] = std::make_shared<NPUTensor>(_name + "_output", gemv_output_dim,
                                                  NPUTensorBufType::ACT, false);
    }

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    spdlog::info("output dim (batch size): {}", _batch_size);

    return _outputs;
}

void PIMGEMVAdd::initialize_tiles() { _tiles.push_back(initialize_instructions()); }

Tile PIMGEMVAdd::initialize_instructions() {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };

    addr_type sram_act_base = SPAD_BASE;
    addr_type sram_acc_base = ACCUM_SPAD_BASE;
    addr_type sram_addr = sram_act_base;
    addr_type sram_acc_addr = sram_acc_base;
    for (int i = 0; i < _batch_size; ++i) {
        auto logit = _logits[i];
        auto value = _vs[i];

        uint32_t seq_len = value->get_dims()[1];
        uint32_t ch = value->get_channel();
        uint32_t chunks = ceil((double)seq_len / _page_size);
        // spdlog::info("seq_len: {}", seq_len);

        for (int hi = 0; hi < _nh; hi++) {
            std::map<uint32_t, std::vector<addr_type>> sram_readres_addrs;
            for (int ci = 0; ci < chunks; ci++) {
                uint64_t logit_row = 0;  // FIXME: decode row index from dram address
                uint64_t p_header_addr =
                    AddressConfig::encode_pim_header(ch, logit_row, true, 0, 0);
                //  P_HEADER (for_gwrite=true)
                // tile.instructions.push_back(Instruction{
                //     .opcode = Opcode::PIM_HEADER,
                //     .dest_addr = sram_addr,
                //     .size = 0,
                //     .src_addrs = std::vector<addr_type>{p_header_addr},
                //     .operand_id = _INPUT_OPERAND,
                // });
                // GWRITE (channel, bank, row)
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_GWRITE,
                    .dest_addr = sram_addr,
                    .size = 0,
                    .src_addrs = std::vector<addr_type>{p_header_addr},  // FIXME: gwrite addr
                    .operand_id = _INPUT_OPERAND,
                });

                uint32_t num_comps =
                    (ci == chunks - 1 && (seq_len % _page_size) > 0)
                        ? ceil((double)(seq_len % _page_size) / _datas_per_comp_cmd)
                        : _page_size / _datas_per_comp_cmd;
                uint32_t decoded_num_comps = 1 << LogBase2(num_comps);

                // spdlog::info("num_comps: {}, decoded_num_comps: {}", num_comps,
                // decoded_num_comps);
                if (num_comps > decoded_num_comps) {
                    decoded_num_comps *= 2;
                }
                assert(num_comps <= decoded_num_comps);
                assert(num_comps > 0);

                for (int ti = 0; ti < _tiles_per_chunk; ti++) {
                    int num_comps_debug = 0;
                    int num_readres_debug = 0;

                    uint32_t DRAM_row =
                        value->_rows[ti * chunks + ci];
                    p_header_addr =
                        AddressConfig::encode_pim_header(ch, DRAM_row, false, decoded_num_comps, 1);
                    // P_HEADER (num_comps, num_readres)
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::PIM_HEADER,
                        .dest_addr = sram_addr,
                        .size = 0,
                        .src_addrs = std::vector<addr_type>{p_header_addr},
                        .operand_id = _INPUT_OPERAND,
                    });
                    std::string cmds = "P_HEADER ";

                    uint64_t dram_addr = AddressConfig::make_address(ch, 0, 0, 0, DRAM_row, 0);
                    Instruction comp_inst = Instruction{
                        .opcode = Opcode::PIM_COMP,
                        .dest_addr = sram_addr,
                        .size = 0,
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    };

                    for (int j = 0; j < num_comps; j++) {
                        // COMP * num_comps (channnel, row)
                        tile.instructions.push_back(comp_inst);
                        num_comps_debug++;
                        cmds += "COMP ";
                    }
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::PIM_READRES,
                        .dest_addr = sram_addr,
                        .size = _banks_per_channel * _config.precision,
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    });
                    cmds += "READRES ";
                    num_readres_debug++;
                    if (sram_readres_addrs.find(ti) == sram_readres_addrs.end())  // not exists
                        sram_readres_addrs[ti] = std::vector<addr_type>{sram_addr};
                    else
                        sram_readres_addrs[ti].push_back(sram_addr);

                    sram_addr += _banks_per_channel * _config.precision;

                    // check
                    // if (ch == 0) {
                    //     int num_readres = 1;
                    //     std::string red = "\033[1;31m";
                    //     std::string reset = "\033[0m";
                    //     std::string color = red;
                    //     if (num_comps_debug == num_comps) color = reset;

                    //     spdlog::info("{}num_comps: {}, actual:{},  {}", color, num_comps,
                    //                  num_comps_debug, reset);
                    //     if (num_readres_debug == num_readres)
                    //         color = reset;
                    //     else
                    //         color = red;
                    //     spdlog::info("{}num_readres: {}, actual:{},  {}", color, num_readres,
                    //                  num_readres_debug, reset);
                    //     spdlog::info("{}cmds: {}{}", red, cmds, reset);
                    // }
                }
            }
            if (chunks > 1) {
                for (int ti = 0; ti < _tiles_per_chunk; ++ti) {
                    assert(sram_readres_addrs[ti].size() == chunks);
                    uint32_t column_height =
                        _tiles_per_chunk * _banks_per_channel * _config.precision;
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::ADD,
                        .dest_addr = sram_acc_base,
                        .size = column_height,
                        .src_addrs = sram_readres_addrs[ti],
                    });
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::MOVOUT,
                        .dest_addr = sram_acc_base,
                        .size = column_height,
                        .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                                         ->_inners[hi]
                                         ->get_all_addrs(),  // TODO:
                        .operand_id = _OUTPUT_OPERAND,
                    });
                    sram_acc_base += column_height;
                }
            }
        }
    }
    spdlog::info("tile size: {}", tile.instructions.size());
    return tile;
}

void PIMGEMVAdd::calculate_loops() {
    assert(sram_size_needed() < 3000 KB);

    // todo: set from config
    uint32_t E = _config.model_n_embd;

    // memory spec
    _page_size =
        _config.dram_page_size / _config.precision;  // # of parameter in DRAM row
    _banks_per_channel = _config.dram_banks_per_ch;

    _tiles_per_chunk = ceil((double)_dk / _banks_per_channel);
    _datas_per_comp_cmd = _config.pim_comp_coverage;
}

uint32_t PIMGEMVAdd::sram_size_needed() {
    /// space for gemvadd activation = dk * batch_size?
    uint32_t need_size = _batch_size * _config.model_n_head * _dk * _config.precision;

    return need_size;
}