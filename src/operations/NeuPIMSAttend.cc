#include "NeuPIMSAttend.h"

NeuPIMSAttend::NeuPIMSAttend(std::string name) : Operation(name) {}

std::vector<Ptr<BTensor>> NeuPIMSAttend::get_outputs(std::vector<Ptr<BTensor>> inputs) {
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
        auto L = _logits[i];  // [h, l, seq_len] // l must be seq_len or 1
        auto V = _vs[i];      // [h, seq_len, dk]

        // spdlog::info("(NeuPIMSAttend) L: {}, V: {}", L->get_dims(), V->get_dims());
        // seq_len of L == seq_len of V
        assert(L->get_dims()[2] == V->get_dims()[1]);
        // nh of L == nh of V
        assert(L->get_dims()[0] == V->get_dims()[0]);

        uint32_t l = L->get_dims()[1];
        std::vector<uint32_t> attend_output_dim{_nh, l, _dk};

        _outputs[i] = std::make_shared<NPUTensor>(_name + "_output", attend_output_dim,
                                                  NPUTensorBufType::ACT, false);
    }

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    spdlog::info("output dim (batch size): {}", _batch_size);

    return _outputs;
}

void NeuPIMSAttend::initialize_tiles() {
    int num_npu_tiles = _req_idxs.size();
    int prev_idx = 0;
    for (int i = 0; i < num_npu_tiles; i++) {
        int req_idx = _req_idxs[i];
        if (i == num_npu_tiles - 1) assert(req_idx == _batch_size - 1);

        _tiles.push_back(initialize_instructions(prev_idx, req_idx));
        prev_idx = req_idx;
    }
}

Tile NeuPIMSAttend::initialize_instructions(int start, int end) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };

    for (int i = start; i < end + 1; ++i) {
        auto logit = _logits[i];
        auto value = _vs[i];

        uint32_t seq_len = value->get_dims()[1];
        uint32_t ch = value->get_channel();
        uint32_t chunks = ceil((double)seq_len / _page_size);
        // spdlog::info("seq_len: {}", seq_len);

        if (logit->get_dims()[1] != 1) {
            // spdlog::info("logit dim:{}", logit->get_dims());
            // spdlog::info("value dim:{}", value->get_dims());
            assert(logit->get_dims()[1] == seq_len);

            for (int h_idx = 0; h_idx < _nh; h_idx++) {
                std::vector<addr_type> dram_logit_addrs;
                std::vector<addr_type> dram_value_addrs;

                for (int dk_idx = 0; dk_idx < _dk; dk_idx++) {
                    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
                        dram_value_addrs.push_back(
                            value->get_addr(std::vector<uint32_t>{h_idx, seq_idx, dk_idx}));

                        for (int sseq_idx = 0; sseq_idx < seq_len; sseq_idx++) {
                            dram_logit_addrs.push_back(logit->get_addr({h_idx, seq_idx, sseq_idx}));
                        }
                    }
                }
                auto sram_l_entry = allocate_sram_addr(seq_len * seq_len, false);
                auto sram_v_entry = allocate_sram_addr(seq_len * _dk, false);
                auto sram_a_entry = allocate_sram_addr(seq_len * _dk, true);
                // -- load --
                // MOVIN logit, value
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_l_entry.first,
                    .size = sram_l_entry.second,
                    .src_addrs = dram_logit_addrs,
                    .operand_id = _INPUT_OPERAND  // logit
                });
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_v_entry.first,
                    .size = sram_v_entry.second,
                    .src_addrs = dram_value_addrs,
                    .operand_id = _INPUT_OPERAND  // logit
                });

                // -- compute --
                // GEMM (l*v -> a)
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::GEMM,
                    .dest_addr = sram_a_entry.first,
                    .size = sram_a_entry.second,
                    .src_addrs = std::vector<addr_type>{sram_l_entry.first, sram_v_entry.first},

                    .tile_m = _dk,
                    .tile_k = seq_len,
                    .tile_n = seq_len,
                });

                // MOVOUT
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVOUT,
                    .dest_addr = sram_a_entry.first,
                    .size = sram_a_entry.second,
                    .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                                     ->_inners[h_idx]
                                     ->get_all_addrs(),
                    .operand_id = _OUTPUT_OPERAND,
                });
            }

            continue;
        }

        for (int hi = 0; hi < _nh; hi++) {
            std::map<uint32_t, std::vector<addr_type>> sram_readres_addrs;
            for (int ci = 0; ci < chunks; ci++) {
                uint64_t logit_row = 0;  // FIXME: decode row index from dram address
                uint64_t p_header_addr =
                    AddressConfig::encode_pim_header(ch, logit_row, true, 0, 0);

                addr_type sram_addr_gw = allocate_sram_addr(0, false).first;

                // GWRITE (channel, bank, row)
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_GWRITE,
                    .dest_addr = sram_addr_gw,
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
                    auto sram_entry = allocate_sram_addr(_banks_per_channel, false);
                    addr_type sram_addr = sram_entry.first;

                    uint32_t DRAM_row = value->_rows[ti * chunks + ci];
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

                    uint64_t dram_addr =
                        AddressConfig::encode_pim_comps_readres(ch, DRAM_row, num_comps, true);

                    if (_config.dram_type == DramType::NEWTON) {
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
                            cmds += "COMP ";
                        }
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::PIM_READRES,
                            .dest_addr = sram_addr,
                            .size = sram_entry.second,
                            .src_addrs = std::vector<addr_type>{dram_addr},
                            .operand_id = _INPUT_OPERAND,
                        });
                        cmds += "READRES ";
                    } else {
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::PIM_COMPS_READRES,
                            .dest_addr = sram_addr,
                            .size = sram_entry.second,
                            .src_addrs = std::vector<addr_type>{dram_addr},
                            .operand_id = _INPUT_OPERAND,
                        });
                    }

                    if (sram_readres_addrs.find(ti) == sram_readres_addrs.end())  // not exists
                        sram_readres_addrs[ti] = std::vector<addr_type>{sram_addr};
                    else
                        sram_readres_addrs[ti].push_back(sram_addr);
                }
            }
            if (chunks > 1) {
                for (int ti = 0; ti < _tiles_per_chunk; ++ti) {
                    assert(sram_readres_addrs[ti].size() == chunks);

                    uint32_t column_height = _tiles_per_chunk * _banks_per_channel;
                    auto sram_acc_entry = allocate_sram_addr(column_height, true);

                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::ADD,
                        .dest_addr = sram_acc_entry.first,
                        .size = sram_acc_entry.second,
                        .src_addrs = sram_readres_addrs[ti],
                    });
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::MOVOUT,
                        .dest_addr = sram_acc_entry.first,
                        .size = sram_acc_entry.second,
                        .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                                         ->_inners[hi]
                                         ->get_all_addrs(),
                        .operand_id = _OUTPUT_OPERAND,
                    });
                }
            }
        }
    }
    // spdlog::info("tile size: {}", tile.instructions.size());
    return tile;
}

void NeuPIMSAttend::calculate_loops() {
    // assert(sram_size_needed() < _config.spad_size KB / 2);

    uint32_t E = _config.model_n_embd / _config.n_tp;

    // memory spec
    _page_size = _config.dram_page_size / _config.precision;
    _banks_per_channel = _config.dram_banks_per_ch;

    _tiles_per_chunk = ceil((double)_dk / _banks_per_channel);
    _datas_per_comp_cmd = _config.pim_comp_coverage;

    // npu tiling
    int heads_per_dram_page = floor((double)_page_size / _dk);
    int heads_space_in_page = heads_per_dram_page * _dk;
    int chunks = ceil((double)E / heads_space_in_page);

    int sram_needs = 0;
    for (int i = 0; i < _batch_size; ++i) {
        auto L = _logits[i];  // [h, l, seq_len] // l must be seq_len or 1
        auto V = _vs[i];      // [h, seq_len, dk]

        uint32_t q_len = L->get_dims()[1];
        uint32_t seq_len = V->get_dims()[2];

        int need_sram_for_req = 0;

        if (q_len == 1) {
            // incremental phase
            need_sram_for_req = (seq_len + chunks * _dk) * _nh * _config.precision;
            sram_needs += need_sram_for_req;
        } else {
            // initiation phase
            assert(false);
            // now support only incremental phases
        }

        if (sram_needs > _config.spad_size KB / _config.precision) {
            assert(i > 0);
            _req_idxs.push_back(i - 1);
            sram_needs = need_sram_for_req;
        }
    }
    _req_idxs.push_back(_batch_size - 1);
}

uint32_t NeuPIMSAttend::sram_size_needed() {
    /// space for gemvadd activation = dk * batch_size?
    uint32_t need_size = _batch_size * _config.model_n_head * _dk * _config.precision;

    return 0;  // need_size;
}