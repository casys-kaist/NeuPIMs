#include "NeuPIMSLogitSoftmax.h"

NeuPIMSLogitSoftmax::NeuPIMSLogitSoftmax(std::string name) : Operation(name) {}

std::vector<Ptr<BTensor>> NeuPIMSLogitSoftmax::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _inputs = inputs;

    _batch_size = inputs.size() / 2;

    spdlog::info("NeuPIMSLogitSoftmax (batch size): {}", _batch_size);

    uint32_t i = 0;
    for (auto tensor : inputs) {
        if (i < _batch_size) {
            _qs.push_back(std::static_pointer_cast<NPUTensor>(tensor));
        } else {
            _ks.push_back(std::static_pointer_cast<PIMTensor>(tensor));
        }
        i++;
    }

    _outputs.resize(_batch_size);

    _nh = _qs[0]->get_dims()[0];
    _dk = _qs[0]->get_dims()[2];
    _E = _nh * _dk;
    spdlog::info("(NeuPIMSLogitSoftmax) nh:{}, dk:{}", _nh, _dk);

    // assert(inputs.size() == 2);
    for (int i = 0; i < _batch_size; ++i) {
        auto Q = _qs[i];  // [h, l, d_k]
        auto K = _ks[i];  // [h, d_k, seq_len]

        uint32_t seq_len = K->get_dims()[2];

        // d_k of Q == d_k of K^T
        // nh of Q == nh of K^T
        // spdlog::info("Q: {}, K: {}", Q->get_dims(), K->get_dims());

        assert(Q->get_dims()[0] == K->get_dims()[0]);
        assert(Q->get_dims()[2] == K->get_dims()[1]);

        uint32_t l = Q->get_dims()[1];
        assert(l == 1 || l == K->get_dims()[2]);

        std::vector<uint32_t> logit_output_dim{_nh, l, seq_len};

        _outputs[i] = std::make_shared<NPUTensor>(_name + "_output", logit_output_dim,
                                                  NPUTensorBufType::ACT, false);
    }

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    // spdlog::info("input dims : {} {}", Q->get_dims(), K->get_dims());
    spdlog::info("output dim : {}", _batch_size);

    return _outputs;
}

void NeuPIMSLogitSoftmax::initialize_tiles() {
    int num_npu_tiles = _req_idxs.size();
    int prev_idx = 0;
    for (int i = 0; i < num_npu_tiles; i++) {
        int req_idx = _req_idxs[i];
        if (i == num_npu_tiles - 1) {
            assert(req_idx == _batch_size - 1);
        }
        _tiles.push_back(initialize_instructions(prev_idx, req_idx));
        prev_idx = req_idx;
    }
}

Tile NeuPIMSLogitSoftmax::initialize_instructions(int start, int end) {
    uint32_t banks_per_channel = _config.dram_banks_per_ch;

    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };

    int counter_for_debug = 0;

    for (int i = start; i < end + 1; i++) {
        // spdlog::info("(init_instr) req_idx: {}", i);
        auto query = _qs[i];
        auto key = _ks[i];

        if (query->get_dims()[1] != 1) {  // initiation phase
            // spdlog::info("query dim: {}", query->get_dims());
            // spdlog::info("key dim: {}", key->get_dims());
            // spdlog::info("LogitSoftmax computed in NPU");
            uint32_t seq_len = query->get_dims()[1];
            assert(seq_len == key->get_dims()[2]);

            for (int h_idx = 0; h_idx < _nh; h_idx++) {
                std::vector<addr_type> dram_query_addrs;
                std::vector<addr_type> dram_key_addrs;

                for (int dk_idx = 0; dk_idx < _dk; dk_idx++) {
                    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
                        dram_query_addrs.push_back(
                            query->get_addr(std::vector<uint32_t>{h_idx, seq_idx, dk_idx}));
                        dram_key_addrs.push_back(
                            key->get_addr(std::vector<uint32_t>{h_idx, dk_idx, seq_idx}));
                    }
                }
                auto sram_q_entry = allocate_sram_addr(seq_len * _dk, false);
                auto sram_k_entry = allocate_sram_addr(seq_len * _dk, false);
                auto sram_l_entry = allocate_sram_addr(seq_len * seq_len, true);
                auto sram_ls_entry = allocate_sram_addr(seq_len * seq_len, true);

                // -- load --
                // MOVIN query, key
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_q_entry.first,
                    .size = sram_q_entry.second,
                    .src_addrs = dram_query_addrs,
                    .operand_id = _INPUT_OPERAND,  // query
                });
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_k_entry.first,
                    .size = sram_k_entry.second,
                    .src_addrs = dram_key_addrs,
                    .operand_id = _INPUT_OPERAND + 1,  // key
                });

                // -- compute --
                // GEMM (q*k -> l)
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::GEMM,
                    .dest_addr = sram_l_entry.first,
                    .size = sram_l_entry.second,
                    .src_addrs = std::vector<addr_type>{sram_q_entry.first, sram_k_entry.first},

                    .tile_m = seq_len,
                    .tile_k = _dk,
                    .tile_n = seq_len,
                });
                // Softmax (l -> l)
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::SOFTMAX,
                    .dest_addr = sram_ls_entry.first,
                    .size = sram_ls_entry.second,
                    .src_addrs = std::vector<addr_type>{sram_l_entry.first},
                    .src_from_accum = true,
                });

                // MOVOUT
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVOUT,
                    .dest_addr = sram_ls_entry.first,
                    .size = sram_ls_entry.second,
                    .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                                     ->_inners[h_idx]
                                     ->get_all_addrs(),
                    .operand_id = _OUTPUT_OPERAND,
                });
            }

            continue;
        }

        // spdlog::info("LogitSoftmax computed in PIM");
        uint32_t ch = key->get_channel();
        std::map<uint32_t, std::vector<addr_type>> sram_readres_addrs;

        uint32_t tiles_per_chunk =
            key->get_allocated_seq_len() / banks_per_channel;  // number of comp-readres kernels

        for (int chunk = 0; chunk < _chunks; chunk++) {
            // uint64_t make_address(channel, rank, bankgroup, bank, row, col);
            // uint64_t encode_pim_header(channel, row, bool for_gwrite, num_comps, num_readres);

            uint64_t query_row = 0;  // FIXME: decode row index from dram address
            std::pair<addr_type, uint32_t> sram_entry_for_gw = allocate_sram_addr(0, false);
            uint64_t gwrite_addr =
                AddressConfig::make_address(ch, 0, 0, 0, query_row, 0);  // FIXME: real gwrite addr
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_GWRITE,
                .dest_addr = sram_entry_for_gw.first,
                .size = 0,
                .src_addrs = std::vector<addr_type>{gwrite_addr},  // FIXME: gwrite addr
                .operand_id = _INPUT_OPERAND,
            });
            // GWRITE (channel, bank, row)

            for (int ti = 0; ti < tiles_per_chunk; ti++) {
                std::pair<addr_type, uint32_t> sram_entry = allocate_sram_addr(0, false);
                addr_type sram_addr_phdr = sram_entry.first;
                int num_head_in_tile =
                    (chunk == _chunks - 1) ? _heads_in_last_chunk : _heads_per_tile;

                uint32_t DRAM_row = key->_rows[ti * _chunks + chunk];
                int num_comps = _comps_per_head * num_head_in_tile;
                int num_readres = num_head_in_tile;
                if (num_head_in_tile == 0) {
                    spdlog::info("num_head_in_tile must be greater than 0!!!");
                    exit(-1);
                }
                uint32_t p_header_addr =
                    AddressConfig::encode_pim_header(ch, DRAM_row, false, num_comps, num_readres);
                // P_HEADER (num_comps = comps_per_head * num_heads, num_readres
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_HEADER,
                    .dest_addr = sram_addr_phdr,
                    .size = 0,
                    .src_addrs = std::vector<addr_type>{p_header_addr},
                    .operand_id = _INPUT_OPERAND,
                });

                std::string cmds = "P_HEADER ";

                for (int head = 0; head < num_head_in_tile; head++) {
                    int hi = _heads_per_tile * chunk + head;

                    uint64_t dram_addr = AddressConfig::encode_pim_comps_readres(
                        ch, DRAM_row, _comps_per_head, head == num_head_in_tile - 1);

                    auto sram_entry = allocate_sram_addr(banks_per_channel, false);
                    addr_type sram_addr = sram_entry.first;
                    if (_config.dram_type == DramType::NEWTON) {
                        Instruction comp_inst = Instruction{
                            .opcode = Opcode::PIM_COMP,
                            .dest_addr = sram_addr,
                            .size = 0,
                            .src_addrs = std::vector<addr_type>{dram_addr},
                            .operand_id = _INPUT_OPERAND,
                        };
                        // spdlog::info("comps:{}", _comps_per_head);
                        for (int j = 0; j < _comps_per_head; j++) {
                            // COMP * comps_per_head (channnel, row)
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
                    // spdlog::info("tile_idx:{}, head_idx: {}", ti, hi);
                    if (sram_readres_addrs.find(hi) == sram_readres_addrs.end())  // not exists
                        sram_readres_addrs[hi] = std::vector<addr_type>{sram_addr};
                    else
                        sram_readres_addrs[hi].push_back(sram_addr);
                }
            }
        }
        for (int hi = 0; hi < _nh; hi++) {
            assert(sram_readres_addrs[hi].size() == tiles_per_chunk);
            uint32_t column_height = key->_seq_len;  // tiles_per_chunk * banks_per_channel;
            std::pair<addr_type, uint32_t> sram_acc_entry = allocate_sram_addr(column_height, true);

            // spdlog::info("col height: {}, seq_len: {}", column_height, key->_seq_len);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::SOFTMAX,
                .dest_addr = sram_acc_entry.first,
                .size = sram_acc_entry.second,
                .src_addrs = sram_readres_addrs[hi],
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
            counter_for_debug++;
        }
    }

    // std::string yellow = "\033[1;33m";
    // spdlog::info("{}MOVOUT size: {}\033[0m", yellow, counter_for_debug);
    // spdlog::info("batch size: {}", _batch_size);
    return tile;
}

void NeuPIMSLogitSoftmax::calculate_loops() {
    assert(sram_size_needed() < _config.spad_size KB / 2);

    uint32_t E = _config.model_n_embd / _config.n_tp;
    // dram row capacity (unit: number of parameter)
    uint32_t page_size = _config.dram_page_size / _config.precision;
    uint32_t banks_per_channel = _config.dram_banks_per_ch;
    uint32_t datas_per_comp_cmd = _config.pim_comp_coverage;

    _chunks = ceil((double)E / page_size);            // # of gwrite
    _heads_per_tile = ceil((double)page_size / _dk);  // # of readres
    _heads_in_last_chunk =
        E % page_size == 0 ? _heads_per_tile : ceil((double)(E % page_size) / _dk);
    _comps_per_head = ceil((double)_dk / datas_per_comp_cmd);

    spdlog::info("chunks: {}", _chunks);
    spdlog::info("heads per tile: {}", _heads_per_tile);
    spdlog::info("heads in last chunk: {}", _heads_in_last_chunk);
    spdlog::info("comps per head: {}", _comps_per_head);
}

uint32_t NeuPIMSLogitSoftmax::sram_size_needed() {
    // return sram_size_needed per head
    // initiation phase: li*dk*3 + 2li^2 + li*dk
    // incremental phase: 2*li + dk
    int sram_size = _config.spad_size KB / _config.precision;

    int dram_page_size = _config.dram_page_size / _config.precision;
    int heads_per_dram_page = floor((double)dram_page_size / _dk);
    int heads_space_in_page = heads_per_dram_page * _dk;
    int chunks = ceil((double)_E / heads_space_in_page);

    spdlog::info("heads per dram page: {}", heads_per_dram_page);
    spdlog::info("_E:{}, _E/heads_per_dram_page:{}", _E, chunks);

    int sram_needs = 0;
    for (int i = 0; i < _batch_size; ++i) {
        auto Q = _qs[i];  // [h, q_len, d_k]
        auto K = _ks[i];  // [h, d_k, seq_len]

        uint32_t seq_len = K->get_dims()[2];
        uint32_t q_len = Q->get_dims()[1];
        int need_sram_for_req = 0;

        if (q_len == 1) {
            // incremental phase
            need_sram_for_req = (2 * seq_len + _dk) * _nh * _config.precision;
            sram_needs += need_sram_for_req;
        } else {
            // initiation phase
            assert(false);
            // now support only incremental phases
            need_sram_for_req = (seq_len * _dk * 3 + 2 * seq_len * seq_len + seq_len * _dk) * _nh *
                                _config.precision;
        }
        spdlog::info("(LogitSoftmax) i:{}, ch:{}", i, K->get_channel());

        if (sram_needs > sram_size) {
            spdlog::info("---");
            assert(i > 0);
            _req_idxs.push_back(i - 1);
            sram_needs = need_sram_for_req;
        }
    }
    _req_idxs.push_back(_batch_size - 1);

    return 0;  // need_size;
}