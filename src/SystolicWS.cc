#include "SystolicWS.h"

SystolicWS::SystolicWS(uint32_t id, SimulationConfig config) : Core(id, config) {}

void SystolicWS::cycle() {
    /* Compute unit */
    // todo: matrix_unit.cycle();
    if (!_compute_pipeline.empty() && _compute_pipeline.front().finish_cycle <= _core_cycle) {
        Instruction &inst = _compute_pipeline.front();
        if (inst.dest_addr >= ACCUM_SPAD_BASE) {
            // spdlog::info("calculation finished. instruction: {}, spad_id:{}", inst.repr(),
            //              inst.accum_spad_id);
            _acc_spad.fill(inst.dest_addr, inst.accum_spad_id);
        } else {
            assert(0);
        }
        if (auto tile = inst.parent_tile.lock()) {
            tile->remaining_accum_io--;
            tile->remaining_computes--;
        } else {
            assert(0);
        }
        _compute_pipeline.pop();
        // spdlog::info("cycle: {}, pop {}", _core_cycle, inst.repr());
    }

    /* Checking Vector compute pipeline */
    /*
        if vector pipeline has element and finishes at this cycle
            fill accum spad
            update tile parent
                remaining_accum_io --;
                remaining_computes --;
            pop target data
    */
    // vector pipeline needs iterating vectors
    // todo: vector_unit.cycle();
    for (auto &vector_pipeline : _vector_pipelines) {
        if (!vector_pipeline.empty() && vector_pipeline.front().finish_cycle <= _core_cycle) {
            Instruction &inst = vector_pipeline.front();
            if (inst.dest_addr >= ACCUM_SPAD_BASE) {
                _acc_spad.fill(inst.dest_addr, inst.accum_spad_id);
            } else {
                assert(0);
            }
            if (auto tile = inst.parent_tile.lock()) {
                tile->remaining_accum_io--;
                tile->remaining_computes--;
            } else {
                assert(0);
            }
            vector_pipeline.pop();
        }
    }

    /* LD instruction queue */
    // todo: ld_queue.cycle();
    if (!_ld_inst_queue.empty()) {
        Instruction &front = _ld_inst_queue.front();
        // spdlog::info("{}", front.repr());
        if (front.opcode == Opcode::MOVIN) {
            // bool prefetched = false;
            Sram *buffer;
            int buffer_id;
            if (front.dest_addr >= ACCUM_SPAD_BASE) {
                buffer = &_acc_spad;
                buffer_id = front.accum_spad_id;
            } else {
                buffer = &_spad;
                buffer_id = front.spad_id;
            }

            ast(!front.src_addrs.empty());

            auto accesses = MemoryAccess::from_instruction(
                front, generate_mem_access_id(), _config.dram_req_size, MemoryAccessType::READ,
                true, _id, _core_cycle, buffer_id);

            // xxx is this right? size, count<<src_addrs size
            buffer->reserve(front.dest_addr, buffer_id, front.size, accesses.size());
            if (auto tile = front.parent_tile.lock()) {
                tile->remaining_loads += accesses.size() - 1;
                tile->stat.memory_reads += accesses.size() * AddressConfig::alignment;
            } else {
                assert(0);
            }
            for (auto access : accesses) {
                push_memory_request(access);
            }
            _ld_inst_queue.pop();
        } else if (front.opcode == Opcode::PIM_HEADER || front.opcode == Opcode::PIM_GWRITE ||
                   front.opcode == Opcode::PIM_COMP || front.opcode == Opcode::PIM_READRES ||
                   front.opcode == Opcode::PIM_COMPS_READRES) {
            Sram *buffer;
            int buffer_id;
            if (front.dest_addr >= ACCUM_SPAD_BASE) {
                buffer = &_acc_spad;
                buffer_id = front.accum_spad_id;
            } else {
                buffer = &_spad;
                buffer_id = front.spad_id;
            }

            ast(!front.src_addrs.empty());

            MemoryAccess *mem_request =
                TransToMemoryAccess(front, _config.dram_req_size, _id, _core_cycle, buffer_id);

            if (front.opcode == Opcode::PIM_READRES || front.opcode == Opcode::PIM_COMPS_READRES)
                buffer->reserve(front.dest_addr, buffer_id, front.size, 1);

            push_memory_request(mem_request);
            _ld_inst_queue.pop();

        } else {
            assert(0);
        }
    }

    /* ST instruction queue */
    // todo: st_queue.cycle();
    if (!_st_inst_queue.empty()) {
        Instruction &front = _st_inst_queue.front();
        Sram *buffer;
        int buffer_id;
        if (front.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_acc_spad;
            buffer_id = front.accum_spad_id;
        } else {
            buffer = &_spad;
            buffer_id = front.spad_id;
        }
        // spdlog::info("{}", front.repr());
        if (buffer->check_hit(front.dest_addr, buffer_id) &&
            (front.opcode == Opcode::MOVOUT || front.opcode == Opcode::MOVOUT_POOL)) {
            auto accesses = MemoryAccess::from_instruction(
                front, generate_mem_access_id(), _config.dram_req_size, MemoryAccessType::WRITE,
                true, _id, _core_cycle, buffer_id);
            if (auto tile = front.parent_tile.lock()) {
                tile->remaining_accum_io += accesses.size() - 1;
                tile->stat.memory_writes += accesses.size() * AddressConfig::alignment;
            } else {
                assert(0);
            }
            for (auto access : accesses) {
                push_memory_request(access);
                _waiting_write_reqs++;
            }
            _st_inst_queue.pop();
        }
    }

    /* EX instruction queue */
    if (!_ex_inst_queue.empty()) {
        Instruction ready_inst = get_first_ready_ex_inst();

        if (ready_inst.valid)
            issue_ex_inst(ready_inst);
        else {
            /* Update memory stall stat */
        }
    }

    // if (!_ex_inst_queue.empty() && can_issue_compute(_ex_inst_queue.front())) {
    //     Instruction front = _ex_inst_queue.front();
    //     issue_ex_inst(front);
    // }
    // if (!_ex_inst_queue.empty()) {
    //     auto front = _ex_inst_queue.front();
    //     std::shared_ptr<Tile> parent_tile = front.parent_tile.lock();
    //     if (parent_tile == nullptr) {
    //         assert(0);
    //     }
    //     switch (front.opcode) {
    //         case Opcode::GEMM:
    //         case Opcode::GEMM_PRELOAD:
    //             if (_compute_pipeline.empty()) {
    //                 parent_tile->stat.memory_stalls++;
    //             }
    //             break;
    //         default:
    //             if (_vector_pipeline.empty()) {
    //                 parent_tile->stat.memory_stalls++;
    //             }
    //     }
    // }

    if (!_compute_pipeline.empty()) {
        auto parent_tile = _compute_pipeline.front().parent_tile.lock();
        if (parent_tile == nullptr) {
            assert(0);
        }
        parent_tile->stat.compute_cycles++;
    }
    for (auto &vector_pipeline : _vector_pipelines) {
        if (!vector_pipeline.empty()) {
            auto parent_tile = vector_pipeline.front().parent_tile.lock();
            if (parent_tile == nullptr) {
                assert(0);
            }
            parent_tile->stat.compute_cycles++;
        }
    }

    // xxx will it work well on double buffered code? no.
    bool is_idle = _compute_pipeline.empty();
    for (auto &vector_pipeline : _vector_pipelines) {
        is_idle = is_idle && vector_pipeline.empty();
    }
    if (is_idle) {
        _stat_memory_cycle++;

        if (_ex_inst_queue.empty()) {
            _store_memory_cycle++;
        } else {
            _load_memory_cycle++;
            switch (_ex_inst_queue.front().opcode) {
                case Opcode::GEMM:
                case Opcode::GEMM_PRELOAD:
                    _compute_memory_stall_cycle++;
                    break;
                case Opcode::LAYERNORM:
                    _layernorm_stall_cycle++;
                    break;
                case Opcode::SOFTMAX:
                    _softmax_stall_cycle++;
                    break;
                case Opcode::ADD:
                    _add_stall_cycle++;
                    break;
                case Opcode::GELU:
                    _gelu_stall_cycle++;
                    break;
            }
        }
    } else if (!_compute_pipeline.empty()) {
        _stat_matmul_cycle++;
    } else {
        // } else if (!_vector_pipeline.empty()) {
        // when element in vector pipeline
        for (auto &vector_pipeline : _vector_pipelines) {
            switch (vector_pipeline.front().opcode) {
                case Opcode::LAYERNORM:
                    _stat_layernorm_cycle++;
                    break;
                case Opcode::SOFTMAX:
                    _stat_softmax_cycle++;
                    break;
                case Opcode::ADD:
                    _stat_add_cycle++;
                    break;
                case Opcode::GELU:
                    _stat_gelu_cycle++;
                    break;
            }
        }
    }

    if (!running()) {
        _stat_idle_cycle++;
    }
    Core::cycle();
}

cycle_type SystolicWS::get_inst_compute_cycles(Instruction &inst) {
    return _config.core_height + _config.core_width + inst.size - 1;
    // -- comment out
    // return _config.core_height + _config.core_width - 2 + MAX(inst.size, 4);
    // -- comment out EOF
}

cycle_type SystolicWS::get_vector_compute_cycles(Instruction &inst) {
    cycle_type vec_op_iter = calculate_vector_op_iterations(inst.size);
    cycle_type add_tree_iter = calculate_add_tree_iterations(inst.size);
    cycle_type add_tree, scalar_ops, vector_ops;
    switch (inst.opcode) {
        case Opcode::LAYERNORM:
            add_tree = 2 * add_tree_iter * _config.add_tree_latency;
            scalar_ops = 2 * _config.scalar_mul_latency + _config.scalar_sqrt_latency;
            // 1 addition, 1 subtraction, 1 division, 2 multiplication.
            vector_ops = vec_op_iter * (2 * _config.add_latency + 3 * _config.mul_latency);
            return add_tree + scalar_ops + vector_ops;
        case Opcode::SOFTMAX:
            // 1 add tree, 1 compare tree
            add_tree = 2 * add_tree_iter * _config.add_tree_latency;
            vector_ops =
                vec_op_iter * (_config.add_latency + _config.exp_latency + _config.mul_latency);
            return add_tree + vector_ops;
        case Opcode::ADD:
            return vec_op_iter * _config.add_latency;
        case Opcode::GELU:
            return vec_op_iter * _config.gelu_latency;
        case Opcode::DUMMY:
            return 1;
    }
    spdlog::info("not configured operation. {}", inst.id);
    // assert(0);
    return 0;
}

cycle_type SystolicWS::calculate_add_tree_iterations(uint32_t vector_size) {
    uint32_t calculation_unit = _config.vector_core_width;
    if (vector_size <= calculation_unit) {
        return 1;
    }

    uint32_t ret = vector_size / calculation_unit;
    if (vector_size % calculation_unit != 0) {
        ret++;
    }
    return ret + calculate_add_tree_iterations(ret);
}

cycle_type SystolicWS::calculate_vector_op_iterations(uint32_t vector_size) {
    uint32_t calculation_unit = _config.vector_core_width;
    uint32_t ret = vector_size / calculation_unit;
    if (vector_size % calculation_unit != 0) {
        ret++;
    }
    return ret;
}

void SystolicWS::print_stats() {
    Core::print_stats();
    spdlog::info("Core [{}] : Systolic Inst Issue Count : {}", _id,
                 _stat_systolic_inst_issue_count);
    spdlog::info("Core [{}] : Systolic PRELOAD Issue Count : {}", _id,
                 _stat_systolic_preload_issue_count);
}

void SystolicWS::issue_ex_inst(Instruction inst) {
    // spdlog::info("cycle:{}, {}", _core_cycle, inst.repr());
    if (inst.opcode == Opcode::GEMM || inst.opcode == Opcode::GEMM_PRELOAD) {
        auto parent_tile = inst.parent_tile.lock();
        if (parent_tile == nullptr) {
            assert(0);
        }
        // spdlog::info("COMPUTE Start cycle: {} inst:{}", _core_cycle, inst.repr());
        parent_tile->stat.num_calculation += inst.tile_m * inst.tile_n * inst.tile_k;

        if (inst.opcode == Opcode::GEMM_PRELOAD) {
            _stat_systolic_preload_issue_count++;
            parent_tile->stat.weight_load_cycles += _config.core_width;  // fixme width
        }
        if (_compute_pipeline.empty() && inst.opcode == Opcode::GEMM_PRELOAD) {
            // TODO
            if (inst.is_gemv) {
                inst.start_cycle = _core_cycle + 8;
            } else {
                inst.start_cycle = _core_cycle + _config.core_width;
            }
            inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
        }
        if (_compute_pipeline.empty() && inst.opcode != Opcode::GEMM_PRELOAD) {
            inst.start_cycle = _core_cycle;
            inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
        }
        if (!_compute_pipeline.empty() && inst.opcode == Opcode::GEMM_PRELOAD) {
            // TODO
            // if (inst.is_gemv) {
            //     inst.start_cycle = _compute_pipeline.back().finish_cycle + 8;
            // } else{
            inst.start_cycle = _compute_pipeline.back().finish_cycle;
            // }
            inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
        }
        if (!_compute_pipeline.empty() && inst.opcode != Opcode::GEMM_PRELOAD) {
            if (_core_cycle < _compute_pipeline.back().finish_cycle - _config.core_height -
                                  _config.core_width + 1) {
                inst.start_cycle = _compute_pipeline.back().finish_cycle - _config.core_height -
                                   _config.core_width + 1;
                inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
            } else {
                inst.start_cycle = _core_cycle;
                inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
            }
        }

        // -- comment out
        // if (!_compute_pipeline.empty()) {
        //     /* Preload can be hided */
        //     uint32_t offset = _compute_pipeline.back().size;
        //     // xxx why 4?
        //     // maybe pushing to the systolic array input queue. 4 cycles to start?
        //     offset = MAX(offset, 4);
        //     if (inst.opcode == Opcode::GEMM_PRELOAD) {
        //         // State mul-pre
        //         parent_tile->stat.weight_load_cycles += _config.core_width;  // fixme: width
        //         offset = _config.core_width;                                 // fixme: width
        //     }
        //     inst.start_cycle = _compute_pipeline.back().start_cycle + offset;
        // } else {
        //     inst.start_cycle = _core_cycle;
        //     /* Preload weight to systolic array*/
        //     if (inst.opcode == Opcode::GEMM_PRELOAD) {
        //         /* Weight preload  from buffer latecny + WEight preload
        //          * latency */
        //         inst.start_cycle += _config.core_height + _config.core_width - 1;  // fixme:
        //     }
        // }

        // inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
        // spdlog::info("finish_cycle: {}", inst.finish_cycle);
        // -- comment out EOF

        _compute_pipeline.push(inst);
        _stat_systolic_inst_issue_count++;
    } else if (inst.opcode == Opcode::COMP || inst.opcode == Opcode::IM2COL ||
               inst.opcode == Opcode::LAYERNORM || inst.opcode == Opcode::SOFTMAX ||
               inst.opcode == Opcode::ADD || inst.opcode == Opcode::GELU ||
               inst.opcode == Opcode::DUMMY) {  // vector unit compute
        // spdlog::info("COMPUTE Start cycle: {} inst:{}", _core_cycle, inst.repr());
        std::queue<Instruction> *least_filled_vpu;
        cycle_type finish_cycle = std::numeric_limits<uint64_t>::max();
        for (auto &vector_pipeline : _vector_pipelines) {
            if (vector_pipeline.empty()) {
                least_filled_vpu = &vector_pipeline;
                finish_cycle = _core_cycle;
                break;
            }
            if (vector_pipeline.back().finish_cycle < finish_cycle) {
                least_filled_vpu = &vector_pipeline;
                finish_cycle = _core_cycle;
            }
        }
        inst.start_cycle = finish_cycle;
        inst.finish_cycle = inst.start_cycle + get_vector_compute_cycles(inst);
        least_filled_vpu->push(inst);

        {
            // if (!_vector_pipeline.empty()) {
            //     // loading latency
            //     inst.start_cycle = _vector_pipeline.back().finish_cycle + 1;
            // } else {
            //     inst.start_cycle = _core_cycle;
            // }
            // inst.finish_cycle = inst.start_cycle + get_vector_compute_cycles(inst);
            // _vector_pipeline.push(inst);
        }
    }
    // _ex_inst_queue.pop();  // TODO: erase corresponding inst

    // if dest_addr is on sram, count up. -> wait for _compute_pipeline to
    // finish calculation
    if (_acc_spad.check_allocated(inst.dest_addr, inst.accum_spad_id)) {
        // spdlog::info("allocated: {}", inst.repr());
        _acc_spad.count_up(inst.dest_addr, inst.accum_spad_id);
    }
    // if dest_addr is not on sram, initialize. -> wait for
    // _compute_pipeline to finish calculation
    else {
        // spdlog::info("reserve: {}", inst.repr());
        // spdlog::info("reserve, dest_addr:{:x}, spad_id:{}, size:{}", inst.dest_addr,
        //  inst.accum_spad_id, inst.size);
        _acc_spad.reserve(inst.dest_addr, inst.accum_spad_id, inst.size, 1);
    }
}

// TODO: execution instruction should need to issue multiple execution instructions for multiple
// pipelines
// Instruction SystolicWS::get_first_ready_ex_inst() {
//     for (auto it = _ex_inst_queue.begin(); it != _ex_inst_queue.end(); ++it) {
//         if (can_issue_compute(*it)) {
//             Instruction inst = std::move(*it);
//             _ex_inst_queue.erase(it);
//             return std::move(inst);
//         }
//         // break;  // force in-order remove this
//     }
//     return Instruction{.valid = false};
// }

Instruction SystolicWS::get_first_ready_ex_inst() {
    Instruction inst = _ex_inst_queue.front();
    if (can_issue_compute(inst)) {
        _ex_inst_queue.pop();
        return std::move(inst);
    }

    return Instruction{.valid = false};
}