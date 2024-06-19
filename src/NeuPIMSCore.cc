#include "NeuPIMSCore.h"

#include <memory>

#include "Stat.h"
#include "helper/HelperFunctions.h"

NeuPIMSCore::NeuPIMSCore(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _compute_end_cycle(0),
      _stat_idle_cycle(0),
      _stat_compute_cycle(0),
      _stat_memory_cycle(0),
      _stat_vec_compute_cycle(0),
      _stat_vec_memory_cycle(0),
      _memory_stall_cycle(0),
      _compute_memory_stall_cycle(0),
      _vector_memory_stall_cycle(0),
      _layernorm_stall_cycle(0),
      _softmax_stall_cycle(0),
      _add_stall_cycle(0),
      _gelu_stall_cycle(0),
      _load_memory_cycle(0),
      _store_memory_cycle(0),
      _stat_vec_idle_cycle(0),
      _stat_matmul_cycle(0),
      _stat_layernorm_cycle(0),
      _stat_add_cycle(0),
      _stat_gelu_cycle(0),
      _stat_softmax_cycle(0),
      _spad(Sram(config, _core_cycle, false)),
      _acc_spad(Sram(config, _core_cycle, true)),
      _pim_spad(Sram(config, _core_cycle, false)),
      _pim_acc_spad(Sram(config, _core_cycle, true)) {
    _waiting_write_reqs = 0;
    _running_layer = -1;
    _current_spad = 0;
    _current_acc_spad = 0;
    _memory_request_queues1.resize(_config.dram_channels);
    _memory_request_queues2.resize(_config.dram_channels);
    _vector_pipelines.resize(_config.vector_core_count);
}
bool NeuPIMSCore::can_issue_pim() { return _pim_tiles.empty(); }
// if next_tile.accum == true
//     there is no need for _acc_spad availability
// else
//     need to check _acc_spad availability
bool NeuPIMSCore::can_issue(Tile &next_tile) {
    if (_tiles.empty()) {
        return true;
    }
    // FIXME: disable double buffering
    auto next_spad = _current_spad ^ 1;
    auto next_acc_spad = _current_acc_spad ^ 1;

    // load and compute should be over at the other side
    for (auto tile : _tiles) {
        if (tile->spad_id == next_spad) {
            if ((tile->remaining_loads != 0) || (tile->remaining_computes != 0)) {
                // spdlog::info("issue failed. accum true. spad id {}", tile->spad_id);
                // std::cout << tile->remaining_loads << "\t" << tile->remaining_computes <<
                // std::endl;
                return false;
            }
        }
    }
    if (!next_tile.accum) {
        // load, compute and store should be over at the other side
        for (auto tile : _tiles) {
            if (tile->accum_spad_id == next_acc_spad) {
                if (tile->remaining_accum_io != 0) {
                    // spdlog::info("issue failed. accum false. acc spad id {}",
                    // tile->accum_spad_id); std::cout << tile->remaining_accum_io << std::endl;
                    return false;
                }
            }
        }
    }
    // spdlog::info("issue succeeded. accum {}, spad id {}, acc spad id {}",
    //              next_tile.accum ? "true" : "false", next_spad, next_acc_spad);
    return true;
}

// switch tiles to shared ptr
// todo: check tile start cycle
void NeuPIMSCore::issue(Tile &in_tile) {
    spdlog::info("tile issued {}", in_tile.repr());
    auto tile = std::make_shared<Tile>(in_tile);
    tile->stat = TileStat(_core_cycle);
    if (tile->skip) {
        tile->status = Tile::Status::FINISH;
        _finished_tiles.push(tile);
        return;
    }
    /* Double buffer */
    _current_spad = (_current_spad + 1) % 2;
    _spad.flush(_current_spad);
    tile->spad_id = _current_spad;
    if (!tile->accum) {
        /* Accumeulate tile uses same acc spad buffer */
        // accumulate to same acc_spad if K > 1
        _current_acc_spad = (_current_acc_spad + 1) % 2;
        _acc_spad.flush(_current_acc_spad);
    }
    tile->accum_spad_id = _current_acc_spad;
    tile->status = Tile::Status::RUNNING;
    if (_running_layer != tile->operation_id) {
        _running_layer = tile->operation_id;
    }

    tile->remaining_loads = 0;
    tile->remaining_computes = 0;
    tile->remaining_accum_io = 0;
    for (auto &inst : tile->instructions) {
        inst.parent_tile = std::weak_ptr<Tile>(tile);
        inst.spad_id = tile->spad_id;
        inst.accum_spad_id = tile->accum_spad_id;
        Sram *buffer;
        int buffer_id;
        if (inst.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_acc_spad;
            buffer_id = tile->accum_spad_id;
        } else {
            buffer = &_spad;
            buffer_id = tile->spad_id;
        }
        if (inst.opcode == Opcode::PIM_HEADER) {
            assert(0);
            _ld_inst_queue_for_sa.push(inst);
        } else if (inst.opcode == Opcode::MOVIN || inst.opcode == Opcode::PIM_GWRITE ||
                   inst.opcode == Opcode::PIM_COMP || inst.opcode == Opcode::PIM_READRES ||
                   inst.opcode == Opcode::PIM_COMPS_READRES) {
            switch (inst.opcode) {
                case Opcode::PIM_GWRITE:
                case Opcode::PIM_COMP:
                case Opcode::PIM_READRES:
                case Opcode::PIM_COMPS_READRES:
                    spdlog::info("pim operations unreachable.");
                    assert(0);
            }
            if (!buffer->check_allocated(inst.dest_addr, buffer_id) &&
                buffer->check_remain(inst.size, buffer_id)) {
                tile->remaining_loads++;
                _ld_inst_queue_for_sa.push(inst);
            } else {
                spdlog::info("sram size: {} / sram used: {}", _config.sram_size KB / 2,
                             buffer->get_current_size(buffer_id));
                spdlog::info("instruction destination address {:x}", inst.dest_addr);
                spdlog::info("failed to allocate {} on sram.", inst.size);
                buffer->print_all(buffer_id);
                /*Invalid state */
                assert(0);
            }
        } else if (inst.opcode == Opcode::MOVOUT || inst.opcode == Opcode::MOVOUT_POOL) {
            tile->remaining_accum_io++;
            _st_inst_queue_for_sa.push(inst);
        } else {
            /* Ex inst queue */
            tile->remaining_accum_io++;
            tile->remaining_computes++;
            _ex_inst_queue_for_sa.push(inst);
        }
    }
    // spdlog::info("tile pushed to core._tiles {}", tile.repr());
    _tiles.push_back(tile);
}

void NeuPIMSCore::issue_pim(Tile &in_tile) {
    spdlog::info("pim tile issued {}", in_tile.repr());
    auto tile = std::make_shared<Tile>(in_tile);
    tile->stat = TileStat(_core_cycle);
    if (tile->skip) {
        tile->status = Tile::Status::FINISH;
        _finished_tiles.push(tile);
        return;
    }

    _pim_spad.flush(0);
    _pim_acc_spad.flush(0);

    tile->spad_id = 0;
    tile->accum_spad_id = 0;
    tile->status = Tile::Status::RUNNING;
    if (_running_layer != tile->operation_id) {
        _running_layer = tile->operation_id;
    }

    tile->remaining_loads = 0;
    tile->remaining_computes = 0;
    tile->remaining_accum_io = 0;
    for (auto &inst : tile->instructions) {
        inst.is_pim_inst = true;
        inst.parent_tile = std::weak_ptr<Tile>(tile);
        inst.spad_id = tile->spad_id;
        inst.accum_spad_id = tile->accum_spad_id;
        Sram *buffer;
        int buffer_id;
        if (inst.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_pim_acc_spad;
            buffer_id = tile->accum_spad_id;
        } else {
            buffer = &_pim_spad;
            buffer_id = tile->spad_id;
        }
        if (inst.opcode == Opcode::PIM_HEADER) {
            _ld_inst_queue_for_pim.push(inst);
        } else if (inst.opcode == Opcode::MOVIN || inst.opcode == Opcode::PIM_GWRITE ||
                   inst.opcode == Opcode::PIM_COMP || inst.opcode == Opcode::PIM_READRES ||
                   inst.opcode == Opcode::PIM_COMPS_READRES) {
            if (!buffer->check_allocated(inst.dest_addr, buffer_id) &&
                buffer->check_remain(inst.size, buffer_id)) {
                tile->remaining_loads++;
                _ld_inst_queue_for_pim.push(inst);
            } else {
                spdlog::info("check_allocated: {}",
                             buffer->check_allocated(inst.dest_addr, buffer_id));
                spdlog::info("check_remain: {}", buffer->check_remain(inst.size, buffer_id));
                spdlog::info("sram size: {} / sram used: {}", _config.sram_size KB / 2,
                             buffer->get_current_size(buffer_id));
                spdlog::info("instruction destination address {:x}", inst.dest_addr);
                spdlog::info("failed to allocate {} on sram.", inst.size);
                buffer->print_all(buffer_id);
                /*Invalid state */
                assert(0);
            }
        } else if (inst.opcode == Opcode::MOVOUT || inst.opcode == Opcode::MOVOUT_POOL) {
            tile->remaining_accum_io++;
            _st_inst_queue_for_pim.push(inst);
        } else {
            /* Ex inst queue */
            tile->remaining_accum_io++;
            tile->remaining_computes++;
            _ex_inst_queue_for_pim.push(inst);
        }
    }
    // spdlog::info("tile pushed to core._tiles {}", tile.repr());
    _pim_tiles.push_back(tile);
}

// return EMPTY or FINISHED
Ptr<Tile> NeuPIMSCore::pop_finished_tile() {
    if (_finished_tiles.empty()) {
        return nullptr;
    }

    auto result = _finished_tiles.front();
    result->stat.end_cycle = _core_cycle;
    _finished_tiles.pop();
    return result;
}

void NeuPIMSCore::cycle() {
    _core_cycle++;
    _spad.cycle();
    _acc_spad.cycle();

    // spdlog::info("current spad {}", _current_spad);
    for (auto tile_it = _tiles.begin(); tile_it != _tiles.end();) {
        auto tile = *tile_it;
        // spdlog::info(
        //     "tile index: {}, tile remain_accum_io: {}, remain_computes: {}, remain_loads: {}",
        //     tile_it - _tiles.begin(), tile->remaining_accum_io, tile->remaining_computes,
        //     tile->remaining_loads);
        if ((tile->remaining_accum_io == 0) && (tile->remaining_computes == 0) &&
            (tile->remaining_loads == 0)) {
            tile->status = Tile::Status::FINISH;
            _finished_tiles.push(tile);
            tile_it = _tiles.erase(tile_it);
        } else {
            tile_it++;
        }
    }
    for (auto tile_it = _pim_tiles.begin(); tile_it != _pim_tiles.end();) {
        auto tile = *tile_it;
        // spdlog::info("tile remain_accum_io: {}, remain_computes: {}, remain_loads: {}",
        //              tile->remaining_accum_io, tile->remaining_computes, tile->remaining_loads);
        if ((tile->remaining_accum_io == 0) && (tile->remaining_computes == 0) &&
            (tile->remaining_loads == 0)) {
            tile->status = Tile::Status::FINISH;
            _finished_tiles.push(tile);
            tile_it = _pim_tiles.erase(tile_it);
        } else {
            tile_it++;
        }
    }
    // xxx : need logic for _finished_pim_tiles?
}

bool NeuPIMSCore::running() {
    bool running = false;
    running = running || _tiles.size() > 0;
    running = running || !_compute_pipeline.empty();
    running = running || _waiting_write_reqs != 0;
    running = running || !_ld_inst_queue_for_sa.empty();
    running = running || !_st_inst_queue_for_sa.empty();
    running = running || !_ex_inst_queue_for_sa.empty();
    bool temp = running;
    for (auto &vector_pipeline : _vector_pipelines) {
        running = running || !vector_pipeline.empty();
    }

    int status_check_interval = 1000000;
    if (_core_cycle % status_check_interval == 0 && false) {
        spdlog::info("------Simulator Status Check------");
        spdlog::info("tiles: {}", _tiles.size() > 0);
        spdlog::info("pim_tiles: {}", _pim_tiles.size() > 0);
        spdlog::info("_compute_pipeline: {}", !_compute_pipeline.empty());
        spdlog::info("_waiting_write_reqs: {}", _waiting_write_reqs != 0);
        spdlog::info("_ld_inst_queue_for_sa: {}", !_ld_inst_queue_for_sa.empty());
        spdlog::info("_st_inst_queue_for_sa: {}", !_st_inst_queue_for_sa.empty());
        spdlog::info("_ex_inst_queue_for_sa: {}", !_ex_inst_queue_for_sa.empty());
        spdlog::info("due to vector: {}", running);
        for (auto &vector_pipeline : _vector_pipelines) {
            spdlog::info("vp size: {}", vector_pipeline.size());
        }
        spdlog::info("----------------------------------");
    }

    return running;
}

// push into target channel memory request queue
void NeuPIMSCore::push_memory_request1(MemoryAccess *request) {
    int channel = AddressConfig::mask_channel(request->dram_address);
    _memory_request_queues1[channel].push(request);
}

void NeuPIMSCore::push_memory_request2(MemoryAccess *request) {
    int channel = AddressConfig::mask_channel(request->dram_address);
    _memory_request_queues2[channel].push(request);
}

void NeuPIMSCore::push_memory_response(MemoryAccess *response) {
    assert(!response->request);  // can only push response

    Sram *acc_spad = &_acc_spad;
    Sram *spad = &_spad;
    uint32_t buf_id;
    if (auto parent = response->parent_tile.lock()) {
        if (parent->stage_platform == StagePlatform::PIM) {
            spad = &_pim_spad;
            acc_spad = &_pim_acc_spad;
        }
    }

    bool is_write = response->req_type == MemoryAccessType::WRITE;
    bool is_read = response->req_type == MemoryAccessType::READ;
    if (auto tile = response->parent_tile.lock()) {
        if (is_write) {
            tile->remaining_accum_io--;
        } else {
            tile->remaining_loads--;
        }
    } else {
        assert(0);
    }
    if (is_write) {
        _waiting_write_reqs--;
    } else if (response->req_type == MemoryAccessType::P_HEADER ||
               response->req_type == MemoryAccessType::GWRITE ||
               response->req_type == MemoryAccessType::COMP) {
        // pim_header, pim_gwrite, pim_comp
        // pim_header request does not receive response
    } else if (response->spad_address >= ACCUM_SPAD_BASE) {
        // spdlog::info("{} response to accum_spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit

        // case2: load bias to _accum_spad
        acc_spad->fill(response->spad_address, response->buffer_id);
    } else {
        // spdlog::info("{} response to _spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit
        // case3: load activation or weight to _spad
        spad->fill(response->spad_address, response->buffer_id);
    }
    delete response;
}

// -- seems it is not used.
void NeuPIMSCore::pim_push_memory_response(MemoryAccess *response) {
    assert(!response->request);  // can only push response

    bool is_write = response->req_type == MemoryAccessType::WRITE;
    bool is_read = response->req_type == MemoryAccessType::READ;
    if (auto tile = response->parent_tile.lock()) {
        if (is_write) {
            tile->remaining_accum_io--;
        } else {
            tile->remaining_loads--;
        }
    } else {
        assert(0);
    }
    if (is_write) {
        _waiting_write_reqs--;
    } else if (response->req_type == MemoryAccessType::P_HEADER ||
               response->req_type == MemoryAccessType::GWRITE ||
               response->req_type == MemoryAccessType::COMP) {
        // pim_header, pim_gwrite, pim_comp
        // pim_header request does not receive response
    } else if (response->spad_address >= ACCUM_SPAD_BASE) {
        // spdlog::info("{} response to accum_spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit

        // case2: load bias to _accum_spad
        _pim_acc_spad.fill(response->spad_address, response->buffer_id);
    } else {
        // spdlog::info("{} response to _spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit
        // case3: load activation or weight to _spad
        _pim_spad.fill(response->spad_address, response->buffer_id);
    }
    delete response;
}

// checks if inputs are loaded.
bool NeuPIMSCore::can_issue_compute(Instruction &inst) {
    bool result = true;

    // src addr: spad key
    for (addr_type addr : inst.src_addrs) {
        if (inst.src_from_accum && addr >= ACCUM_SPAD_BASE) {
            result = result && _acc_spad.check_hit(addr, inst.accum_spad_id);
            continue;
        }

        result = result && _spad.check_hit(addr, inst.spad_id);
    }
    if (!result) {
        for (addr_type addr : inst.src_addrs) {
            // spdlog::info("NeuPIMSCore[{}] Dependency fail : {:x} , {} for {}", _id, addr,
            //              _spad.check_hit(addr, inst.spad_id), inst.repr());
        }
    }
    // spdlog::info("can_issue_compute: {} {}", result ? "okay" : "nope", inst.repr());
    return result;
}

bool NeuPIMSCore::pim_can_issue_compute(Instruction &inst) {
    bool result = true;

    // src addr: spad key
    for (addr_type addr : inst.src_addrs) {
        if (inst.src_from_accum && addr >= ACCUM_SPAD_BASE) {
            result = result && _pim_acc_spad.check_hit(addr, inst.accum_spad_id);
            continue;
        }

        result = result && _pim_spad.check_hit(addr, inst.spad_id);
    }
    if (!result) {
        for (addr_type addr : inst.src_addrs) {
            // spdlog::info("NeuPIMSCore[{}] Dependency fail : {:x} , {} for {}", _id, addr,
            //              _spad.check_hit(addr, inst.spad_id), inst.repr());
        }
    }
    // spdlog::info("can_issue_compute: {} {}", result ? "okay" : "nope", inst.repr());
    return result;
}

void NeuPIMSCore::print_stats() {
    spdlog::info(
        "NeuPIMSCore [{}] : MatMul cycle {} LayerNorm cycle {} Softmax cycle {} "
        "Add cycle {}  Gelu cycle {}",
        _id, _stat_matmul_cycle, _stat_layernorm_cycle, _stat_softmax_cycle, _stat_add_cycle,
        _stat_gelu_cycle);
    spdlog::info(
        "NeuPIMSCore [{}] : MatMul stall cycle {} LayerNorm stall cycle {} "
        "Softmax stall cycle {} Add stall cycle {} Gelu stall cycle {}",
        _id, _compute_memory_stall_cycle, _layernorm_stall_cycle, _softmax_stall_cycle,
        _add_stall_cycle, _gelu_stall_cycle);

    spdlog::info(
        "NeuPIMSCore [{}] : Load stall cycle {} Store stall cycle {} "
        "Total memory stall {} Idle cycle {}",
        _id, _load_memory_cycle, _store_memory_cycle,

        _stat_memory_cycle, _stat_idle_cycle);
    // spdlog::info(
    //     "NeuPIMSCore [{}] : Compute cycle {} Memory Stall Cycle {} Idle Cycle {}",
    //     _id, _stat_compute_cycle, _stat_memory_cycle, _stat_idle_cycle);

    // spdlog::info(
    //     "NeuPIMSCore [{}] : Vec Compute cycle {} Vec Memory Stall Cycle {} Vec Idle
    //     " "Cycle {}", _id, _stat_vec_compute_cycle, _stat_vec_memory_cycle,
    //     _stat_vec_idle_cycle);
    spdlog::info("NeuPIMSCore [{}] : Total cycle: {}", _id, _core_cycle);
}

void NeuPIMSCore::log() {}