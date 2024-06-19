#include "Core.h"

#include <memory>

#include "Stat.h"
#include "helper/HelperFunctions.h"

Core::Core(uint32_t id, SimulationConfig config)
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
      _acc_spad(Sram(config, _core_cycle, true)) {
    _waiting_write_reqs = 0;
    _running_layer = -1;
    _current_spad = 0;
    _current_acc_spad = 0;
    _memory_request_queues.resize(_config.dram_channels);
    _vector_pipelines.resize(_config.vector_core_count);
}

// if next_tile.accum == true
//     there is no need for _acc_spad availability
// else
//     need to check _acc_spad availability
bool Core::can_issue(Tile &next_tile) {
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
void Core::issue(Tile &in_tile) {
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
            _ld_inst_queue.push(inst);
        } else if (inst.opcode == Opcode::MOVIN || inst.opcode == Opcode::PIM_GWRITE ||
                   inst.opcode == Opcode::PIM_COMP || inst.opcode == Opcode::PIM_READRES ||
                   inst.opcode == Opcode::PIM_COMPS_READRES) {
            if (!buffer->check_allocated(inst.dest_addr, buffer_id) &&
                buffer->check_remain(inst.size, buffer_id)) {
                tile->remaining_loads++;
                _ld_inst_queue.push(inst);
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
            _st_inst_queue.push(inst);
        } else {
            /* Ex inst queue */
            tile->remaining_accum_io++;
            tile->remaining_computes++;
            _ex_inst_queue.push_back(inst);
        }
    }
    // spdlog::info("tile pushed to core._tiles {}", tile.repr());
    _tiles.push_back(tile);
}

// return EMPTY or FINISHED
Ptr<Tile> Core::pop_finished_tile() {
    if (_finished_tiles.empty()) {
        return nullptr;
    }

    auto result = _finished_tiles.front();
    result->stat.end_cycle = _core_cycle;
    _finished_tiles.pop();
    return result;
}

// get instruction from tile
// if instruction is ld
//   put into `_ld_inst_queue`
// elif instruction is st
//   put into `_st_inst_queue`
// elif instruction is execution
//   put into `_ex_inst_queue`
// if the tile runs out of instructions
//  pop tile
void Core::cycle() {
    _core_cycle++;
    _spad.cycle();
    _acc_spad.cycle();

    for (auto tile_it = _tiles.begin(); tile_it != _tiles.end();) {
        auto tile = *tile_it;
        // spdlog::info("tile remain_accum_io: {}, remain_computes: {}, remain_loads: {}",
        //              tile->remaining_accum_io, tile->remaining_computes, tile->remaining_loads);
        if ((tile->remaining_accum_io == 0) && (tile->remaining_computes == 0) &&
            (tile->remaining_loads == 0)) {
            tile->status = Tile::Status::FINISH;
            _finished_tiles.push(tile);
            tile_it = _tiles.erase(tile_it);
        } else {
            tile_it++;
        }
    }
}

bool Core::running() {
    bool running = false;
    running = running || _tiles.size() > 0;
    running = running || !_compute_pipeline.empty();
    running = running || !_vector_pipeline.empty();  // Vector unit (Might need to modify)
    running = running || _waiting_write_reqs != 0;
    running = running || !_ld_inst_queue.empty();
    running = running || !_st_inst_queue.empty();
    running = running || !_ex_inst_queue.empty();

    for (auto &vector_pipeline : _vector_pipelines) {
        running = running || !vector_pipeline.empty();
    }

    // spdlog::info("Core::{} because {}", running ? "running" : "idle",
    //              !_vector_pipeline.empty()
    //                  ? "vector_pipeline"
    //                  : (_waiting_write_reqs
    //                         ? "waiting_write_reqs"
    //                         : (!_ld_inst_queue.empty()
    //                                ? "ld_inst_queue"
    //                                : (!_ex_inst_queue.empty()
    //                                       ? "ex_inst_queue"
    //                                       : (!_st_inst_queue.empty()
    //                                              ? "st_inst_queue"
    //                                              : (_tiles.size() > 0 ? "tiles" : "???"))))));

    return running;
}

// push into target channel memory request queue
void Core::push_memory_request(MemoryAccess *request) {
    int channel = AddressConfig::mask_channel(request->dram_address);

    // if (ch != channel) {
    //     spdlog::info("channel: {}, ch: {}", channel, ch);
    //     assert(0);
    // }

    // if (channel == 0) {
    //     std::string red = "\033[1;31m";
    //     std::string reset = "\033[0m";
    //     spdlog::info("{}Core push_mem_req(cid:{}) {}{}", red, channel,
    //                  memAccessTypeString(request->req_type), reset);
    // }

    _memory_request_queues[channel].push(request);
}

void Core::push_memory_response(MemoryAccess *response) {
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
        _acc_spad.fill(response->spad_address, response->buffer_id);
    } else {
        // spdlog::info("{} response to _spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit
        // case3: load activation or weight to _spad
        _spad.fill(response->spad_address, response->buffer_id);
    }
    delete response;
}

// checks if inputs are loaded.
bool Core::can_issue_compute(Instruction &inst) {
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
            // spdlog::info("Core[{}] Dependency fail : {:x} , {} for {}", _id, addr,
            //              _spad.check_hit(addr, inst.spad_id), inst.repr());
        }
    }
    // spdlog::info("can_issue_compute: {} {}", result ? "okay" : "nope", inst.repr());
    return result;
}

void Core::print_stats() {
    spdlog::info(
        "Core [{}] : MatMul cycle {} LayerNorm cycle {} Softmax cycle {} "
        "Add cycle {}  Gelu cycle {}",
        _id, _stat_matmul_cycle, _stat_layernorm_cycle, _stat_softmax_cycle, _stat_add_cycle,
        _stat_gelu_cycle);
    spdlog::info(
        "Core [{}] : MatMul stall cycle {} LayerNorm stall cycle {} "
        "Softmax stall cycle {} Add stall cycle {} Gelu stall cycle {}",
        _id, _compute_memory_stall_cycle, _layernorm_stall_cycle, _softmax_stall_cycle,
        _add_stall_cycle, _gelu_stall_cycle);

    spdlog::info(
        "Core [{}] : Load stall cycle {} Store stall cycle {} "
        "Total memory stall {} Idle cycle {}",
        _id, _load_memory_cycle, _store_memory_cycle,

        _stat_memory_cycle, _stat_idle_cycle);
    // spdlog::info(
    //     "Core [{}] : Compute cycle {} Memory Stall Cycle {} Idle Cycle {}",
    //     _id, _stat_compute_cycle, _stat_memory_cycle, _stat_idle_cycle);

    // spdlog::info(
    //     "Core [{}] : Vec Compute cycle {} Vec Memory Stall Cycle {} Vec Idle
    //     " "Cycle {}", _id, _stat_vec_compute_cycle, _stat_vec_memory_cycle,
    //     _stat_vec_idle_cycle);
    spdlog::info("Core [{}] : Total cycle: {}", _id, _core_cycle);
}