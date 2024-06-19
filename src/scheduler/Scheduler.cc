#include "Scheduler.h"

#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type *core_cycle)
    : _config(config), _core_cycle(core_cycle), _cycles(0) {
    _max_batch_size = 512;  // config.max_batch_size;  // config.max_batch_size;
    _model_program = nullptr;
    _active_reqs = 0;
    _next_ch = 0;
}

void Scheduler::launch(Ptr<Model> model) {
    // assign model
    // make modelprogram with batched request and model in schedule
    _model = model;
    spdlog::info("MODEL {} Launched in Scheduler", model->get_name());
}

void Scheduler::batch_request(Ptr<InferRequest> request) {
    if (!request->is_initiated) {
        _active_reqs++;

        // NOTE: KV cache generation in scheduler, WRITE in ModelProgram
        uint32_t h = _config.model_n_head / _config.n_tp;
        uint32_t d_k = _config.model_n_embd / _config.model_n_head;
        uint32_t seq_len = request->input_size;

        std::vector<uint32_t> dim_key{h, d_k, seq_len};
        std::vector<uint32_t> dim_value{h, seq_len, d_k};

        for (int layer = 0; layer < _config.model_n_layer; ++layer) {
            // in case of request that is not `initiated`, construct pim tensor
            if (_config.run_mode == RunMode::NPU_ONLY) {
                // set KV cache's produce=true -> ModelProgram
                auto k = std::make_shared<NPUTensor>(
                    name_gen(std::to_string(request->id), "KEY", std::to_string(layer)), dim_key,
                    NPUTensorKVType::KEY, true);
                auto v = std::make_shared<NPUTensor>(
                    name_gen(std::to_string(request->id), "VALUE", std::to_string(layer)),
                    dim_value, NPUTensorKVType::VALUE, true);
                request->K_cache.push_back(k);
                request->V_cache.push_back(v);
            } else {
                // PIM tensor
                uint32_t ch = _next_ch % _config.dram_channels;
                spdlog::info("Scheduler allocate request#{}(seq_len:{}) to channel {}<<",
                             request->id, seq_len, ch);
                auto k = std::make_shared<PIMTensor>(
                    name_gen(std::to_string(request->id), "KEY", std::to_string(layer)), ch,
                    dim_key, PIMTensorKVType::KEY, true);
                auto v = std::make_shared<PIMTensor>(
                    name_gen(std::to_string(request->id), "VALUE", std::to_string(layer)), ch,
                    dim_value, PIMTensorKVType::VALUE, true);
                request->K_cache.push_back(k);
                request->V_cache.push_back(v);
            }
        }

        request->is_initiated = true;
        _next_ch++;
    }
    _breq.push_back(request);
}

void Scheduler::make_program() {
    spdlog::info(">> MAKE NEW PROGRAM (breq.size: {}) <<", _breq.size());
    std::shared_ptr<BatchedRequest> breq = std::make_shared<BatchedRequest>(_breq);

    _model_program = std::make_unique<ModelProgram>(_model, breq);
    refresh_status();
}

void Scheduler::cycle() {
    _cycles++;

    // << Scheduling decision trigger: when meet every following conditions >>
    // 1. request_queue is not empty
    // 2. there is no running model_program (iteration completed)
    // 3. scheduling condition:
    //    e.g.) request_queue size > 5, or waiting_time > 100
    if (_model_program == nullptr && _breq.size() > 0) {
        make_program();
    }
}

void Scheduler::add_request(std::shared_ptr<InferRequest> request) {
    // spdlog::info("Scheduler::add_request()");
    _request_queue.push_back(request);
}

bool Scheduler::has_completed_request() { return !_completed_request_queue.empty(); }

std::shared_ptr<InferRequest> Scheduler::pop_completed_request() {
    // spdlog::info("Scheduler::pop_completed_request()");
    auto completed_req = _completed_request_queue.front();
    _completed_request_queue.pop();
    return completed_req;
}

Tile &Scheduler::top_tile(uint32_t core_id) {
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue.empty()) {
        return empty_tile;
    } else {
        Tile &tile = _executable_tile_queue.front();
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            return tile;
        }
    }
}

// TODO: Add base address for each addr in tiles / XXX: < necessary comment?
// TODO: something wrong with functionality. seems it's not a necessary function
void Scheduler::get_tile(uint32_t core_id) {
    if (_executable_tile_queue.empty()) {
        return;
    } else {
        Tile &tile = _executable_tile_queue.front();
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        } else {
            _active_operation_stats[tile.operation_id].launched_tiles++;
            _executable_tile_queue.pop_front();
            spdlog::debug("Operation {} Core {} Get Tile at {}", tile.optype, core_id,
                          *_core_cycle);
            return;
        }
    }
}

//  update operation stat
//  if operation is finished
//      apply to _model_program
void Scheduler::finish_tile(uint32_t core_id, Tile &tile) {
    spdlog::debug("Tile {} Core {} Finish Tile at {}", tile.operation_id, core_id, *_core_cycle);
    assert(_active_operation_stats.find(tile.operation_id) != _active_operation_stats.end());
    assert(_finished_operation_stats.find(tile.operation_id) == _finished_operation_stats.end());
    assert(_active_operation_stats[tile.operation_id].remain_tiles > 0);
    _active_operation_stats[tile.operation_id].remain_tiles--;
    _model_program->finish_operation_tile(tile);

    if (_active_operation_stats[tile.operation_id].remain_tiles == 0) {
        spdlog::info("Layer {} finish at {}", _active_operation_stats[tile.operation_id].name,
                     *_core_cycle);
        spdlog::info("Total compute time {}",
                     *_core_cycle - _active_operation_stats[tile.operation_id].start_cycle);
        _model_program->finish_operation(tile.operation_id);
        _finished_operation_stats[tile.operation_id] = _active_operation_stats[tile.operation_id];
        _active_operation_stats.erase(tile.operation_id);
    }
    refresh_status();
}

bool Scheduler::empty() { return _model_program == nullptr; }
bool Scheduler::running() { return !_request_queue.empty() || !_completed_request_queue.empty(); }

void Scheduler::finish_program() {
    spdlog::info("Model finish at {}", *_core_cycle);

    _model_program->log();

    // < todos when the model program has finished >
    // - increment `generated` of InferRequest to 1 in batched request
    // - return completed request to client
    for (auto it = _breq.begin(); it != _breq.end(); it++) {
        Ptr<InferRequest> request = *it;

        // iteration done -> update request stat in batch
        request->is_initiated = true;
        request->generated++;

        // clear child operations of Key/Value tensor
        for (int layer = 0; layer < _config.model_n_layer; ++layer) {
            request->K_cache[layer]->clear_child_nodes();
            request->V_cache[layer]->clear_child_nodes();
        }

        if (request->output_size == request->generated) {
            assert(request->is_initiated);
            // spdlog::info("Scheduler::return request_id: {}", request->id);
            _completed_request_queue.push(request);

            // when completed, free KV cache
            for (auto itr = _request_queue.begin(); itr != _request_queue.end();) {
                Ptr<InferRequest> cur = *itr;
                if (cur->id == request->id) {
                    itr = _request_queue.erase(itr);
                    _active_reqs--;
                    spdlog::info("Scheduler::request {} done!", request->id);
                } else {
                    itr++;
                }
            }
        }
    }

    _model_program = nullptr;
    _breq.clear();
}
//  if _model_program exist,
//      if _model_program is finished
//          update batched requests
//          moveout generated request
//      if _executable_tile_queue is empty && no active operation
//          get operation from _model_program and activate
void Scheduler::refresh_status() {
    if (_model_program != nullptr) {
        if (_model_program->check_finish()) {
            finish_program();
            // exit(0);
        }
    }
    // initiate operation
    // xxx is count_active_operations() == 0 necessary?
    if (_model_program != nullptr && _executable_tile_queue.empty() &&
        count_active_operations() == 0) {
        spdlog::info("executable operation count {}",
                     _model_program->get_executable_operations().size());
        auto op = _model_program->get_executable_operations().front();
        spdlog::info("Start operation {}", op->get_name());
        // fixed: is it okay to set produced when the operation starts?
        // move to Model::finish_operation
        // for (int output_id = 0; output_id < op->num_outputs(); output_id++)
        //     op->get_output(output_id)->set_produced();
        // spdlog::info("tile count {}", op->get_tiles().size());
        // todo: delete >>> gsheo
        // if (op->get_name().find("MatMul") != std::to_string::npos) {
        //     spdlog::info("MatMul tile count {} instruction size: {}",
        //                  op->get_tiles().size(),
        //                  op->get_tiles().front().instructions.size());
        //     exit(-1);
        // }
        assert(op->get_tiles().size());
        _executable_tile_queue = op->get_tiles();
        _active_operation_stats[op->get_id()] = RunningOperationStat{
            .id = op->get_id(),
            .name = op->get_name(),
            // xxx necessary?
            // .launched = true,
            .start_cycle = *_core_cycle,
            .total_tiles = (uint32_t)_executable_tile_queue.size(),
            .remain_tiles = (uint32_t)_executable_tile_queue.size(),
            .launched_tiles = 0,
        };
    }
}

uint32_t Scheduler::count_active_operations() { return _active_operation_stats.size(); }
