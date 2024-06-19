#include "Scheduler.h"

#include <cmath>

#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle)
    : _config(config), _core_cycle(core_cycle), _cycles(0) {
    _max_batch_size = 1024;   // 256;   // config.max_batch_size;
    _max_active_reqs = 1024;  // 256;  // 70;
    _active_reqs = 0;
    _next_ch = 0;

    // Model dimension init
    _nh = _config.model_n_head / _config.n_tp;
    _dk = _config.model_n_embd / _config.model_n_head;
    _effective_e = _nh * _dk;

    // Memory spec init
    _dram_channels = _config.dram_channels;
    _dram_page_size = _config.dram_page_size / _config.precision;
    _dram_banks_per_ch = _config.dram_banks_per_ch;

    // 1: Systolic Array Program
    // 2: PIM Program
    _model_program1 = nullptr;
    _model_program2 = nullptr;

    _init_stage = Stage::A;
    // _init_stage = Stage::B;
    _stage = _init_stage;
    _just_one_stage = false;

    _has_stage_changed = false;

    // Request queue for channel

    for (int i = 0; i < _dram_channels; i++) {
        auto req_q = std::vector<Ptr<InferRequest>>();
        auto req_latency_q = std::vector<uint32_t>();
        req_q.reserve(_max_batch_size);
        req_latency_q.reserve(_max_batch_size);
        _active_request_queues.push_back(req_q);
        _active_request_latency_queues.push_back(req_latency_q);
    }

    // KV allocate by pim tile
    int model_weight = _config.model_params_b * _config.precision / _config.n_tp;  // GB
    int memory_capacity = _dram_channels;                                          // GB
    int available_for_kv = memory_capacity - model_weight;                         // GB
    int pim_tile_size = _config.dram_page_size * _dram_banks_per_ch;               // B
    _total_tiles = floor((double)available_for_kv GB / pim_tile_size);
    _total_available_tiles = _total_tiles;

    int tiles_per_channel = floor((double)_total_tiles / _dram_channels);
    _available_tiles.reserve(_dram_channels);
    for (int i = 0; i < _dram_channels; i++) {
        _available_tiles.push_back(tiles_per_channel);
    }

    spdlog::info("Total PIM tiles: {}", _total_tiles);
    spdlog::info("Tiles per channel: {}", tiles_per_channel);

    // how often to create a page per number of tokens.
    _key_period = _dram_banks_per_ch;
    _value_period = _dram_page_size;

    // how many PIM tiles compose a page.
    _key_page_size = ceil((double)_effective_e / _value_period);
    _value_page_size = ceil((double)_effective_e / _key_period);

    spdlog::info("_key_period: {}", _key_period);
    spdlog::info("_key_page_size: {}", _key_page_size);
    spdlog::info("_value_period: {}", _value_period);
    spdlog::info("_value_page_size: {}", _value_page_size);
    spdlog::info("Effective E(_nh * _dk):{}", _nh * _dk);

    // PIM GEMV latency
    _gwrite_latency = 100;
    _gemv_latency = 184;
}

void Scheduler::launch(Ptr<Model> model) {
    _model = model;
    spdlog::info("MODEL {} Launched in Scheduler", model->get_name());
}

/* Deprecated: allocate channel when making dataset */
// if return -1, it means there is no available tile for this request
int Scheduler::allocate_pim_tile(uint32_t seq_len) {
    // granularity of key
    // granularity of value

    int key_pages = ceil((double)seq_len / _key_period);
    int value_pages = ceil((double)seq_len / _value_period);

    int key_tiles = key_pages * _key_page_size;
    int value_tiles = value_pages * _value_page_size;
    int required_tiles_for_kv_cache = key_tiles + value_tiles;

    uint32_t ch;
    if (true || _config.baseline_exp) {
        // >> newton: round-robin channel allocate
        ch = _next_ch % _dram_channels;
        int available_tiles = _available_tiles[ch];
        if (available_tiles >= required_tiles_for_kv_cache) {
            _available_tiles[ch] -= required_tiles_for_kv_cache;
        } else {
            int trial = _dram_channels;
            while (trial--) {
                ch = _next_ch % _dram_channels;
                available_tiles = _available_tiles[ch];
                if (available_tiles >= required_tiles_for_kv_cache) {
                    _available_tiles[ch] -= required_tiles_for_kv_cache;
                    break;
                }
                ch = -1;
                _next_ch++;
            }
        }
        if (ch == -1) {
            spdlog::info("No available tiles for this request");
            return -1;
        }
        _next_ch++;
    } else {
        // >> neupims: channel load balancing
        // todo: implement this part
    }

    assert(ch != -1);
    // spdlog::info("seqlen: {}", seq_len);
    // spdlog::info("required Key pages: {}", key_pages);
    // spdlog::info("required Value pages: {}", value_pages);
    // spdlog::info("required KV tiles: {}", required_tiles_for_kv_cache);

    // _total_available_tiles -= required_tiles_for_kv_cache;
    // spdlog::info("Remain tiles: {}", _total_available_tiles);
    // spdlog::info("Remain tiles in ch#{}: {}", ch, _available_tiles[ch]);
    // spdlog::info("--------------------");
    return ch;
}

void Scheduler::setup_requests() {
    uint32_t batch_size = 0;
    for (auto it = _request_queue.begin(); it != _request_queue.end(); it++) {
        if (batch_size == _max_batch_size) break;
        Ptr<InferRequest> request = *it;
        assert(request->output_size > request->generated);

        if (!request->is_initiated) {
            int ch = request->channel;
            assert(ch < _dram_channels);
            spdlog::info("request#{} seq_len:{} channel:{}", request->id, request->input_size,
                         request->channel);
            // allocate_pim_tile(request->input_size);
            if (ch == -1) continue;

            uint32_t seq_len = request->input_size;

            std::vector<uint32_t> dim_key{_nh, _dk, seq_len};
            std::vector<uint32_t> dim_value{_nh, seq_len, _dk};

            if (_active_reqs >= _max_active_reqs) continue;
            _active_reqs++;
            // spdlog::info("Scheduler allocate request#{}(seq_len:{}) to channel {}<<",
            //              request->id, seq_len, ch);
            auto k = std::make_shared<PIMTensor>(
                name_gen(std::to_string(request->id), "KEY", std::to_string(0)), ch, dim_key,
                PIMTensorKVType::KEY, true);
            auto v = std::make_shared<PIMTensor>(
                name_gen(std::to_string(request->id), "VALUE", std::to_string(0)), ch, dim_value,
                PIMTensorKVType::VALUE, true);
            request->K_cache.push_back(k);
            request->V_cache.push_back(v);

            _active_request_queues[ch].push_back(request);
            uint32_t mha_latency = estimate_mha_latency(request);
            _active_request_latency_queues[ch].push_back(mha_latency);

            request->is_initiated = true;
        }

        batch_size++;
    }

    spdlog::info("---------");
    int min_latency = 9000000;
    int max_latency = 0;
    for (int ch = 0; ch < _dram_channels; ch++) {
        int channel_total_latency = 0;
        for (auto latency : _active_request_latency_queues[ch]) {
            channel_total_latency += latency;
        }
        spdlog::info("channel #{} remain_tiles: {}, total MHA latency: {}", ch,
                     _available_tiles[ch], channel_total_latency);
        min_latency = MIN(min_latency, channel_total_latency);
        max_latency = MAX(max_latency, channel_total_latency);
    }
    spdlog::info("---------");
    spdlog::info("MIN: {}, MAX: {}, difference: {}", min_latency, max_latency,
                 max_latency - min_latency);

    // exit(-1);
}

void Scheduler::make_program() {
    std::shared_ptr<BatchedRequest> sub_batch_on_sa;  // = std::make_shared<BatchedRequest>(_breq1);
    std::shared_ptr<BatchedRequest> sub_batch_on_pim;  //= std::make_shared<BatchedRequest>(_breq2);
    if (static_cast<int>(_stage) % 2 == 0) {
        sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq1);
        sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq2);
    } else {
        sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq2);
        sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq1);
    }

    spdlog::info("New Program for SA  (sub-batch.size: {})", sub_batch_on_sa->_reqs.size());
    spdlog::info("New Program for PIM (sub-batch.size: {})", sub_batch_on_pim->_reqs.size());

    _model_program1 =
        std::make_unique<StageProgram>(_model, sub_batch_on_sa, StagePlatform::SA, _stage);
    _model_program2 =
        std::make_unique<StageProgram>(_model, sub_batch_on_pim, StagePlatform::PIM, _stage);

    refresh_status1();
    refresh_status2();
}

int Scheduler::estimate_mha_latency(Ptr<InferRequest> request) {
    // todo: calculate latency with sequence length
    int latency = 0;
    int seq_len = request->input_size;

    // key * query
    int chunks = ceil((double)_effective_e / _dram_page_size);
    int tiles = ceil((double)seq_len / _dram_banks_per_ch);
    latency += chunks * _gwrite_latency;
    latency += chunks * tiles * _gemv_latency;

    // logit * value
    chunks = ceil((double)seq_len / _dram_page_size) * _nh;
    tiles = ceil((double)_dk / _dram_banks_per_ch);
    latency += chunks * _gwrite_latency;
    latency += chunks * tiles * _gemv_latency;

    return latency;
}

void Scheduler::group_sub_batches() {
    if (_config.baseline_exp) {
        //>>>
        // Consolidate to one batch
        for (int ch = 0; ch < _dram_channels; ch++) {
            auto req_queue = _active_request_queues[ch];
            for (auto it = req_queue.begin(); it != req_queue.end(); it++) {
                Ptr<InferRequest> request = *it;
                _breq1.push_back(request);
            }
        }
        return;
        //<<<
    }

    for (int ch = 0; ch < _dram_channels; ch++) {
        auto req_queue = _active_request_queues[ch];
        auto latency_queue = _active_request_latency_queues[ch];
        assert(req_queue.size() == latency_queue.size());
        auto index_lists = partition_lists(latency_queue);
        std::vector<int> list1 = index_lists.first;
        std::vector<int> list2 = index_lists.second;

        int sum_list1_latencies = 0;
        int sum_list2_latencies = 0;
        std::string list1_str = "";
        std::string list2_str = "";
        std::string time1_str = "";
        std::string time2_str = "";
        for (auto it = list1.begin(); it != list1.end(); it++) {
            int req_id = *it;
            Ptr<InferRequest> request = req_queue[req_id];
            sum_list1_latencies += latency_queue[req_id];
            _breq1.push_back(request);
            list1_str += std::to_string(req_id) + ", ";
            time1_str += std::to_string(latency_queue[req_id]) + ", ";
        }
        for (auto it = list2.begin(); it != list2.end(); it++) {
            int req_id = *it;
            Ptr<InferRequest> request = req_queue[req_id];
            sum_list2_latencies += latency_queue[req_id];
            _breq2.push_back(request);
            list2_str += std::to_string(req_id) + ", ";
            time2_str += std::to_string(latency_queue[req_id]) + ", ";
        }
        // spdlog::info("====Channel {}====", ch);
        // spdlog::info("#1 sum:{:2d}, idx:[{}], time:[{}]", sum_list1_latencies, list1_str,
        //              time1_str);
        // spdlog::info("#2 sum:{:2d}, idx:[{}], time:[{}]", sum_list2_latencies, list2_str,
        //              time2_str);
        // spdlog::info("req_q.size:{}, latency_q.size:{}", req_queue.size(), latency_queue.size());
    }

    spdlog::info("total batch_size: {}", _breq1.size() + _breq2.size());
}

// Called exactly once
void Scheduler::init_batches() {
    setup_requests();
    group_sub_batches();
}

void Scheduler::cycle() {
    bool step_next_stage = _model_program1 == nullptr && _model_program2 == nullptr;

    if (step_next_stage && _stage == _init_stage && !_request_queue.empty()) {
        init_batches();
        // exit(-1);
    }

    _cycles++;

    bool lets_make_program1 = _model_program1 == nullptr && _breq1.size() > 0;
    bool lets_make_program2 = _model_program2 == nullptr && _breq2.size() > 0;
    // spdlog::info("lets_make_program1: {}", lets_make_program1);
    // spdlog::info("lets_make_program2: {}", lets_make_program2);

    if (lets_make_program1 && lets_make_program2) {
        if (_stage == Stage::Finish) {
            cleanup_sub_batch(_breq1);
            cleanup_sub_batch(_breq2);
            _breq1.clear();
            _breq2.clear();
            return;
        } else {
            std::string red = "\033[1;31m";
            std::string reset = "\033[0m";
            spdlog::info("{}----------Stage {}----------{}", red, stageToString(_stage), reset);
            make_program();
        }
    }
    if (_config.baseline_exp) {
        // >> newton
        bool both_program_none = _model_program1 == nullptr && _model_program2 == nullptr;
        bool exist_request = _breq2.size() > 0 || _breq1.size() > 0;
        if (both_program_none && exist_request) {
            if (_stage == Stage::Finish) {
                cleanup_sub_batch(_breq1);
                cleanup_sub_batch(_breq2);
                _breq1.clear();
                _breq2.clear();
                return;
            } else {
                std::string red = "\033[1;31m";
                std::string reset = "\033[0m";
                spdlog::info("{}----------Stage {}----------{}", red, stageToString(_stage), reset);
                make_program();
            }
        }
        // <<
    }
}

void Scheduler::add_request(std::shared_ptr<InferRequest> request) {
    _request_queue.push_back(request);
}

bool Scheduler::has_completed_request() { return !_completed_request_queue.empty(); }

std::shared_ptr<InferRequest> Scheduler::pop_completed_request() {
    // spdlog::info("Scheduler::pop_completed_request()");
    auto completed_req = _completed_request_queue.front();
    _completed_request_queue.pop();
    return completed_req;
}

Tile& Scheduler::top_tile1(uint32_t core_id) {
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue1.empty()) {
        return empty_tile;
    } else {
        Tile& tile = _executable_tile_queue1.front();
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            tile.stage_platform = StagePlatform::SA;
            return tile;
        }
    }
}

Tile& Scheduler::top_tile2(uint32_t core_id) {
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue2.empty()) {
        return empty_tile;
    } else {
        Tile& tile = _executable_tile_queue2.front();
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            tile.stage_platform = StagePlatform::PIM;
            return tile;
        }
    }
}

// ??: Add base address for each addr in tiles / XXX: < necessary comment?
// ??: something wrong with functionality. seems it's not a necessary function
void Scheduler::get_tile1(uint32_t core_id) {
    if (_executable_tile_queue1.empty()) {
        return;
    } else {
        Tile& tile = _executable_tile_queue1.front();
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue1.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        } else {
            _active_operation_stats[tile.operation_id].launched_tiles++;
            _executable_tile_queue1.pop_front();
            spdlog::debug("Operation {} Core {} Get Tile at {}", tile.optype, core_id,
                          *_core_cycle);
            return;
        }
    }
}
void Scheduler::get_tile2(uint32_t core_id) {
    if (_executable_tile_queue2.empty()) {
        return;
    } else {
        Tile& tile = _executable_tile_queue2.front();
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue2.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        } else {
            _active_operation_stats[tile.operation_id].launched_tiles++;
            _executable_tile_queue2.pop_front();
            spdlog::debug("Operation {} Core {} Get Tile at {}", tile.optype, core_id,
                          *_core_cycle);
            return;
        }
    }
}

//  update operation stat
//  if operation is finished
//      apply to _model_program & return true
bool Scheduler::finish_tile(uint32_t core_id, Tile& tile) {
    bool result = false;
    spdlog::debug("Tile {} Core {} Finish Tile at {}", tile.operation_id, core_id, *_core_cycle);
    assert(_active_operation_stats.find(tile.operation_id) != _active_operation_stats.end());
    assert(_finished_operation_stats.find(tile.operation_id) == _finished_operation_stats.end());
    assert(_active_operation_stats[tile.operation_id].remain_tiles > 0);
    _active_operation_stats[tile.operation_id].remain_tiles--;

    spdlog::info("Finish tile stage_platform:{}", stagePlatformToString(tile.stage_platform));

    if (tile.stage_platform == StagePlatform::SA)
        _model_program1->finish_operation_tile(tile);
    else
        _model_program2->finish_operation_tile(tile);

    if (_active_operation_stats[tile.operation_id].remain_tiles == 0) {
        result = true;
        spdlog::info("Layer {} finish at {}", _active_operation_stats[tile.operation_id].name,
                     *_core_cycle);
        spdlog::info("Total compute time {}",
                     *_core_cycle - _active_operation_stats[tile.operation_id].start_cycle);

        if (tile.stage_platform == StagePlatform::SA)
            _model_program1->finish_operation(tile.operation_id);
        else
            _model_program2->finish_operation(tile.operation_id);

        _finished_operation_stats[tile.operation_id] = _active_operation_stats[tile.operation_id];
        _active_operation_stats.erase(tile.operation_id);
    }

    if (tile.stage_platform == StagePlatform::SA)
        refresh_status1();
    else
        refresh_status2();

    return result;
}

bool Scheduler::empty1() { return _model_program1 == nullptr; }
bool Scheduler::empty2() { return _model_program2 == nullptr; }

bool Scheduler::running() { return !_request_queue.empty() || !_completed_request_queue.empty(); }

void Scheduler::cleanup_sub_batch(std::vector<Ptr<InferRequest>> sub_batch) {
    // < todos when the model program has finished >
    // - increment `generated` of InferRequest to 1 in batched request
    // - return completed request to client
    for (auto it = sub_batch.begin(); it != sub_batch.end(); it++) {
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
                    // spdlog::info("Scheduler::request {} done!", request->id);
                } else {
                    itr++;
                }
            }
        }
    }
}

void Scheduler::refresh_stage() {
    bool stage_done = _model_program1 == nullptr && _model_program2 == nullptr;
    if (stage_done) {
        std::string red = "\033[1;31m";
        std::string reset = "\033[0m";
        std::string stage_name = stageToString(_stage);
        spdlog::info("{}------- Stage {} Done -------{}", red, stage_name, reset);

        // Update stat
        _stage_stats.push_back(std::make_pair(stage_name, _cycles));

        _prev_stage = _stage;

        // Update stage
        int stageValue = static_cast<int>(_stage);
        stageValue++;
        _stage = static_cast<Stage>(stageValue);

        _has_stage_changed = true;

        if (_config.baseline_exp) {
            // >> newton
            if (_stage == Stage::C) _stage = Stage::E;
            if (_stage == Stage::F) _stage = Stage::Finish;
            // << newton
        }
        if (_just_one_stage) _stage = Stage::Finish;  // force to execute just one stage
    }
}

void Scheduler::finish_program1() {
    spdlog::info("Model finish at {}", *_core_cycle);
    _model_program1->log();

    _model_program1 = nullptr;
    refresh_stage();
    // todo: stat for sa_program

    // cleanup_sub_batch(_breq1);
    // _breq1.clear();
}

void Scheduler::finish_program2() {
    spdlog::info("Model finish at {}", *_core_cycle);
    _model_program2->log();

    _model_program2 = nullptr;
    refresh_stage();
    // todo: stat for pim_program

    // cleanup_sub_batch(_breq2);
    // _breq2.clear();
}

void Scheduler::refresh_status1() {
    if (_model_program1 != nullptr) {
        if (_model_program1->check_finish()) {
            finish_program1();
            // exit(0);
        }
    }
    // initiate operation
    // xxx is count_active_operations() == 0 necessary?
    if (_model_program1 != nullptr && _executable_tile_queue1.empty()) {
        // spdlog::info("executable operation count {}",
        //              _model_program1->get_executable_operations().size());
        auto op = _model_program1->get_executable_operations().front();
        spdlog::info("Start operation {}", op->get_name());
        if (count_active_operations()) {
            // for (auto& op_stat : _active_operation_stats) {
            //     spdlog::info("op stat currently in is {}", op_stat.second.name);
            // }
            if (_active_operation_stats.find(op->get_id()) != _active_operation_stats.end()) {
                return;
            }
        }

        assert(op->get_tiles().size());
        _executable_tile_queue1 = op->get_tiles();
        _active_operation_stats[op->get_id()] = RunningOperationStat{
            .id = op->get_id(),
            .name = op->get_name(),
            // xxx necessary?
            // .launched = true,
            .start_cycle = *_core_cycle,
            .total_tiles = (uint32_t)_executable_tile_queue1.size(),
            .remain_tiles = (uint32_t)_executable_tile_queue1.size(),
            .launched_tiles = 0,
        };
    } else {
        // spdlog::info("is model null {} / is executable tile queue empty {} / count active ops
        // {}",
        //              _model_program1 == nullptr, _executable_tile_queue1.empty(),
        //              count_active_operations());
        // for (auto& op_stat : _active_operation_stats) {
        //     spdlog::info("op stat currently in is {}", op_stat.second.name);
        // }
    }
}

void Scheduler::refresh_status2() {
    if (_model_program2 != nullptr) {
        if (_model_program2->check_finish()) {
            finish_program2();
            // exit(0);
        }
    }
    // initiate operation
    // xxx is count_active_operations() == 0 necessary?
    if (_model_program2 != nullptr && _executable_tile_queue2.empty()
        //  && count_active_operations() == 0) {
    ) {
        // spdlog::info("executable operation count {}",
        //              _model_program2->get_executable_operations().size());
        auto op = _model_program2->get_executable_operations().front();
        spdlog::info("Start operation {}", op->get_name());
        if (count_active_operations()) {
            if (_active_operation_stats.find(op->get_id()) != _active_operation_stats.end()) {
                return;
            }
        }

        assert(op->get_tiles().size());
        _executable_tile_queue2 = op->get_tiles();
        _active_operation_stats[op->get_id()] = RunningOperationStat{
            .id = op->get_id(),
            .name = op->get_name(),
            // xxx necessary?
            // .launched = true,
            .start_cycle = *_core_cycle,
            .total_tiles = (uint32_t)_executable_tile_queue2.size(),
            .remain_tiles = (uint32_t)_executable_tile_queue2.size(),
            .launched_tiles = 0,
        };
    }
}

uint32_t Scheduler::count_active_operations() { return _active_operation_stats.size(); }

std::pair<std::vector<int>, std::vector<int>> Scheduler::partition_lists(
    std::vector<uint32_t> inputList) {
    int totalSum = 0;
    for (int num : inputList) {
        totalSum += num;
    }

    int n = inputList.size();
    int targetSum = totalSum / 2;

    // Initialize a matrix to store intermediate results
    std::vector<std::vector<bool>> dp(n + 1, std::vector<bool>(targetSum + 1, false));

    // Base case: an empty subset can always achieve a sum of 0
    for (int i = 0; i <= n; ++i) {
        dp[i][0] = true;
    }

    // Fill the matrix using dynamic programming
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= targetSum; ++j) {
            dp[i][j] = dp[i - 1][j];
            if (j >= inputList[i - 1]) {
                dp[i][j] = dp[i][j] || dp[i - 1][j - inputList[i - 1]];
            }
        }
    }

    // Find the maximum sum that can be achieved
    int maxSum = 0;
    for (int j = targetSum; j >= 0; --j) {
        if (dp[n][j]) {
            maxSum = j;
            break;
        }
    }

    // Reconstruct the two lists
    std::vector<int> list1, list2;
    int i = n, j = maxSum;
    while (i > 0 && j > 0) {
        if (dp[i][j] && !dp[i - 1][j]) {
            list1.push_back(i - 1);
            j -= inputList[i - 1];
        } else {
            list2.push_back(i - 1);
        }
        --i;
    }

    // If there are remaining elements, add them to list1
    while (i > 0) {
        list2.push_back(i - 1);
        --i;
    }

    return std::make_pair(list1, list2);
}

void Scheduler::print_stat() {
    int prev_cycles = 0;
    for (auto stage_stat : _stage_stats) {
        auto stage_name = stage_stat.first;
        auto stage_cycles = stage_stat.second;
        auto exec_cycles = stage_cycles - prev_cycles;

        spdlog::info("Stage {} : {} cycles", stage_name, exec_cycles);

        prev_cycles = stage_cycles;
    }
}