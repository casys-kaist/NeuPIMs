#include "Simulator.h"

#include <filesystem>
#include <string>

#include "NeuPIMSystolicWS.h"
#include "SystolicOS.h"
#include "SystolicWS.h"
#include "scheduler/NeuPIMScheduler.h"
#include "scheduler/OrcaScheduler.h"

namespace fs = std::filesystem;

Simulator::Simulator(SimulationConfig config) : _config(config), _core_cycles(0) {
    // Create dram object
    _core_period = 1.0 / ((double)config.core_freq);
    _icnt_period = 1.0 / ((double)config.icnt_freq);
    _dram_period = 1.0 / ((double)config.dram_freq);
    _core_time = 0.0;
    _dram_time = 0.0;
    _icnt_time = 0.0;

    std::string pim_config =
        fs::path(__FILE__).parent_path().append(config.pim_config_path).string();
    spdlog::info("Newton config: {}", pim_config);
    config.pim_config_path = pim_config;
    _dram = std::make_unique<PIM>(config);

    // Create interconnect object
    _icnt = std::make_unique<SimpleInterconnect>(config);
    // if (config.icnt_type == IcntType::SIMPLE) {
    // } else if (config.icnt_type == IcntType::BOOKSIM2) {
    //     _icnt = std::make_unique<Booksim2Interconnect>(config);
    // } else {
    //     assert(0);
    // }

    // Create core objects
    _cores.resize(config.num_cores);
    _n_cores = config.num_cores;
    _n_memories = config.dram_channels;
    for (int core_index = 0; core_index < _n_cores; core_index++) {
        assert(config.sub_batch_mode);
        spdlog::info("initializing NeuPIM SystolicWS cores.");
        _cores[core_index] = std::make_unique<NeuPIMSystolicWS>(core_index, _config);
        // if (config.sub_batch_mode) {
        // } else if (config.core_type == CoreType::SYSTOLIC_OS) {
        //     spdlog::info("initializing SystolicOS cores.");
        //     _cores[core_index] = std::make_unique<SystolicOS>(core_index, _config);
        // } else if (config.core_type == CoreType::SYSTOLIC_WS) {
        // spdlog::info("initializing SystolicWS cores.");
        // _cores[core_index] = std::make_unique<SystolicWS>(core_index, _config);
        // }
    }

    if (config.scheduler_type == "simple") {
        _scheduler = std::make_unique<OrcaScheduler>(_config, &_core_cycles);
    } else if (config.scheduler_type == "neupims") {
        _scheduler = std::make_unique<NeuPIMScheduler>(_config, &_core_cycles);
    }

    // } else if (config.scheduler_type == "time_multiplex") {
    //     _scheduler = std::make_unique<TimeMultiplexScheduler>(_config, &_core_cycles);
    // } else if (config.scheduler_type == "spatial_split") {
    //     _scheduler = std::make_unique<HalfSplitScheduler>(_config, &_core_cycles);
    // }

    _client = std::make_unique<Client>(_config);
}

void Simulator::update_stage_stat() {
    Stage done_stage = _scheduler->get_prev_stage();
    _dram->log(done_stage);

    _stage_stats.push_back(StageStat{.stage = done_stage,
                                     .done_cycle = _core_cycles,
                                     .pim_cycles = _dram->get_avg_pim_cycle(),
                                     .npu_cycles = 0,
                                     .mem_bw_util = _dram->get_avg_bw_util()});
}

void Simulator::log_stage_stat() {
    std::string fname = Config::global_config.log_dir + "/_summary.tsv";
    std::ofstream ofile(fname);
    if (!ofile.is_open()) {
        assert(0);
    }

    std::string header = "";
    header += "Stage\t";
    header += "total_cycles\t";
    header += "pim_cycles\t";
    header += "mem_bw_util\t";
    ofile << header + "\n";

    int prev_cycle = 0;

    for (int i = 0; i < _stage_stats.size(); i++) {
        StageStat stage_stat = _stage_stats[i];
        std::string stage_row = "";

        int total_cycle = stage_stat.done_cycle - prev_cycle;
        prev_cycle = stage_stat.done_cycle;
        stage_row += stageToString(stage_stat.stage) + "\t";
        stage_row += std::to_string(total_cycle) + "\t";
        stage_row += std::to_string(stage_stat.pim_cycles) + "\t";
        stage_row += std::to_string(stage_stat.mem_bw_util) + "\t";

        ofile << stage_row + "\n";
    }

    ofile.close();
}

void Simulator::run(std::string model_name) {
    spdlog::info("======Start Simulation=====");
    _scheduler->launch(_model);
    spdlog::info("assign model {}", model_name);
    cycle();
}

void Simulator::cycle() {
    OpStat op_stat;
    ModelStat model_stat;
    uint32_t tile_count;
    while (running()) {
        int model_id = 0;

        set_cycle_mask();
        // Core Cycle
        if (_cycle_mask & CORE_MASK) {
            while (_client->has_request()) {  // FIXME: change while to if
                std::shared_ptr<InferRequest> infer_request = _client->pop_request();
                _scheduler->add_request(infer_request);
            }
            _client->cycle();

            while (_scheduler->has_completed_request()) {
                std::shared_ptr<InferRequest> response = _scheduler->pop_completed_request();
                _client->receive_response(response);
            }

            if (_scheduler->has_stage_changed()) {
                _scheduler->reset_has_stage_changed_status();
                // _icnt->log(_scheduler->get_prev_stage());
                update_stage_stat();
            }
            _scheduler->cycle();

            for (int core_id = 0; core_id < _n_cores; core_id++) {
                auto finished_tile = _cores[core_id]->pop_finished_tile();
                if (finished_tile == nullptr) {
                } else if (finished_tile->status == Tile::Status::FINISH) {
                    _scheduler->finish_tile(core_id, *finished_tile);
                }

                // Issue new tile to core
                if (_scheduler->empty1() && _scheduler->empty2()) continue;

                // >>> todo: support 2 sub-batch
                if (!_scheduler->empty1()) {
                    Tile &tile = _scheduler->top_tile1(core_id);
                    if ((tile.status != Tile::Status::EMPTY) && _cores[core_id]->can_issue(tile)) {
                        if (tile.status == Tile::Status::INITIALIZED) {
                            assert(tile.stage_platform == StagePlatform::SA);
                            _cores[core_id]->issue(tile);
                            _scheduler->get_tile1(core_id);
                        }
                    }
                }
                if (!_scheduler->empty2()) {
                    Tile &tile = _scheduler->top_tile2(core_id);
                    if ((tile.status != Tile::Status::EMPTY) && _cores[core_id]->can_issue_pim()) {
                        if (tile.status == Tile::Status::INITIALIZED) {
                            assert(tile.stage_platform == StagePlatform::PIM);
                            _cores[core_id]->issue_pim(tile);
                            _scheduler->get_tile2(core_id);
                        }
                    }
                }
                // <<< todo: support 2 sub-batch
                _cores[core_id]->cycle();
            }
            _core_cycles++;
        }

        // DRAM cycle
        if (_cycle_mask & DRAM_MASK) {
            _dram->cycle();
        }
        // Interconnect cycle
        if (_cycle_mask & ICNT_MASK) {
            for (int core_id = 0; core_id < _n_cores; core_id++) {
                for (uint32_t channel_index = 0; channel_index < _n_memories; ++channel_index) {
                    // core -> ICNT
                    auto core_ind = core_id * _n_cores + channel_index;
                    if (_cores[core_id]->has_memory_request(channel_index)) {
                        MemoryAccess *front = _cores[core_id]->top_memory_request(channel_index);
                        front->core_id = core_id;
                        if (!_icnt->is_full(core_ind, front)) {
                            _icnt->push(core_ind, get_dest_node(front), front);
                            _cores[core_id]->pop_memory_request(channel_index);
                        }
                    }
                    // ICNT -> core
                    if (!_icnt->is_empty(core_ind)) {
                        _cores[core_id]->push_memory_response(_icnt->top(core_ind));
                        _icnt->pop(core_ind);
                    }
                }
            }

            for (int dram_ind = 0; dram_ind < _n_memories; dram_ind++) {
                auto mem_ind = _n_cores * _n_memories + dram_ind;

                // ICNT to memory (log write)
                // (Sub-batch#1 -> DRAM)
                if (_icnt->has_memreq1(dram_ind)) {
                    auto memreq_sa = _icnt->memreq_top1(dram_ind);
                    if (!_dram->is_full(dram_ind, memreq_sa)) {
                        _dram->push(dram_ind, memreq_sa);
                        _icnt->memreq_pop1(dram_ind);
                    }
                }
                // (Sub-batch#2 -> DRAM) // only for NeuPIMs
                if (!_config.baseline_exp && _icnt->has_memreq2(dram_ind)) {
                    auto memreq_pim = _icnt->memreq_top2(dram_ind);
                    if (!_dram->is_full(dram_ind, memreq_pim)) {
                        _dram->push(dram_ind, memreq_pim);
                        _icnt->memreq_pop2(dram_ind);
                    }
                }

                // Pop response to ICNT from dram (log read)
                if (!_dram->is_empty(dram_ind) && !_icnt->is_full(mem_ind, _dram->top(dram_ind))) {
                    _icnt->push(mem_ind, get_dest_node(_dram->top(dram_ind)), _dram->top(dram_ind));
                    _dram->pop(dram_ind);
                }
            }

            _icnt->cycle();
        }
    }
    spdlog::info("Simulation Finished");
    /* Print simulation stats */
    for (int core_id = 0; core_id < _n_cores; core_id++) {
        _cores[core_id]->print_stats();
    }
    // _icnt->print_stats();
    // _icnt->log();
    _dram->print_stat();
    _scheduler->print_stat();
    log_stage_stat();
}

void Simulator::launch_model(Ptr<Model> model) { _model = model; }

bool Simulator::running() {
    bool running = false;

    for (auto &core : _cores) {
        running = running || core->running();
    }
    running = running || _icnt->running();
    running = running || _dram->running();
    running = running || _scheduler->running();
    running = running || _client->running();
    // if (!_client->running() && _cores[0]->running()) {
    //     // for debug
    //     spdlog::info("core[1] running: {}", _cores[0]->running());
    //     spdlog::info("icnt running: {}", _icnt->running());
    //     spdlog::info("dram running: {}", _dram->running());
    //     spdlog::info("scheduler running: {}", _scheduler->running());
    //     spdlog::info("client running: {}", _client->running());
    //     exit(-1);
    // }

    return running;
}

void Simulator::set_cycle_mask() {
    _cycle_mask = 0x0;
    double minimum_time = MIN3(_core_time, _dram_time, _icnt_time);
    if (_core_time <= minimum_time) {
        _cycle_mask |= CORE_MASK;
        _core_time += _core_period;
    }
    if (_dram_time <= minimum_time) {
        _cycle_mask |= DRAM_MASK;
        _dram_time += _dram_period;
    }
    if (_icnt_time <= minimum_time) {
        _cycle_mask |= ICNT_MASK;
        _icnt_time += _icnt_period;
    }
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
    if (access->request) {
        // MemoryAccess not issued
        return _n_cores * _n_memories + _dram->get_channel_id(access);
    } else {
        // MemoryAccess after it is pushed into dram
        return access->core_id * _n_memories + _dram->get_channel_id(access);
    }
}