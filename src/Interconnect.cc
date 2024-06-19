#include "Interconnect.h"

#include <cmath>
#include <filesystem>

#include "booksim2/Interconnect.hpp"

namespace fs = std::filesystem;

void Interconnect::log(Stage stage) {
    std::string fname =
        Config::global_config.log_dir + "/memio_stage_" + stageToString(stage) + "_ch_";
    for (size_t i = 0; i < _stats.size(); ++i) {
        Logger::log(_stats[i], fname + std::to_string(i));
        auto last_stat = _stats[i].back();
        _stats[i].clear();
        _stats[i].push_back(last_stat);
    }
}

cycle_type Interconnect::get_core_cycle() {
    return (cycle_type)((double)_cycles * (double)Config::global_config.core_freq /
                        (double)Config::global_config.icnt_freq);
}

void Interconnect::update_stat(MemoryAccess memory_access, uint64_t ch_idx) {
    // READ, WRITE, GWRITE, COMP, READRES, P_HEADER, COMPS_READRES, SIZE
    switch (memory_access.req_type) {
        case MemoryAccessType::READ:
            _stats[ch_idx].back().memory_reads += memory_access.size;
            break;
        case MemoryAccessType::WRITE:
            _stats[ch_idx].back().memory_writes += memory_access.size;
            break;
        case MemoryAccessType::READRES:
        case MemoryAccessType::COMPS_READRES:
            _stats[ch_idx].back().pim_reads += memory_access.size;
            break;
            // default:
            //     ast(0);
    }
}

SimpleInterconnect::SimpleInterconnect(SimulationConfig config) : _latency(config.icnt_latency) {
    spdlog::info("Initialize SimpleInterconnect");
    _cycles = 0;
    _config = config;
    _n_nodes = config.num_cores * config.dram_channels + config.dram_channels;
    _dram_offset = config.num_cores * config.dram_channels;
    _in_buffers.resize(_n_nodes);
    _out_buffers.resize(config.num_cores * config.dram_channels);

    _mem_sa_q_turn = true;
    _mem_req_queue1.resize(config.dram_channels);  // for SA
    _mem_req_queue2.resize(config.dram_channels);  // for PIM

    _busy_node.resize(_n_nodes);
    for (int node = 0; node < _n_nodes; node++) {
        _busy_node[node] = false;
    }
    // TODO: make it configurable
    _mem_cycle_interval = 250;
    _stats.resize(config.dram_channels);
    for (size_t i = 0; i < config.dram_channels; ++i) {
        _stats[i].push_back(MemoryIOStat(0, i, _mem_cycle_interval));
    }
}

bool SimpleInterconnect::running() { return false; }

void SimpleInterconnect::cycle() {
    for (int node = 0; node < _n_nodes; node++) {
        // std::cout << "in buffer " << node << " : " << _in_buffers[node].size() << std::endl;
        // std::cout << "out buffer " << node << " : " << _out_buffers[node].size() << std::endl;
    }
    // in_bufs -> out_bufs
    // one of core -> dram or dram -> core is possible.
    for (int node = 0; node < _n_nodes; node++) {
        int src_node = (_rr_start + node) % _n_nodes;
        if (!_in_buffers[src_node].empty() &&
            _in_buffers[src_node].front().finish_cycle <= _cycles) {
            uint32_t dest = _in_buffers[src_node].front().dest;
            if (!_busy_node[dest]) {
                if (dest < _dram_offset) {
                    _out_buffers[dest].push(_in_buffers[src_node].front().access);
                } else {
                    uint32_t mem_ch = dest - _dram_offset;
                    MemoryAccess *mem_req = _in_buffers[src_node].front().access;
                    if (!_config.sub_batch_mode) {
                        // When single buffer PIM (Newton), there is single batch,
                        // so use one interconnect queue.
                        mem_req->stage_platform = StagePlatform::SA;
                    }
                    assert(mem_req->stage_platform == StagePlatform::SA ||
                           mem_req->stage_platform == StagePlatform::PIM);
                    if (mem_req->stage_platform == StagePlatform::SA)
                        _mem_req_queue1[mem_ch].push(mem_req);
                    else if (mem_req->stage_platform == StagePlatform::PIM)
                        _mem_req_queue2[mem_ch].push(mem_req);
                    else
                        exit(-1);
                }
                _in_buffers[src_node].pop();
                _busy_node[dest] = true;
                // spdlog::_log_filece("PUSH TO OUTBUFFER {} {}", src_node, dest);
            }
        }
    }

    for (auto ch_idx = 0; ch_idx < _config.dram_channels; ++ch_idx) {
        if (_stats[ch_idx].back().start_cycle + _mem_cycle_interval < get_core_cycle()) {
            auto stat = MemoryIOStat((get_core_cycle() / _mem_cycle_interval) * _mem_cycle_interval,
                                     ch_idx, _mem_cycle_interval);
            _stats[ch_idx].push_back(stat);
        }
    }

    for (int node = 0; node < _n_nodes; node++) {
        _busy_node[node] = false;
    }
    _rr_start = (_rr_start + 1) % _n_nodes;
    _cycles++;
}

void SimpleInterconnect::push(uint32_t src, uint32_t dest, MemoryAccess *request) {
    // -- initialize entity
    SimpleInterconnect::Entity entity;
    if (_in_buffers[src].empty())
        entity.finish_cycle = _cycles + _latency;
    else
        entity.finish_cycle = _in_buffers[src].back().finish_cycle + 1;
    entity.dest = dest;
    entity.access = request;

    // -- push to _in_buffer
    _in_buffers[src].push(entity);
}

bool SimpleInterconnect::is_full(uint32_t nid, MemoryAccess *request) {
    // TODO: limit buffersize
    return false;
}

bool SimpleInterconnect::is_empty(uint32_t nid) {
    assert(nid < _dram_offset);
    return _out_buffers[nid].empty();
}

MemoryAccess *SimpleInterconnect::top(uint32_t nid) {
    assert(!is_empty(nid));
    assert(nid < _dram_offset);
    return _out_buffers[nid].front();
}

void SimpleInterconnect::pop(uint32_t nid) {
    assert(nid < _dram_offset);
    auto mem_access = top(nid);

    // -- collect log
    if (nid < _dram_offset) update_stat(*mem_access, nid % _config.dram_channels);

    _out_buffers[nid].pop();
}
// below 3 method is used to send "Memory request" to "Dram" in "Interconnect"
// - has_memreq
// - memreq_top
// - memreq_pop

bool SimpleInterconnect::has_memreq1(uint32_t cid) { return !_mem_req_queue1[cid].empty(); }
bool SimpleInterconnect::has_memreq2(uint32_t cid) { return !_mem_req_queue2[cid].empty(); }

MemoryAccess *SimpleInterconnect::memreq_top1(uint32_t cid) {
    assert(has_memreq1(cid));
    return _mem_req_queue1[cid].front();
}
MemoryAccess *SimpleInterconnect::memreq_top2(uint32_t cid) {
    assert(has_memreq2(cid));
    return _mem_req_queue2[cid].front();
}

void SimpleInterconnect::memreq_pop1(uint32_t cid) {
    assert(has_memreq1(cid));
    _mem_req_queue1[cid].pop();
}

void SimpleInterconnect::memreq_pop2(uint32_t cid) {
    assert(has_memreq2(cid));
    _mem_req_queue2[cid].pop();
}
