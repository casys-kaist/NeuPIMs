#include "Dram.h"

#include "helper/HelperFunctions.h"

// >>> gsheo

PIM::PIM(SimulationConfig config)
    : _mem(std::make_unique<dramsim3::NewtonSim>(config.pim_config_path, config.log_dir)) {
    _total_processed_requests.resize(config.dram_channels);
    _processed_requests.resize(config.dram_channels);

    for (int ch = 0; ch < config.dram_channels; ch++) {
        _total_processed_requests[ch] = 0;
        _processed_requests[ch] = 0;
    }
    _config = config;
    _cycles = 0;
    _burst_cycle = _mem->GetBurstLength() / 2;  // double data rate

    spdlog::info("Newton init");
}

bool PIM::running() { return false; }

void PIM::cycle() {
    _mem->ClockTick();
    _cycles++;
    int interval = 10000;
    if (_cycles % interval == 0) {
        for (int ch = 0; ch < _config.dram_channels; ch++) {
            float util = ((float)_processed_requests[ch] * _burst_cycle) / interval * 100;
            spdlog::debug("DRAM CH[{}]: BW Util {:.2f}%", ch, util);
            _total_processed_requests[ch] += _processed_requests[ch];
            _processed_requests[ch] = 0;
        }
    }
}

uint64_t PIM::make_address(int channel, int rank, int bankgroup, int bank, int row, int col) {
    return _mem->MakeAddress(channel, rank, bankgroup, bank, row, col);
}
uint64_t PIM::encode_pim_header(int channel, int row, bool for_gwrite, int num_comps,
                                int num_readres) {
    return _mem->EncodePIMHeader(channel, row, for_gwrite, num_comps, num_readres);
}

bool PIM::is_full(uint32_t cid, MemoryAccess *request) {
    bool full = !_mem->WillAcceptTransaction(request->dram_address, int(request->req_type));
    // if (cid == 0) {
    // if (full)
    //     spdlog::info("MEMORY channel {} is full!! {}", cid,
    //     memAccessTypeString(request->req_type));
    // else
    //     spdlog::info("MEMORY channel {} can receive mem_req {}", cid,
    //                  memAccessTypeString(request->req_type));
    // }

    return full;
}

void PIM::push(uint32_t cid, MemoryAccess *request) {
    // std::string acc_type_str = memAccessTypeString(request->req_type);
    // if (cid == 0) spdlog::info("{} cid:{}", acc_type_str, cid);
    uint32_t mem_ch = get_channel_id(request);
    assert(mem_ch == cid);

    const addr_type atomic_bytes = _mem->GetBurstLength() * _mem->GetBusBits() / 8;
    // const addr_type atomic_bytes =
    //     _mem->GetBurstLength() * _mem->GetBusBits() / 8;
    const addr_type target_addr = request->dram_address;
    // align address
    const addr_type start_addr = target_addr - (target_addr % atomic_bytes);

    assert(start_addr == target_addr);
    assert(request->size == atomic_bytes);

    int count = 0;
    request->request = false;

    _mem_req_cnt++;
    _mem->AddTransaction(target_addr, int(request->req_type), request);
}

bool PIM::is_empty(uint32_t cid) {
    // spdlog::info("pim is_empty(" + std::to_string(cid) +
    //              "):" + std::to_string(_mem->IsEmpty(cid)));
    return _mem->IsEmpty(cid);
}

MemoryAccess *PIM::top(uint32_t cid) {
    assert(!is_empty(cid));
    return (MemoryAccess *)_mem->Top(cid);
}

void PIM::pop(uint32_t cid) {
    // spdlog::info("pim pop()");
    assert(!is_empty(cid));
    _mem->Pop(cid);
    _processed_requests[cid]++;
}

uint32_t PIM::get_channel_id(MemoryAccess *access) {
    // spdlog::info("pim get_channel_id()");
    return _mem->GetChannel(access->dram_address);
}

void PIM::print_stat() {
    // spdlog::info("pim print_stat()");
    uint32_t total_reqs = 0;
    for (int ch = 0; ch < _config.dram_channels; ch++) {
        float util = ((float)_total_processed_requests[ch] * _burst_cycle) / _cycles * 100;
        spdlog::info("DRAM CH[{}]: AVG BW Util {:.2f}%", ch, util);
        total_reqs += _total_processed_requests[ch];
    }
    float util = ((float)total_reqs * _burst_cycle / _config.dram_channels) / _cycles * 100;
    spdlog::info("DRAM: AVG BW Util {:.2f}%", util);
    spdlog::info("DRAM total cycles: {}", _cycles);
    spdlog::info("DRAM total processed memory requests: {}", _mem_req_cnt);
    _mem->PrintStats();
}

// <<< gsheo
