#ifndef DRAM_H
#define DRAM_H

#include <queue>
#include <utility>

#include "Common.h"
#include "Logger.h"
#include "Stat.h"
#include "newtonsim/NewtonSim.h"

class Dram {
   public:
    virtual bool running() = 0;
    virtual void cycle() = 0;
    virtual bool is_full(uint32_t cid, MemoryAccess *request) = 0;
    virtual void push(uint32_t cid, MemoryAccess *request) = 0;
    virtual bool is_empty(uint32_t cid) = 0;
    virtual MemoryAccess *top(uint32_t cid) = 0;
    virtual void pop(uint32_t cid) = 0;
    virtual uint32_t get_channel_id(MemoryAccess *request) = 0;
    virtual void print_stat() {}
    addr_type get_addr_align() { return _addr_align; }

    virtual double get_avg_bw_util() = 0;
    virtual uint64_t get_avg_pim_cycle() = 0;
    virtual void reset_pim_cycle() = 0;
    virtual void log(Stage stage) = 0;

   protected:
    SimulationConfig _config;
    uint32_t _n_ch;
    cycle_type _cycles;
    addr_type _addr_align;
};

class PIM : public Dram {
   public:
    PIM(SimulationConfig config);

    virtual bool running() override;
    virtual void cycle() override;
    virtual bool is_full(uint32_t cid, MemoryAccess *request) override;
    virtual void push(uint32_t cid, MemoryAccess *request) override;
    virtual bool is_empty(uint32_t cid) override;
    virtual MemoryAccess *top(uint32_t cid) override;
    virtual void pop(uint32_t cid) override;
    virtual uint32_t get_channel_id(MemoryAccess *request) override;
    virtual void print_stat() override;

    uint64_t MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col);
    uint64_t EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps, int num_readres);
    void update_stat(uint32_t cid);
    void log(Stage stage);

    std::unique_ptr<dramsim3::NewtonSim> _mem;
    std::vector<uint64_t> _total_processed_requests;
    std::vector<uint64_t> _processed_requests;
    int _mem_req_cnt = 0;
    int _burst_cycle;
    std::vector<std::vector<MemoryIOStat>> _stats;
    uint64_t _stat_interval;

    // stats
    uint64_t _stage_cycles;
    uint64_t _total_done_requests;
    double get_avg_bw_util() override;
    uint64_t get_avg_pim_cycle() override;
    void reset_pim_cycle() override;
};

#endif
