#ifndef INTERCONNECT_H
#define INTERCONNECT_H
#include <list>

#include "Common.h"
#include "Logger.h"
#include "Stat.h"
#include "booksim2/Interconnect.hpp"
#include "helper/HelperFunctions.h"

class Interconnect {
   public:
    virtual bool running() = 0;
    virtual void cycle() = 0;
    virtual void push(uint32_t src, uint32_t dest, MemoryAccess *request) = 0;
    virtual bool is_full(uint32_t src, MemoryAccess *request) = 0;
    virtual bool is_empty(uint32_t nid) = 0;
    virtual MemoryAccess *top(uint32_t nid) = 0;
    virtual void pop(uint32_t nid) = 0;
    virtual void print_stats() = 0;

    virtual bool has_memreq1(uint32_t cid) = 0;
    virtual bool has_memreq2(uint32_t cid) = 0;
    virtual MemoryAccess *memreq_top1(uint32_t cid) = 0;
    virtual MemoryAccess *memreq_top2(uint32_t cid) = 0;
    virtual void memreq_pop1(uint32_t cid) = 0;
    virtual void memreq_pop2(uint32_t cid) = 0;

    void log(Stage stage);
    void update_stat(MemoryAccess mem_access, uint64_t ch_idx);
    inline cycle_type get_core_cycle();

   protected:
    SimulationConfig _config;
    uint32_t _n_nodes;
    uint32_t _dram_offset;
    uint64_t _cycles;
    std::vector<std::vector<MemoryIOStat>> _stats;
    MemoryIOStat _stat;
    // this variable is the unit of memory io request counts in core cycles
    // if it is 50, the number of memory io requests are merged in 50 core cycles granularity
    uint64_t _mem_cycle_interval;
};

// Simple without conflict interconnect
class SimpleInterconnect : public Interconnect {
   public:
    SimpleInterconnect(SimulationConfig config);
    virtual bool running() override;
    virtual void cycle() override;
    virtual void push(uint32_t src, uint32_t dest, MemoryAccess *request) override;
    virtual bool is_full(uint32_t src, MemoryAccess *request) override;
    virtual bool is_empty(uint32_t nid) override;
    virtual MemoryAccess *top(uint32_t nid) override;
    virtual void pop(uint32_t nid) override;
    virtual void print_stats() override {}

    virtual bool has_memreq1(uint32_t cid) override;
    virtual bool has_memreq2(uint32_t cid) override;
    virtual MemoryAccess *memreq_top1(uint32_t cid) override;
    virtual MemoryAccess *memreq_top2(uint32_t cid) override;
    virtual void memreq_pop1(uint32_t cid) override;
    virtual void memreq_pop2(uint32_t cid) override;

   private:
    uint32_t _latency;
    double _bandwidth;
    uint32_t _rr_start;
    uint32_t _buffer_size;

    struct Entity {
        cycle_type finish_cycle;
        uint32_t dest;
        MemoryAccess *access;
    };

    std::vector<std::queue<MemoryAccess *>> _out_buffers;  // buffer for (ICNT -> Module)
    std::vector<std::queue<Entity>> _in_buffers;           // buffer for (Module -> ICNT)
    std::vector<bool> _busy_node;

    // memory request queue
    bool _mem_sa_q_turn;  // for checking queue1, queue2 in turn
    std::vector<std::queue<MemoryAccess *>> _mem_req_queue1;
    std::vector<std::queue<MemoryAccess *>> _mem_req_queue2;
};

class Booksim2Interconnect : public Interconnect {
   public:
    Booksim2Interconnect(SimulationConfig config);
    virtual bool running() override;
    virtual void cycle() override;
    virtual void push(uint32_t src, uint32_t dest, MemoryAccess *request) override;
    virtual bool is_full(uint32_t src, MemoryAccess *request) override;
    virtual bool is_empty(uint32_t nid) override;
    virtual MemoryAccess *top(uint32_t nid) override;
    virtual void pop(uint32_t nid) override;
    virtual void print_stats() override;

   private:
    uint32_t _ctrl_size;
    std::string _config_path;
    std::unique_ptr<booksim2::Interconnect> _booksim;

    booksim2::Interconnect::Type get_booksim_type(MemoryAccess *access);
    uint32_t get_packet_size(MemoryAccess *access);
};
#endif