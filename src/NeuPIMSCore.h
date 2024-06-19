#pragma once

#include <list>
#include <memory>
#include <vector>

#include "Dram.h"
#include "SimulationConfig.h"
#include "Sram.h"
#include "Stat.h"

class NeuPIMSCore {
   public:
    NeuPIMSCore(uint32_t id, SimulationConfig config);
    virtual bool running();
    virtual bool can_issue(Tile &next_tile);
    virtual bool can_issue_pim();
    virtual void issue(Tile &in_tile);

    virtual void issue_pim(Tile &in_tile);

    virtual void log();

    virtual Ptr<Tile> pop_finished_tile();

    virtual void cycle();

    // add index to each methods
    virtual bool has_memory_request1(uint32_t index) {
        return _memory_request_queues1[index].size() > 0;
    }
    virtual void pop_memory_request1(uint32_t index) {
        assert(has_memory_request1(index));
        _memory_request_queues1[index].pop();
    }
    virtual MemoryAccess *top_memory_request1(uint32_t index) {
        return _memory_request_queues1[index].front();
    }
    virtual bool has_memory_request2(uint32_t index) {
        return _memory_request_queues2[index].size() > 0;
    }
    virtual void pop_memory_request2(uint32_t index) {
        assert(has_memory_request2(index));
        _memory_request_queues2[index].pop();
    }
    virtual MemoryAccess *top_memory_request2(uint32_t index) {
        return _memory_request_queues2[index].front();
    }
    virtual void push_memory_request1(MemoryAccess *request);
    virtual void push_memory_request2(MemoryAccess *request);

    virtual void push_memory_response(MemoryAccess *response);
    virtual void pim_push_memory_response(MemoryAccess *response);
    virtual void print_stats();
    virtual cycle_type get_compute_cycles() { return _stat_compute_cycle; }

   protected:
    virtual bool can_issue_compute(Instruction &inst);
    virtual bool pim_can_issue_compute(Instruction &inst);
    virtual cycle_type get_inst_compute_cycles(Instruction &inst) = 0;

    const uint32_t _id;
    const SimulationConfig _config;

    cycle_type _core_cycle;
    uint64_t _compute_end_cycle;
    cycle_type _stat_compute_cycle;
    cycle_type _stat_idle_cycle;
    cycle_type _stat_memory_cycle;
    cycle_type _accum_request_rr_cycle;
    cycle_type _max_request_rr_cycle;
    cycle_type _min_request_rr_cycle;
    cycle_type _memory_stall_cycle;
    cycle_type _compute_memory_stall_cycle;
    cycle_type _vector_memory_stall_cycle;
    cycle_type _layernorm_stall_cycle;
    cycle_type _softmax_stall_cycle;
    cycle_type _add_stall_cycle;
    cycle_type _gelu_stall_cycle;
    cycle_type _load_memory_cycle;
    cycle_type _store_memory_cycle;

    /* Vector Unit Params */
    cycle_type _stat_vec_compute_cycle;
    cycle_type _stat_vec_memory_cycle;  // Does not acctuall count yet
    cycle_type _stat_vec_idle_cycle;    // Does not acctuall count yet

    cycle_type _stat_matmul_cycle;
    cycle_type _stat_layernorm_cycle;
    cycle_type _stat_add_cycle;
    cycle_type _stat_gelu_cycle;
    cycle_type _stat_softmax_cycle;

    int _running_layer;
    std::deque<std::shared_ptr<Tile>> _tiles;
    std::deque<std::shared_ptr<Tile>> _pim_tiles;
    std::queue<std::shared_ptr<Tile>> _finished_tiles;

    std::queue<Instruction> _compute_pipeline;
    std::vector<std::queue<Instruction>> _vector_pipelines;

    // SA Sub-batch queue
    std::queue<Instruction> _ld_inst_queue_for_sa;
    std::queue<Instruction> _st_inst_queue_for_sa;
    std::queue<Instruction> _ex_inst_queue_for_sa;

    // PIM Sub-batch queue
    std::queue<Instruction> _ld_inst_queue_for_pim;
    std::queue<Instruction> _st_inst_queue_for_pim;
    std::queue<Instruction> _ex_inst_queue_for_pim;

    // make it to vector
    std::vector<std::queue<MemoryAccess *>> _memory_request_queues1;
    std::vector<std::queue<MemoryAccess *>> _memory_request_queues2;

    std::queue<MemoryAccess *> _memory_request_queue;
    std::queue<MemoryAccess *> _memory_response_queue;
    uint32_t _waiting_write_reqs;

    int _current_spad;
    int _current_acc_spad;
    Sram _spad;
    Sram _acc_spad;
    Sram _pim_spad;
    Sram _pim_acc_spad;
};