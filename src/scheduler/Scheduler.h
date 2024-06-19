#pragma once
#include "../Common.h"
#include "../Model.h"
#include "../ModelProgram.h"
#include "../StageProgram.h"

class Scheduler {
   public:
    Scheduler(SimulationConfig config, const cycle_type *core_cycle);
    void launch(Ptr<Model> model);
    Tile &top_tile1(uint32_t core_id);
    Tile &top_tile2(uint32_t core_id);
    void get_tile1(uint32_t core_id);
    void get_tile2(uint32_t core_id);

    bool finish_tile(uint32_t core_id, Tile &tile);
    bool empty1();
    bool empty2();
    bool running();

    void print_stat();

    bool has_stage_changed() { return _has_stage_changed; }
    Stage get_prev_stage() { return _prev_stage; }
    void reset_has_stage_changed_status() { _has_stage_changed = false; }

    /* for communicating inference request & response with Client */
    virtual void cycle();
    void add_request(std::shared_ptr<InferRequest> request);
    bool has_completed_request();
    std::shared_ptr<InferRequest> pop_completed_request();

   protected:
    // xxx think of better way to check lifetime of operation.
    // maybe operation stat is not the name you want.
    typedef struct {
        uint32_t id;
        uint32_t request_id;
        std::string name;
        // xxx necessary?
        // bool launched;
        cycle_type start_cycle;
        uint32_t total_tiles;
        uint32_t remain_tiles;
        uint32_t launched_tiles;
    } RunningOperationStat;

    const cycle_type *_core_cycle;
    Ptr<Model> _model;

    std::unique_ptr<StageProgram> _model_program1;
    std::unique_ptr<StageProgram> _model_program2;
    std::deque<Tile> _executable_tile_queue1;
    std::deque<Tile> _executable_tile_queue2;

    SimulationConfig _config;
    // xxx necessary?
    robin_hood::unordered_map<uint32_t, RunningOperationStat> _finished_operation_stats;
    robin_hood::unordered_map<uint32_t, RunningOperationStat> _active_operation_stats;

    Stage _prev_stage;  // for stat
    bool _has_stage_changed;

    virtual void refresh_status1();
    virtual void refresh_status2();

    uint32_t count_active_operations();

    uint32_t _cycles;
    std::deque<std::shared_ptr<InferRequest>> _request_queue;
    std::queue<std::shared_ptr<InferRequest>> _completed_request_queue;
    std::vector<std::vector<Ptr<InferRequest>>> _active_request_queues;
    std::vector<std::vector<uint32_t>> _active_request_latency_queues;
    std::vector<uint32_t> _active_request_accum_latencys;
    uint32_t _max_batch_size;
    uint32_t _max_active_reqs;

    std::vector<Ptr<InferRequest>> _breq1;
    std::vector<Ptr<InferRequest>> _breq2;

    // channel load balancing
    bool _ch_load_balancing;
    uint32_t _next_ch;
    bool compare_by_seqlen(const Ptr<InferRequest> &a, const Ptr<InferRequest> &b) {
        return a->input_size > b->input_size;
    }

    // model dimension
    uint32_t _nh;
    uint32_t _dk;
    uint32_t _effective_e;

    // memory spec
    uint32_t _dram_channels;
    uint32_t _dram_page_size;  // params
    uint32_t _dram_banks_per_ch;
    uint32_t _gwrite_latency;
    uint32_t _gemv_latency;

    void init_batches();
    void allocate_requests();  // allocate channel & assign kv cache
    void group_sub_batches();  // sub-batch interleaving algorithm
    int estimate_mha_latency(Ptr<InferRequest> request);

    int allocate_pim_tile(uint32_t seq_len);

    bool _partition_alg_simple;
    std::pair<std::vector<int>, std::vector<int>> partition_lists_dp(
        std::vector<uint32_t> latency_list);
    std::pair<std::vector<int>, std::vector<int>> partition_lists_simple(
        std::vector<uint32_t> latency_list);

    void make_program();

    void refresh_stage();
    void finish_program1();
    void finish_program2();

    void cleanup_sub_batch(std::vector<Ptr<InferRequest>> sub_batch);

    uint32_t _active_reqs;

    Stage _stage;
    Stage _init_stage;     // default A, if you want to start from other stage, set it
    bool _just_one_stage;  // default false, if you want to run just one stage, set it

    uint32_t _total_tiles;
    uint32_t _total_available_tiles;
    std::vector<uint32_t> _available_tiles;

    uint32_t _key_period;  // how often to create a page per number of tokens.
    uint32_t _value_period;
    uint32_t _key_page_size;  // # of pim tile in page (related to available_tiles)
    uint32_t _value_page_size;

    // explanation on Stage
    //
    // |     |     A    |     B    |         C        |         D        |     E     |     F     |
    // |-----|:--------:|:--------:|:----------------:|:----------------:|:---------:|:---------:|
    // |  SA | QKVgen#1 | QKVgen#2 | Pj/FFNs/QKVgen#1 | Pj/FFNs/QKVgen#2 | Pj/FFNs#1 | Pj/FFNs#2 |
    // | PIM |     -    | MHA#1    | MHA#2            | MHA#1            |   MHA#2   |     -     |
    //
    // number of layers (variable): N
    // Total execution time: A + B + (C+D)*(N-1) + E + F
    //

    std::vector<std::pair<std::string, uint32_t>> _stage_stats;
};
