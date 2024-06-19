#pragma once
#include "../Common.h"
#include "../Model.h"
#include "../ModelProgram.h"

class Scheduler {
   public:
    Scheduler(SimulationConfig config, const cycle_type *core_cycle);
    void launch(Ptr<Model> model);
    Tile &top_tile(uint32_t core_id);
    void get_tile(uint32_t core_id);
    void finish_tile(uint32_t core_id, Tile &tile);
    bool empty();
    bool running();

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
    std::unique_ptr<ModelProgram> _model_program;
    std::deque<Tile> _executable_tile_queue;
    SimulationConfig _config;
    // xxx necessary?
    robin_hood::unordered_map<uint32_t, RunningOperationStat> _finished_operation_stats;
    robin_hood::unordered_map<uint32_t, RunningOperationStat> _active_operation_stats;
    virtual void refresh_status();
    uint32_t count_active_operations();

    uint32_t _cycles;
    std::deque<std::shared_ptr<InferRequest>> _request_queue;
    std::queue<std::shared_ptr<InferRequest>> _completed_request_queue;
    uint32_t _max_batch_size;
    std::vector<Ptr<InferRequest>> _breq;
    uint32_t _next_ch;

    void batch_request(Ptr<InferRequest> request);
    void make_program();
    void finish_program();  //

    uint32_t _active_reqs;
};
