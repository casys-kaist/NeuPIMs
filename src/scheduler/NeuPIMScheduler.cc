#include "NeuPIMScheduler.h"

NeuPIMScheduler::NeuPIMScheduler(SimulationConfig config, const cycle_type *core_cycle)
    : Scheduler(config, core_cycle) {
    spdlog::info("NeuPIMSScheduler init");
}

void NeuPIMScheduler::cycle() {
    assert(0);
    // if (_model_program == nullptr && !_request_queue.empty()) {
    //     uint32_t batch_size = 0;

    //     for (auto it = _request_queue.begin(); it != _request_queue.end(); it++) {
    //         if (batch_size == _max_batch_size) break;
    //         Ptr<InferRequest> request = *it;
    //         assert(request->output_size > request->generated);

    //         if (!request->is_initiated && !can_add_new_request(request)) continue;
    //         batch_request(request);
    //         batch_size++;
    //     }
    //     spdlog::info("batch_size: {}", batch_size);
    // }
    Scheduler::cycle();
}

bool NeuPIMScheduler::can_add_new_request(Ptr<InferRequest> request) {
    // [ ] check allocated K/V caches size
    // use request->input_size
    return true;
}