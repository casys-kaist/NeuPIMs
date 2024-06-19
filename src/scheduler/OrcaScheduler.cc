#include "OrcaScheduler.h"

OrcaScheduler::OrcaScheduler(SimulationConfig config, const cycle_type *core_cycle)
    : Scheduler(config, core_cycle) {
    // [ ] init max_active_reqs from config. memory capacity
    spdlog::info("OrcaScheduler init");
    _max_active_reqs = 512;
}

void OrcaScheduler::cycle() {
    if (_model_program == nullptr && !_request_queue.empty()) {
        uint32_t batch_size = 0;

        for (auto it = _request_queue.begin(); it != _request_queue.end(); it++) {
            if (batch_size == _max_batch_size) break;
            Ptr<InferRequest> request = *it;
            assert(request->output_size > request->generated);

            if (!request->is_initiated && _active_reqs >= _max_active_reqs) continue;
            batch_request(request);
            batch_size++;
        }
        spdlog::info("batch_size: {}", batch_size);
    }
    Scheduler::cycle();
}
