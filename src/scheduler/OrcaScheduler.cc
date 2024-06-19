#include "OrcaScheduler.h"

OrcaScheduler::OrcaScheduler(SimulationConfig config, const cycle_type *core_cycle)
    : Scheduler(config, core_cycle) {
    // [ ] init max_active_reqs from config. memory capacity
    spdlog::info("OrcaScheduler init");
}

void OrcaScheduler::cycle() { Scheduler::cycle(); }
