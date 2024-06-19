#pragma once
#include "Scheduler.h"

class OrcaScheduler : public Scheduler {
   public:
    OrcaScheduler(SimulationConfig config, const cycle_type *core_cycle);
    void cycle() override;

   private:
    uint32_t _max_active_reqs;
};
