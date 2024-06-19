#pragma once
#include "Scheduler.h"

class NeuPIMScheduler : public Scheduler {
   public:
    NeuPIMScheduler(SimulationConfig config, const cycle_type *core_cycle);
    void cycle() override;

   private:
    bool can_add_new_request(Ptr<InferRequest> request);
};
