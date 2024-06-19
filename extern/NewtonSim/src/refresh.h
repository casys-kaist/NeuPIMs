#ifndef __REFRESH_H
#define __REFRESH_H

#include <vector>

#include "channel_state.h"
#include "common.h"
#include "configuration.h"
#include "simple_stats.h"

namespace dramsim3 {

class Refresh {
   public:
    Refresh(const Config &config, ChannelState &channel_state, SimpleStats &simple_stats);
    void ClockTick();
    std::pair<int, int> GetRefreshSlack();

   private:
    uint64_t clk_;
    int refresh_interval_;
    const Config &config_;
    ChannelState &channel_state_;
    RefreshPolicy refresh_policy_;
    SimpleStats &simple_stats_;

    int next_rank_, next_bg_, next_bank_;

    void InsertRefresh();

    void IterateNext();
};

}  // namespace dramsim3

#endif