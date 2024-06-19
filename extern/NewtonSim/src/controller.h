#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <fstream>
#include <map>
#include <unordered_set>
#include <vector>

#include "channel_state.h"
#include "command_queue.h"
#include "common.h"
#include "refresh.h"
#include "simple_stats.h"

namespace dramsim3 {

enum class RowBufPolicy { OPEN_PAGE, CLOSE_PAGE, SIZE };

class Controller {
  public:
    virtual void ClockTick() = 0;
    virtual bool WillAcceptTransaction(uint64_t hex_addr, TransactionType req_type) = 0;
    virtual bool AddTransaction(Transaction trans) = 0;
    virtual int QueueUsage() const = 0;
    // Stats output
    virtual void PrintEpochStats() = 0;
    virtual void PrintFinalStats() = 0;
    virtual void ResetStats() = 0;
    virtual std::pair<uint64_t, TransactionType> ReturnDoneTrans(uint64_t clock) = 0;
    virtual void ResetPIMCycle() = 0;
    virtual uint64_t GetPIMCycle() = 0;
};
} // namespace dramsim3
#endif
