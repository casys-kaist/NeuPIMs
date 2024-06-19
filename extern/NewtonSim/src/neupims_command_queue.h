// #ifndef __COMMAND_QUEUE_H
// #define __COMMAND_QUEUE_H

#include <unordered_set>
#include <vector>

#include "channel_state.h"
#include "command_queue.h"
#include "common.h"
#include "configuration.h"
#include "simple_stats.h"

namespace dramsim3 {

class NeuPIMSCommandQueue {
  public:
    NeuPIMSCommandQueue(int channel_id, const Config &config, ChannelState &channel_state,
                        SimpleStats &simple_stats);
    Command GetCommandToIssue(std::pair<int, int> refresh_slack);
    Command FinishRefresh();
    void ClockTick();
    bool WillAcceptCommand(int rank, int bankgroup, int bank) const;
    bool AddCommand(Command cmd);
    bool QueueEmpty() const;
    int QueueUsage() const;
    bool QueueEmpty(int rank) const;
    int GetPIMQueueSize() const;
    void FinishGwrite() {
        is_gwriting_ = false;
        remain_slack_ = 0;
    }
    void PrintAllQueue() const; // for debugging
    std::vector<bool> rank_q_empty;
    uint64_t total_pim_cycles_;
    void ResetPIMCycle() { total_pim_cycles_ = 0; }
    uint64_t GetPIMCycle() { return total_pim_cycles_; }

  private:
    bool ArbitratePrecharge(const CMDIterator &cmd_it, const CMDQueue &queue) const;
    bool HasRWDependency(const CMDIterator &cmd_it, const CMDQueue &queue) const;
    Command GetFirstReadyInQueue(CMDQueue &queue, std::pair<int, int> refresh_slack);
    Command GetReadyInPIMQueue(std::pair<int, int> refresh_slack);
    int GetQueueIndex(int rank, int bankgroup, int bank) const;
    CMDQueue &GetQueue(bool is_pimq_cmd, int rank, int bankgroup, int bank);
    CMDQueue &GetNextQueue();
    void GetRefQIndices(const Command &ref);
    void EraseRWCommand(const Command &cmd);
    Command PrepRefCmd(const CMDIterator &it, const Command &ref) const;
    bool CanMeetRefreshDeadline(const CMDIterator cmd_it, std::pair<int, int> refresh_slack);
    // for debug
    void PrintQueue(CMDQueue &queue) const;

    QueueStructure queue_structure_;
    const Config &config_;
    ChannelState &channel_state_;
    SimpleStats &simple_stats_;

    std::vector<CMDQueue> queues_;
    CMDQueue pim_queue_;

    // Refresh related data structures
    std::unordered_set<int> ref_q_indices_;
    bool is_in_ref_;
    bool is_pim_mode_;
    bool skip_pim_;

    int num_queues_;
    size_t queue_size_;
    size_t pim_cmd_queue_size_; // gsheo: add pim queue size
    int queue_idx_;
    uint64_t clk_;

    int channel_id_;
    int remain_slack_ = 0;
    int reserved_row_for_pim_ = -1;

    bool is_gwriting_;
    Address gwrite_target_;
};

} // namespace dramsim3
// #endif
