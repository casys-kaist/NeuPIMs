#include <fstream>
#include <map>
#include <unordered_set>
#include <vector>

#include "channel_state.h"
#include "common.h"
#include "controller.h"
#include "newton_command_queue.h"
#include "refresh.h"
#include "simple_stats.h"

namespace dramsim3 {
class NewtonController : public Controller {
  public:
    NewtonController(int channel, const Config &config, const Timing &timing);

    void ClockTick() override;
    bool WillAcceptTransaction(uint64_t hex_addr,
                               TransactionType req_type) override; // const
    bool AddTransaction(Transaction trans) override;
    int QueueUsage() const override;
    // Stats output
    void PrintEpochStats() override;
    void PrintFinalStats() override;
    void ResetStats() override { simple_stats_.Reset(); }
    std::pair<uint64_t, TransactionType> ReturnDoneTrans(uint64_t clock) override;

    int channel_id_;

    // stat for pim utilization
    void ResetPIMCycle() override;
    uint64_t GetPIMCycle() override;

  private:
    uint64_t clk_;
    const Config &config_;
    SimpleStats simple_stats_;
    ChannelState channel_state_;
    NewtonCommandQueue pim_cmd_queue_;
    Refresh refresh_;
    uint64_t last_issue_clk_;

    // queue that takes transactions from CPU side
    std::vector<Transaction> read_queue_;
    std::vector<Transaction> write_buffer_;
    std::vector<Transaction> pim_queue_; // gsheo: add pim queue

    // transactions that are not completed, use map for convenience
    std::multimap<uint64_t, Transaction> pending_rd_q_;
    std::multimap<uint64_t, Transaction> pending_wr_q_;
    std::multimap<uint64_t, Transaction> pending_pim_q_; // gsheo: add pim queue

    // completed transactions
    std::vector<Transaction> return_queue_;

    // row buffer policy
    RowBufPolicy row_buf_policy_;

    // used to calculate inter-arrival latency
    uint64_t last_trans_clk_;

    // transaction queueing
    bool rw_dependency_lock_;
    uint64_t rw_dependency_addr_;
    int write_draining_;
    void ScheduleTransaction();
    void IssueCommand(const Command &tmp_cmd);
    Command TransToCommand(const Transaction &trans);
    void UpdateCommandStats(const Command &cmd);
    void PrintTransactionQueue() const;
    // pim
    Command DecodePIMTransaction(const Transaction &trans);
};
} // namespace dramsim3