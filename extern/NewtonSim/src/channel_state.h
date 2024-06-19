#ifndef __CHANNEL_STATE_H
#define __CHANNEL_STATE_H

#include <cassert>
#include <iomanip>
#include <vector>

#include "bankstate.h"
#include "common.h"
#include "configuration.h"
#include "timing.h"

namespace dramsim3 {

class ChannelState {
  public:
    ChannelState(int channel_id, const Config &config, const Timing &timing);
    Command GetReadyCommand(const Command &cmd, uint64_t clk) const;
    void UpdateState(const Command &cmd);
    void UpdateTiming(const Command &cmd, uint64_t clk);
    void UpdateTimingAndStates(const Command &cmd, uint64_t clk);
    bool ActivationWindowOk(int rank, uint64_t curr_time) const;
    void UpdateActivationTimes(int rank, uint64_t curr_time);
    bool IsRowOpen(int rank, int bankgroup, int bank) const {
        return bank_states_[rank][bankgroup][bank].IsRowOpen();
    }
    bool IsOpenTargetBanks(const Command &cmd); // deprecated.
    bool IsAllBankIdleInRank(int rank) const;
    bool IsPIMIdleInRank(int rank) const;
    bool IsRankSelfRefreshing(int rank) const { return rank_is_sref_[rank]; }
    bool IsRefreshWaiting() const { return !refresh_q_.empty(); }
    bool IsRWPendingOnRef(const Command &cmd) const;
    const Command &PendingRefCommand() const { return refresh_q_.front(); }
    void BankNeedRefresh(int rank, int bankgroup, int bank, bool need);
    void RankNeedRefresh(int rank, bool need);
    int OpenRow(int rank, int bankgroup, int bank) const {
        return bank_states_[rank][bankgroup][bank].OpenRow();
    }
    int RowHitCount(int rank, int bankgroup, int bank) const {
        return bank_states_[rank][bankgroup][bank].RowHitCount();
    };
    void PrintAllBankStates() const;
    bool CheckAllBanksSamePIMOpenRow(int reserved_row);

    int EstimatePIMOperationLatency(const Command &cmd, uint64_t clk);

    std::vector<int> rank_idle_cycles;
    int pim_open_row_ = -1;

  private:
    int channel_id_;
    const Config &config_;
    const Timing &timing_;
    bool comp_overhead_flag_;

    std::vector<bool> rank_is_sref_;
    std::vector<std::vector<std::vector<BankState>>> bank_states_;
    std::vector<Command> refresh_q_;

    std::vector<std::vector<uint64_t>> four_aw_;
    std::vector<std::vector<uint64_t>> thirty_two_aw_;
    bool IsFAWReady(int rank, uint64_t curr_time) const;
    bool Is32AWReady(int rank, uint64_t curr_time) const;
    // Update timing of the bank the command corresponds to
    void UpdateSameBankTiming(const Address &addr,
                              const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
                              uint64_t clk);

    void UpdateSameBankgroupTiming(const Address &addr,
                                   const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
                                   uint64_t clk);

    // Update timing of the other banks in the same bankgroup as the command
    void UpdateOtherBanksSameBankgroupTiming(
        const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
        uint64_t clk);

    // Update timing of banks in the same rank but different bankgroup as the
    // command
    void UpdateOtherBankgroupsSameRankTiming(
        const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
        uint64_t clk);

    // Update timing of banks in a different rank as the command
    void UpdateOtherRanksTiming(const Address &addr,
                                const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
                                uint64_t clk);

    // Update timing of the entire rank (for rank level commands)
    void UpdateSameRankTiming(const Address &addr,
                              const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
                              uint64_t clk);

    // >>> gsheo
    void UpdateTimingForPIM(const Command &cmd,
                            const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
                            uint64_t clk);
    // for COMPS_READRES
    void UpdateChannelTiming(const Command &cmd,
                             const std::vector<std::pair<CommandType, int>> &cmd_timing_list,
                             uint64_t clk);
};

} // namespace dramsim3
#endif
