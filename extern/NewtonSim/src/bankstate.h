#ifndef __BANKSTATE_H
#define __BANKSTATE_H

#include <vector>

#include "common.h"

namespace dramsim3 {

class BankState {
   public:
    BankState(bool enable_dual_buffer);
    /* bank state description */
    /* The row corresponding to OPEN: open_row_ is active */
    /* CLOSED: No row is open, active */
    /* SREF: self-refresh status, waiting for refresh to end.
     * * SREF_EXIT command.  */
    /* PIM: State for Newton computation. (new)  */
    /* PD, SIZE: invalid state. */
    // gsheo: With OPENforPIM, whether it's 32 or k
    // Since there will be a known number of COMP commands issued, before that
    // Do not precharge until COMP and READRES are completed safely
    // Don't do it!
    enum class State { OPEN, CLOSED, SREF, PD, SIZE };
    enum class StateDouble {
        OPEN_PIMOPEN = int(State::OPEN) * 10 + int(State::OPEN),
        OPEN_PIMCLOSED = int(State::OPEN) * 10 + int(State::CLOSED),
        CLOSED_PIMOPEN = int(State::CLOSED) * 10 + int(State::OPEN),
        CLOSED_PIMCLOSED = int(State::CLOSED) * 10 + int(State::CLOSED),
    };
    Command GetReadyCommand(const Command &cmd, uint64_t clk) const;

    // Update the state of the bank resulting after the execution of the command
    void UpdateState(const Command &cmd);

    // Update the existing timing constraints for the command
    void UpdateTiming(const CommandType cmd_type, uint64_t time);

    bool IsRowOpen() const { return state_ == State::OPEN; }
    bool IsUsed() const { return IsRowOpen(); }
    bool IsPIMUsed() const { return pim_state_ == State::OPEN; }
    int OpenRow() const { return open_row_; }
    int PIMOpenRow() const { return enable_dual_buffer_ ? pim_open_row_ : open_row_; }
    int RowHitCount() const { return row_hit_count_; }
    std::string StateToString() const {
        switch (state_) {
            case State::OPEN:
                return "OPEN";
                break;
            case State::CLOSED:
                return "CLOSED";
                break;
            case State::SREF:
                return "SREF";
                break;
            case State::PD:
                return "PD";
                break;
            case State::SIZE:
                return "SIZE";
                break;
            default:
                return "UNKNOWN STATE";
                break;
        }
    }
    std::string PIMStateToString() const {
        State pim_state = enable_dual_buffer_ ? pim_state_ : state_;
        switch (pim_state) {
            case State::OPEN:
                return "OPEN";
                break;
            case State::CLOSED:
                return "CLOSED";
                break;
            case State::SREF:
                return "SREF";
                break;
            case State::PD:
                return "PD";
                break;
            case State::SIZE:
                return "SIZE";
                break;
            default:
                return "UNKNOWN STATE";
                break;
        }
    }

   private:
    bool enable_dual_buffer_;

    // Current state of the Bank
    // Apriori or instantaneously transitions on a command.
    State state_;      // row buffer state
    State pim_state_;  // pim row buffer state
    bool pim_lock_;    // used only for single buffer mode

    // Earliest time when the particular Command can be executed in this bank
    std::vector<uint64_t> cmd_timing_;

    // Currently open row
    int open_row_;      // open row of row buffer
    int pim_open_row_;  // open row of pim row buffer

    // consecutive accesses to one row
    int row_hit_count_;

    // gsheo: PIM Command set count
    int pim_enter_count_;

    void PrintStateAndCommand(Command cmd) const {
        PrintError("cid:", cmd.Channel(), "PIM State:", PIMStateToString(),
                   ", Command:", cmd.CommandTypeString());
    }

    Command GetReadyCommandSingle(const Command &cmd, uint64_t clk) const;
    void UpdateStateSingle(const Command &cmd);

    Command GetReadyPIMCommand(const Command &cmd, uint64_t clk) const;
    Command GetReadyNormalCommand(const Command &cmd, uint64_t clk) const;
    void UpdateNormalState(const Command &cmd);
    void UpdatePIMState(const Command &cmd);
};

}  // namespace dramsim3
#endif
