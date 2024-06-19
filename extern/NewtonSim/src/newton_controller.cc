#include "newton_controller.h"

#include <iomanip>
#include <iostream>
#include <limits>

namespace dramsim3 {

NewtonController::NewtonController(int channel, const Config &config, const Timing &timing)
    : channel_id_(channel), clk_(0), config_(config), simple_stats_(config_, channel_id_),
      channel_state_(channel, config, timing),
      pim_cmd_queue_(channel_id_, config, channel_state_, simple_stats_),
      refresh_(config, channel_state_, simple_stats_),

      row_buf_policy_(config.row_buf_policy == "CLOSE_PAGE" ? RowBufPolicy::CLOSE_PAGE
                                                            : RowBufPolicy::OPEN_PAGE),
      last_trans_clk_(0), write_draining_(0) {

    read_queue_.reserve(config_.trans_queue_size);
    write_buffer_.reserve(config_.trans_queue_size); // BUG: increase write_buffer size

    rw_dependency_lock_ = false;
    rw_dependency_addr_ = 0;
}

// stat for pim utilization
void NewtonController::ResetPIMCycle() { pim_cmd_queue_.ResetPIMCycle(); }
uint64_t NewtonController::GetPIMCycle() { return pim_cmd_queue_.GetPIMCycle(); }

// - [x] handle pim command
std::pair<uint64_t, TransactionType> NewtonController::ReturnDoneTrans(uint64_t clk) {
    auto it = return_queue_.begin();
    while (it != return_queue_.end()) {
        if (clk >= it->complete_cycle) {
            TransactionType type = TransactionType::SIZE;

            if (it->is_write()) {
                simple_stats_.Increment("num_writes_done");
            } else if (it->is_read()) {
                PrintTransactionLog("ReturnDoneRead", channel_id_, clk_, *it);
                simple_stats_.Increment("num_reads_done");
                simple_stats_.AddValue("read_latency", clk_ - it->added_cycle);
            } else if (it->req_type == TransactionType::GWRITE) {
                pim_cmd_queue_.FinishGwrite();
                PrintInfo("cid:", channel_id_,
                          "GWRITE done, gwrite_latency:", clk_ - it->added_cycle);
                simple_stats_.AddValue("gwrite_latency", clk_ - it->added_cycle);
            } else if (it->req_type == TransactionType::COMP) {
                PrintInfo("COMP done, cid:", channel_id_);
            } else if (it->req_type == TransactionType::READRES) {
                PrintInfo("READRES done, cid:", channel_id_);
                simple_stats_.Increment("num_readres_done");
            }

            auto pair = std::make_pair(it->addr, it->req_type);
            it = return_queue_.erase(it);
            return pair;
        } else {
            ++it;
        }
    }
    return std::make_pair(-1, TransactionType::SIZE);
}

void NewtonController::ClockTick() {
    // update refresh counter
    refresh_.ClockTick();

    bool cmd_issued = false;

    Command cmd;
    if (channel_state_.IsRefreshWaiting()) {
        PrintColor(Color::RED, "Refresh Waiting.., clk:", clk_);
        cmd = pim_cmd_queue_.FinishRefresh();
    }

    if (!cmd.IsValid()) {
        std::pair<int, int> refresh_slack = refresh_.GetRefreshSlack();
        cmd = pim_cmd_queue_.GetCommandToIssue(refresh_slack);
    }

    if (cmd.IsValid()) {
        IssueCommand(cmd);
        cmd_issued = true;
    }

    // power updates pt 1
    for (int i = 0; i < config_.ranks; i++) {
        if (channel_state_.IsRankSelfRefreshing(i)) {
            simple_stats_.IncrementVec("sref_cycles", i);
        } else {
            bool all_idle = channel_state_.IsAllBankIdleInRank(i);
            if (all_idle) {
                simple_stats_.IncrementVec("all_bank_idle_cycles", i);
                channel_state_.rank_idle_cycles[i] += 1;
            } else {
                simple_stats_.IncrementVec("rank_active_cycles", i);
                // reset
                channel_state_.rank_idle_cycles[i] = 0;
            }
        }
    }

    // power updates pt 2: move idle ranks into self-refresh mode to save power
    if (config_.enable_self_refresh && !cmd_issued) {
        for (auto i = 0; i < config_.ranks; i++) {
            if (channel_state_.IsRankSelfRefreshing(i)) {
                // wake up!
                if (!pim_cmd_queue_.rank_q_empty[i]) {
                    auto addr = Address();
                    addr.rank = i;
                    auto cmd = Command(CommandType::SREF_EXIT, addr, -1);
                    cmd = channel_state_.GetReadyCommand(cmd, clk_);
                    if (cmd.IsValid()) {
                        IssueCommand(cmd);
                        break;
                    }
                }
            } else {
                if (pim_cmd_queue_.rank_q_empty[i] &&
                    channel_state_.rank_idle_cycles[i] >= config_.sref_threshold) {
                    auto addr = Address();
                    addr.rank = i;
                    auto cmd = Command(CommandType::SREF_ENTER, addr, -1);
                    cmd = channel_state_.GetReadyCommand(cmd, clk_);
                    if (cmd.IsValid()) {
                        IssueCommand(cmd);
                        break;
                    }
                }
            }
        }
    }

    ScheduleTransaction();
    clk_++;
    pim_cmd_queue_.ClockTick();
    simple_stats_.Increment("num_cycles");

    //>>> gsheo: for debug (to fix infinite loop)
    int interval = 20;
    // channel_id_ == TROUBLE_CHANNEL
    if (clk_ % interval == 0 && LOGGING_CONFIG::STATUS_CHECK) {
        if (LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE) {
            if (channel_id_ != LOGGING_CONFIG::TROUBLE_CHANNEL)
                return;
        }
        PrintDebug("-------NewtonSim Status Check (cid:", channel_id_, ")-------");

        bool clean_related_read =
            read_queue_.empty() && pending_rd_q_.empty() && pim_cmd_queue_.QueueEmpty();
        bool clean_related_write =
            write_buffer_.empty() && pending_wr_q_.empty() && pim_cmd_queue_.QueueEmpty();
        bool clean_related_pim =
            read_queue_.empty() && pending_rd_q_.empty() && pim_cmd_queue_.QueueEmpty(-1);

        if (clean_related_read && clean_related_write && clean_related_pim) {
            PrintDebug("All queue empty!");
            return;
        }
        PrintDebug("clk:", clk_);
        PrintDebug("pim_queue.size:", pim_queue_.size());
        PrintDebug("pim_command_queue size:", pim_cmd_queue_.GetPIMQueueSize());
        PrintDebug("read_queue.size:", read_queue_.size());
        PrintDebug("write_buffer.size:", write_buffer_.size());
        PrintDebug("pending_rd_q_", pending_rd_q_.size());
        PrintDebug("pending_wr_q_", pending_wr_q_.size());
        PrintDebug("pending_pim_q_", pending_pim_q_.size());
        pim_cmd_queue_.PrintAllQueue();
        if (write_buffer_.empty() && !pending_wr_q_.empty() && pim_cmd_queue_.QueueEmpty()) {
            PrintDebug("Something wrong!!");
            PrintDebug("write_buffer is empty, but pending_wr_q_ is not empty");
            // show content:
            // for (auto it = pending_wr_q_.begin(); it != pending_wr_q_.end(); ++it) {
            //     std::cout << "addr:" << (*it).first
            //               << " => type:" << (*it).second.TransactionTypeString() << '\n';
            // }
        }
        if (read_queue_.empty() && !pending_rd_q_.empty() && pim_cmd_queue_.QueueEmpty()) {
            PrintDebug("Something wrong!!");
            PrintDebug("read_queue is empty, but pending_rd_q_ is not empty");
            // show content:
            // for (auto it = pending_rd_q_.begin(); it != pending_rd_q_.end(); ++it) {
            //     std::cout << "addr:" << (*it).first
            //               << " => type:" << (*it).second.TransactionTypeString() << '\n';
            // }
        }
    }
    // <<< gsheo

    // >>> gsheo: for debug (read nothing else, during pim computation)
    // if (clk_ == 22750) {
    //     LOGGING_CONFIG::STATUS_CHECK = true;
    //     LOGGING_CONFIG::PIMSIM_LOGGING = false;
    //     LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE = false;
    //     LOGGING_CONFIG::TROUBLE_CHANNEL = 0;
    // }
    // if (clk_ == 28340) {
    //     LOGGING_CONFIG::STATUS_CHECK = false;
    //     LOGGING_CONFIG::PIMSIM_LOGGING = false;
    //     LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE = false;
    //     // exit(0);
    // }

    // <<< gsheo

    return;
}

bool NewtonController::WillAcceptTransaction(uint64_t hex_addr, TransactionType req_type) {
    bool is_write = req_type == TransactionType::WRITE;
    bool is_read = req_type == TransactionType::READ;

    if (is_write) {
        return write_buffer_.size() < write_buffer_.capacity();
    } else {
        return read_queue_.size() < read_queue_.capacity();
    }
}

bool NewtonController::AddTransaction(Transaction trans) {
    trans.added_cycle = clk_;
    simple_stats_.AddValue("interarrival_latency", clk_ - last_trans_clk_);
    last_trans_clk_ = clk_;

    if (trans.is_write()) {
        PrintTransactionLog("AddTransaction(WR)", channel_id_, clk_, trans);
        if (pending_wr_q_.count(trans.addr) == 0) { // can not merge writes
            pending_wr_q_.insert(std::make_pair(trans.addr, trans));
            write_buffer_.push_back(trans);
        }
        trans.complete_cycle = clk_ + 1;
        return_queue_.push_back(trans);
        return true;
    } else if (trans.is_read()) {
        // if in write buffer, use the write buffer value
        // >> gsheo: debug
        PrintTransactionLog("AddTransaction(RD)", channel_id_, clk_, trans);
        // << debug
        if (pending_wr_q_.count(trans.addr) > 0) {
            trans.complete_cycle = clk_ + 1;
            return_queue_.push_back(trans);
            return true;
        }

        pending_rd_q_.insert(std::make_pair(trans.addr, trans));
        if (pending_rd_q_.count(trans.addr) == 1) {
            read_queue_.push_back(trans);
        }
        return true;
    } else {
        if (trans.req_type == TransactionType::P_HEADER) {
            auto cmd = TransToCommand(trans);
            assert(!cmd.for_gwrite);
            if (cmd.for_gwrite)
                return false; // skip p_header for gwrite
        } else {
            pending_pim_q_.insert(std::make_pair(trans.addr, trans));
            PrintWarning("cid:", channel_id_, "insert to pending_pim_q",
                         trans.TransactionTypeString(), "addr:", HexString(trans.addr));
        }
        read_queue_.push_back(trans);
        return true;
    }
}

void NewtonController::PrintTransactionQueue() const {
    auto queue = read_queue_;
    std::string commands = "";
    for (auto it = queue.begin(); it != queue.end(); it++) {
        commands += it->TransactionTypeString() + " ";
    }
    PrintWarning("READ trans_q (cid:", channel_id_, "):", commands);

    commands = "";
    for (auto it = write_buffer_.begin(); it != write_buffer_.end(); it++) {
        commands += it->TransactionTypeString() + " ";
    }
    PrintWarning("WRITE trans_q (cid:", channel_id_, "):", commands);
}

void NewtonController::ScheduleTransaction() {
    if (!rw_dependency_lock_ && write_draining_ == 0) {
        // we basically have a upper and lower threshold for write buffer
        if ((write_buffer_.size() >= write_buffer_.capacity()) ||
            (write_buffer_.size() > 8 && pim_cmd_queue_.QueueEmpty())) {
            // write_buffer is full or
            // there are transactions more than 8 and cmd_queue is empty
            write_draining_ = write_buffer_.size();
        }
    }

    // in Newton, only after read-write complete, execute PIM
    int pim_q_size = pim_cmd_queue_.GetPIMQueueSize();

    enum QueueToSchedule { READ_Q, WRITE_BUFFER, SIZE };
    QueueToSchedule queue_to_schedule = SIZE;
    // Newton: single row buffer PIM
    if (rw_dependency_lock_)
        queue_to_schedule = READ_Q;
    else if (write_draining_ > 0)
        queue_to_schedule = WRITE_BUFFER;
    else
        queue_to_schedule = READ_Q;

    assert(queue_to_schedule != SIZE);

    std::vector<Transaction> &queue = queue_to_schedule == READ_Q ? read_queue_ : write_buffer_;

    for (auto it = queue.begin(); it != queue.end(); it++) {
        auto cmd = TransToCommand(*it);

        int rank = cmd.IsGwrite() ? -1 : cmd.Rank();

        if (pim_cmd_queue_.WillAcceptCommand(rank, cmd.Bankgroup(), cmd.Bank())) {
            if (cmd.IsWrite()) {
                // Enforce R->W dependency
                if (pending_rd_q_.count(it->addr) > 0) {
                    // if there is read transaction (it->addr),
                    // first push it
                    if (read_queue_.size() > 0) {
                        for (int i = 0; i < read_queue_.size(); i++) {
                            if (read_queue_[i].addr == it->addr) {
                                // PrintDebug("(ScheduleTransaction) R->W dependency:!", it->addr);
                                rw_dependency_lock_ = true;
                                rw_dependency_addr_ = it->addr;
                                break;
                            }
                        }
                    }
                    write_draining_ = 0;
                    break;
                } else if (pending_pim_q_.count(it->addr) > 0) {
                    auto pim_trans_ = pending_pim_q_.find(cmd.hex_addr);
                    if (pim_trans_->second.added_cycle < it->added_cycle) {
                        write_draining_ = 0;
                        break;
                    }
                }
                write_draining_ -= 1;
            }
            if (cmd.IsRead()) {
                if (rw_dependency_lock_ && rw_dependency_addr_ == cmd.hex_addr) {
                    // PrintDebug("(ScheduleTransaction) Solve R->W dependency:!", it->addr);
                    rw_dependency_addr_ = 0;
                    rw_dependency_lock_ = false;
                }
            }
            pim_cmd_queue_.AddCommand(cmd);
            queue.erase(it);
            break;
        }
    }
}

void NewtonController::IssueCommand(const Command &cmd) {
    PrintControllerLog("IssueCommand", channel_id_, clk_, cmd);

    last_issue_clk_ = clk_;

    // if read/write, update pending queue and return queue
    if (cmd.IsRead()) {
        auto num_reads = pending_rd_q_.count(cmd.hex_addr);
        if (num_reads == 0) {
            std::cerr << cmd.hex_addr << " not in read queue! " << std::endl;
            exit(1);
        }
        // if there are multiple reads pending return them all
        while (num_reads > 0) {
            auto it = pending_rd_q_.find(cmd.hex_addr);
            it->second.complete_cycle = clk_ + config_.read_delay;
            return_queue_.push_back(it->second);
            pending_rd_q_.erase(it);
            num_reads -= 1;
        }
    } else if (cmd.IsWrite()) {
        // there should be only 1 write to the same location at a time
        auto it = pending_wr_q_.find(cmd.hex_addr);
        if (it == pending_wr_q_.end()) {
            std::cerr << cmd.hex_addr << " not in write queue!" << std::endl;
            exit(1);
        }
        auto wr_lat = clk_ - it->second.added_cycle + config_.write_delay;
        simple_stats_.AddValue("write_latency", wr_lat);
        pending_wr_q_.erase(it);
    } else if (cmd.IsReadRes() || cmd.IsGwrite() || cmd.IsPIMComp()) {
        auto num_pending_pim_trans = pending_pim_q_.count(cmd.hex_addr);
        if (num_pending_pim_trans == 0) {
            PrintError("cid:", channel_id_, "not in pending pim queue!", cmd.CommandTypeString(),
                       "addr:", HexString(cmd.hex_addr));
        }

        auto it = pending_pim_q_.find(cmd.hex_addr);
        // readres delay == read delay
        if (cmd.IsReadRes())
            it->second.complete_cycle = clk_ + config_.read_delay;
        if (cmd.IsGwrite())
            it->second.complete_cycle = clk_ + config_.gwrite_delay;
        if (cmd.IsPIMComp())
            it->second.complete_cycle = clk_;

        return_queue_.push_back(it->second);
        pending_pim_q_.erase(it);
    } else if (cmd.IsPIMHeader()) {
        PrintInfo("START GEMV");
        // PrintError("you cannot issue PIM header command!");
        return;
    }
    // must update stats before states (for row hits)
    UpdateCommandStats(cmd);
    channel_state_.UpdateTimingAndStates(cmd, clk_);
}

// - [x] translate PIM transaction to command
Command NewtonController::TransToCommand(const Transaction &trans) {
    auto addr = config_.AddressMapping(trans.addr);
    CommandType cmd_type;
    if (row_buf_policy_ == RowBufPolicy::OPEN_PAGE) {
        switch (trans.req_type) {
        case TransactionType::READ:
            cmd_type = CommandType::READ;
            break;
        case TransactionType::WRITE:
            cmd_type = CommandType::WRITE;
            break;
        case TransactionType::GWRITE:
            cmd_type = CommandType::GWRITE;
            break;
        case TransactionType::COMP:
            addr.rank = -1;
            addr.bankgroup = -1;
            addr.bank = -1;
            cmd_type = CommandType::COMP;
            break;
        case TransactionType::READRES:
            return DecodePIMTransaction(trans);
        case TransactionType::PWRITE:
            cmd_type = CommandType::PWRITE;
            break;
        case TransactionType::P_HEADER:
            cmd_type = CommandType::P_HEADER;
            break;
        default:
            break;
        }
    } else {
        cmd_type = trans.is_write() ? CommandType::WRITE_PRECHARGE : CommandType::READ_PRECHARGE;
    }

    if (cmd_type == CommandType::P_HEADER) {
        // decode PIM header packet
        bool for_gwrite = false;
        int num_comps = 0;
        int num_readres = 0;

        int col_low_bits = LogBase2(config_.BL);
        int actual_col_bits = LogBase2(config_.columns) - col_low_bits - 1;
        int column_mask = ~(-1 << actual_col_bits); // 0000000111

        if (addr.column >> actual_col_bits == 1) {
            for_gwrite = true;
            return Command(cmd_type, addr, trans.addr, for_gwrite, 0, 0);
        }

        num_comps = 1 << (addr.column & column_mask);

        int bk_idx = addr.rank * config_.bankgroups * config_.banks_per_group +
                     addr.bankgroup * config_.banks_per_group + addr.bank;

        num_readres = 1 << bk_idx;
        addr.rank = -1;

        return Command(cmd_type, addr, trans.addr, for_gwrite, num_comps, num_readres);
    }
    return Command(cmd_type, addr, trans.addr);
}

Command NewtonController::DecodePIMTransaction(const Transaction &trans) {
    assert(trans.req_type == TransactionType::READRES);
    CommandType cmd_type = CommandType::READRES;

    auto addr = config_.AddressMapping(trans.addr);
    int num_comps = 0;

    num_comps += addr.rank * config_.bankgroups * config_.banks_per_group;
    num_comps += addr.bankgroup * config_.banks_per_group;
    num_comps += addr.bank;
    num_comps += 1;
    // we have only 5 bits usable,
    // encode (num_comps-1) to rabgba bit 

    addr.rank = -1;
    addr.bankgroup = -1;
    addr.bank = -1;
    bool is_last = addr.column == 1;

    // fix num_readres to 1
    return Command(cmd_type, addr, trans.addr, is_last, num_comps);
}

int NewtonController::QueueUsage() const { return pim_cmd_queue_.QueueUsage(); }

void NewtonController::PrintEpochStats() {
    simple_stats_.Increment("epoch_num");
    simple_stats_.PrintEpochStats();

    return;
}

void NewtonController::PrintFinalStats() {
    simple_stats_.PrintFinalStats();

    return;
}

// - [x] add PIM command stats
void NewtonController::UpdateCommandStats(const Command &cmd) {
    switch (cmd.cmd_type) {
    case CommandType::READ:
    case CommandType::READ_PRECHARGE:
        simple_stats_.Increment("num_read_cmds");
        if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(), cmd.Bank()) != 0) {
            simple_stats_.Increment("num_read_row_hits");
        }
        break;
    case CommandType::WRITE:
    case CommandType::WRITE_PRECHARGE:
        simple_stats_.Increment("num_write_cmds");
        if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(), cmd.Bank()) != 0) {
            simple_stats_.Increment("num_write_row_hits");
        }
        break;
    case CommandType::ACTIVATE:
        simple_stats_.Increment("num_act_cmds");
        break;
    case CommandType::PRECHARGE:
        simple_stats_.Increment("num_pre_cmds");
        break;
    case CommandType::REFRESH:
        simple_stats_.Increment("num_ref_cmds");
        break;
    case CommandType::REFRESH_BANK:
        simple_stats_.Increment("num_refb_cmds");
        break;
    case CommandType::SREF_ENTER:
        simple_stats_.Increment("num_srefe_cmds");
        break;
    case CommandType::SREF_EXIT:
        simple_stats_.Increment("num_srefx_cmds");
        break;
    case CommandType::GWRITE:
        simple_stats_.Increment("num_gwrite_cmds");
        break;
    case CommandType::G_ACT:
        simple_stats_.Increment("num_gact_cmds");
        break;
    case CommandType::COMP:
        simple_stats_.Increment("num_comp_cmds");
        break;
    case CommandType::READRES:
        simple_stats_.Increment("num_readres_cmds");
        break;
    case CommandType::PIM_PRECHARGE:
        simple_stats_.Increment("num_pim_precharge_cmds");
        break;
    case CommandType::P_HEADER:
        // simple_stats_.Increment("num_pim_precharge_cmds");
        break;
    case CommandType::PWRITE:
        // simple_stats_.Increment("num_pim_precharge_cmds");
        break;
    default:
        PrintError(cmd.CommandTypeString());
        AbruptExit(__FILE__, __LINE__);
    }
}

} // namespace dramsim3
