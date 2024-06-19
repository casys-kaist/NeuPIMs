#include "neupims_command_queue.h"

namespace dramsim3 {

NeuPIMSCommandQueue::NeuPIMSCommandQueue(int channel_id, const Config &config,
                                         ChannelState &channel_state, SimpleStats &simple_stats)
    : channel_id_(channel_id), rank_q_empty(config.ranks, true), config_(config),
      channel_state_(channel_state), simple_stats_(simple_stats), is_in_ref_(false),
      is_gwriting_(false), skip_pim_(false),
      queue_size_(static_cast<size_t>(config_.cmd_queue_size)), queue_idx_(0), clk_(0) {
    if (config_.queue_structure == "PER_BANK") {
        queue_structure_ = QueueStructure::PER_BANK;
        num_queues_ = config_.banks * config_.ranks;
    } else if (config_.queue_structure == "PER_RANK") {
        queue_structure_ = QueueStructure::PER_RANK;
        num_queues_ = config_.ranks;
    } else {
        std::cerr << "Unsupportted queueing structure " << config_.queue_structure << std::endl;
        AbruptExit(__FILE__, __LINE__);
    }
    total_pim_cycles_ = 0;
    queues_.reserve(num_queues_);

    for (int i = 0; i < num_queues_; i++) {
        auto cmd_queue = std::vector<Command>();
        cmd_queue.reserve(config_.cmd_queue_size);
        queues_.push_back(cmd_queue);
    }
    // last queue is for pim command
    pim_cmd_queue_size_ = 128; // TODO: get from config
    pim_queue_ = std::vector<Command>();
    pim_queue_.reserve(pim_cmd_queue_size_);
}
void NeuPIMSCommandQueue::ClockTick() {
    clk_ += 1;
    if (!pim_queue_.empty()) {
        total_pim_cycles_++;
        simple_stats_.Increment("pim_cycles");
    }
}

void NeuPIMSCommandQueue::PrintAllQueue() const {
    int idx = 0;

    // PrintInfo("NeuPIMSCommandQueue::PrintAllQueue()");
    std::string cmd_q_sizes = "";
    for (auto q : queues_) {
        cmd_q_sizes += std::to_string(q.size()) + " ";
        idx++;
    }
    PrintDebug("cmd_q_sizes:", cmd_q_sizes, "pim_q_size:", pim_queue_.size());
}

// called by controller:ClockTick()
Command NeuPIMSCommandQueue::GetCommandToIssue(std::pair<int, int> refresh_slack) {
    if (!skip_pim_) {
        if (!pim_queue_.empty() && !is_in_ref_) {
            // First, check pim queue
            auto pim_cmd = GetReadyInPIMQueue(refresh_slack);
            if (pim_cmd.IsValid()) {
                if (pim_cmd.IsPIMCommand() || pim_cmd.IsPIMHeader())
                    EraseRWCommand(pim_cmd);
                return pim_cmd;
            } else {
                // check whether to find other read/write command
                PrintWarning("cid:", channel_id_, "remain_slack_:", remain_slack_);
                if (remain_slack_ < 10)
                    return Command();
            }
        }
    }

    PrintInfo("cid:", channel_id_, "skip_pim:", skip_pim_, "is_in_ref:", is_in_ref_);

    for (int i = 0; i < num_queues_; i++) {
        auto &queue = GetNextQueue();
        // if we're refresing, skip the command queues that are involved
        if (is_in_ref_) {
            if (ref_q_indices_.find(queue_idx_) != ref_q_indices_.end()) {
                continue;
            }
        }
        auto cmd = GetFirstReadyInQueue(queue, refresh_slack);
        if (cmd.IsValid()) {
            if (cmd.IsReadWrite())
                EraseRWCommand(cmd);
            return cmd;
        }
    }

    return Command();
}

Command NeuPIMSCommandQueue::FinishRefresh() {
    // we can do something fancy here like clearing the R/Ws
    // that already had ACT on the way but by doing that we
    // significantly pushes back the timing for a refresh
    // so we simply implement an ASAP approach
    auto ref = channel_state_.PendingRefCommand();
    if (!is_in_ref_) {
        GetRefQIndices(ref);
        is_in_ref_ = true;
    }

    // either precharge or refresh
    auto cmd = channel_state_.GetReadyCommand(ref, clk_);

    if (cmd.IsRefresh()) {
        ref_q_indices_.clear();
        is_in_ref_ = false;
        skip_pim_ = false;
        remain_slack_ = 0;
        reserved_row_for_pim_ = -1;
    }
    return cmd;
}

bool NeuPIMSCommandQueue::ArbitratePrecharge(const CMDIterator &cmd_it,
                                             const CMDQueue &queue) const {
    auto cmd = *cmd_it;

    if (cmd.IsGwrite()) {
        return true;
    }

    for (auto prev_itr = queue.begin(); prev_itr != cmd_it; prev_itr++) {
        if (prev_itr->Rank() == cmd.Rank() && prev_itr->Bankgroup() == cmd.Bankgroup() &&
            prev_itr->Bank() == cmd.Bank()) {
            return false;
        }
    }

    bool pending_row_hits_exist = false;
    int open_row = channel_state_.OpenRow(cmd.Rank(), cmd.Bankgroup(), cmd.Bank());
    for (auto pending_itr = cmd_it; pending_itr != queue.end(); pending_itr++) {
        if (pending_itr->Row() == open_row && pending_itr->Bank() == cmd.Bank() &&
            pending_itr->Bankgroup() == cmd.Bankgroup() && pending_itr->Rank() == cmd.Rank()) {
            pending_row_hits_exist = true;
            break;
        }
    }

    bool rowhit_limit_reached =
        channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(), cmd.Bank()) >= 4;
    if (!pending_row_hits_exist || rowhit_limit_reached) {
        simple_stats_.Increment("num_ondemand_pres");
        return true;
    }
    return false;
}

bool NeuPIMSCommandQueue::WillAcceptCommand(int rank, int bankgroup, int bank) const {
    if (rank == -1) {
        return pim_queue_.size() < pim_cmd_queue_size_; // pim command queue
    }
    int q_idx = GetQueueIndex(rank, bankgroup, bank);
    return queues_[q_idx].size() < queue_size_;
}

bool NeuPIMSCommandQueue::QueueEmpty() const {
    for (const auto q : queues_) {
        if (!q.empty()) {
            return false;
        }
    }
    // all other cmd queues are empty
    return true;
    // pim_queue_.empty();  // is it correct??
}
bool NeuPIMSCommandQueue::QueueEmpty(int rank) const {
    if (rank == -1)
        return pim_queue_.empty();
    int q_idx = GetQueueIndex(rank, -1, -1);
    return queues_[q_idx].empty();
}

int NeuPIMSCommandQueue::GetPIMQueueSize() const { return pim_queue_.size(); }

bool NeuPIMSCommandQueue::AddCommand(Command cmd) {
    auto &queue = GetQueue(cmd.PIMQCommand(), cmd.Rank(), cmd.Bankgroup(), cmd.Bank());

    if (queue.size() < queue_size_) {
        queue.push_back(cmd);

        if (cmd.Rank() == -1) {
            // mark NOT empty for all ranks
            for (int i = 0; i < rank_q_empty.size(); i++) {
                rank_q_empty[i] = false;
            }
            return true;
        }
        rank_q_empty[cmd.Rank()] = false;
        return true;
    } else {
        return false;
    }
}

CMDQueue &NeuPIMSCommandQueue::GetNextQueue() {
    queue_idx_++;
    if (queue_idx_ == num_queues_) {
        queue_idx_ = 0;
    }
    return queues_[queue_idx_];
}

void NeuPIMSCommandQueue::GetRefQIndices(const Command &ref) {
    if (ref.cmd_type == CommandType::REFRESH) {
        if (queue_structure_ == QueueStructure::PER_BANK) {
            for (int i = 0; i < num_queues_; i++) { // except for pim_q
                if (i / config_.banks == ref.Rank()) {
                    ref_q_indices_.insert(i);
                }
            }
        } else {
            ref_q_indices_.insert(ref.Rank());
        }
    } else { // refb
        int idx = GetQueueIndex(ref.Rank(), ref.Bankgroup(), ref.Bank());
        ref_q_indices_.insert(idx);
    }
    return;
}

int NeuPIMSCommandQueue::GetQueueIndex(int rank, int bankgroup, int bank) const {
    if (rank == -1) {
        return -1;
    }

    if (queue_structure_ == QueueStructure::PER_RANK) {
        return rank;
    } else {
        return rank * config_.banks + bankgroup * config_.banks_per_group + bank;
    }
}

CMDQueue &NeuPIMSCommandQueue::GetQueue(bool is_pimq_cmd, int rank, int bankgroup, int bank) {
    int index;
    if (is_pimq_cmd) {
        return pim_queue_;
    } else {
        index = GetQueueIndex(rank, bankgroup, bank);
    }

    return queues_[index];
}

bool NeuPIMSCommandQueue::CanMeetRefreshDeadline(const CMDIterator cmd_it,
                                                 std::pair<int, int> refresh_slack) {
    int refresh_rank = refresh_slack.first;
    int remain_to_refresh = refresh_slack.second;

    int estimated_latency = channel_state_.EstimatePIMOperationLatency(*cmd_it, clk_);

    int remain_slack = remain_to_refresh - estimated_latency;
    remain_slack_ = 0;
    if (remain_slack > 0) {
        remain_slack_ = remain_slack;
    }

    return remain_slack > 0;
}

void NeuPIMSCommandQueue::PrintQueue(CMDQueue &queue) const {
    // print all commands in pim cmd queue
    std::string commands_in_q = "";
    for (auto cmd_it = queue.begin(); cmd_it != queue.end(); cmd_it++) {
        commands_in_q += cmd_it->CommandTypeString() + " ";
    }
    PrintImportant("cmd_q( cid:", channel_id_, ")", commands_in_q);
}

Command NeuPIMSCommandQueue::GetReadyInPIMQueue(std::pair<int, int> refresh_slack) {
    // estimation = channel_state_.EstimatePIMOperationLatency
    // when pim_mode, execute only pim command 
    // in case of pim header, erase without return, return next pim_cmd & pim_mode on

    for (auto cmd_it = pim_queue_.begin(); cmd_it != pim_queue_.end(); cmd_it++) {
        if (is_gwriting_) {
            if (cmd_it->IsGwrite()) {
                PrintGreen("Get gwrite ready command");
                Command ready_cmd = channel_state_.GetReadyCommand(*cmd_it, clk_);
                return ready_cmd;
            } else {
                // should wait for gwrite complete
                return Command();
            }
        } else if (cmd_it->IsGwrite()) {
            // ready for GWRITE
            bool can_issue_gwrite = CanMeetRefreshDeadline(cmd_it, refresh_slack);
            if (can_issue_gwrite) {
                PrintGreen("is_gwriting ON");
                is_gwriting_ = true;
                gwrite_target_ = cmd_it->addr;

                Command ready_cmd = channel_state_.GetReadyCommand(*cmd_it, clk_);
                return ready_cmd;
            } else {
                skip_pim_ = true;
                if (channel_id_ == 4)
                    PrintWarning("cid:", channel_id_, "skip_pim ON", "gwrite//");
                return Command();
            }
        }

        if (channel_id_ == 0 && cmd_it->cmd_type == CommandType::READRES) {
            PrintInfo("(GetReadyInPIMQueue) survey on cmd:", cmd_it->CommandTypeString());
            // PrintQueue(queue);
        }
        if (cmd_it->IsPIMHeader()) {
            // PrintDebug("(GetReadyInPIMQueue) PIM_HEADER! num_comps:{} num_readres:{}",
            //            cmd_it->num_comps, cmd_it->num_readres);
            // ready for GEMV
            bool can_issue_gemv = CanMeetRefreshDeadline(cmd_it, refresh_slack);
            if (can_issue_gemv) {
                reserved_row_for_pim_ = cmd_it->Row();
                Command cmd = channel_state_.GetReadyCommand(*cmd_it, clk_);

                return cmd;
            } else {
                if (channel_id_ == 4)
                    PrintWarning("cid:", channel_id_, "skip_pim ON", "gemv//");
                skip_pim_ = true;
                return Command();
            }
        }

        Command cmd = channel_state_.GetReadyCommand(*cmd_it, clk_);

        return cmd;
    }

    return Command();
}

Command NeuPIMSCommandQueue::GetFirstReadyInQueue(CMDQueue &queue,
                                                  std::pair<int, int> refresh_slack) {
    // estimation = channel_state_.EstimatePIMOperationLatency
    // in case of pim header, erase without return, return next pim_cmd & pim_mode on

    for (auto cmd_it = queue.begin(); cmd_it != queue.end(); cmd_it++) {
        if (reserved_row_for_pim_ == cmd_it->Row()) {
            assert(reserved_row_for_pim_ != -1);
            // skip commands who wants pim processing row
            continue;
        }
        if (is_gwriting_) {
            if (gwrite_target_.rank == cmd_it->Rank() &&
                gwrite_target_.bankgroup == cmd_it->Bankgroup() &&
                gwrite_target_.bank == cmd_it->Bank())
                continue;
        }
        Command cmd = channel_state_.GetReadyCommand(*cmd_it, clk_);
        if (!cmd.IsValid()) {
            continue;
        }
        if (cmd.cmd_type == CommandType::PRECHARGE) {
            if (!ArbitratePrecharge(cmd_it, queue)) {
                continue;
            }
        } else if (cmd.IsWrite()) {
            if (HasRWDependency(cmd_it, queue)) {
                continue;
            }
        }

        if (remain_slack_ > 0 && !pim_queue_.empty()) {
            // PrintQueue(queue);
            PrintWarning(cmd.CommandTypeString(), "for", cmd_it->CommandTypeString());

            int cmd_overhead = 0;
            int precharge_to_activate = config_.tRP;
            int activate_to_read = config_.tRCDRD;
            int activate_to_write = config_.tRCDWR;
            // int activate_to_gact
            // int activate_to_gwrite
            if (cmd.cmd_type == CommandType::PRECHARGE) {
                // precharge & activate & command overhead

                // todo: modify overhead by cmd
                cmd_overhead = precharge_to_activate + activate_to_write;
                if (remain_slack_ > cmd_overhead) {
                    remain_slack_ -= precharge_to_activate;
                    PrintGreen("SELECT DRAM COMMAND!!", cmd.CommandTypeString());
                    simple_stats_.Increment("num_parallel_prec_cmds");
                    return cmd;
                } else
                    continue;

            } else if (cmd.cmd_type == CommandType::ACTIVATE) {
                // activate & command overhead
                cmd_overhead = activate_to_write;

                if (remain_slack_ > cmd_overhead) {
                    remain_slack_ -= activate_to_write;
                    PrintGreen("SELECT DRAM COMMAND!!", cmd.CommandTypeString());
                    simple_stats_.Increment("num_parallel_act_cmds");
                    return cmd;
                } else
                    continue;
            } else {
                assert(cmd.cmd_type == cmd_it->cmd_type);
                if (cmd.cmd_type == CommandType::READ)
                    simple_stats_.Increment("num_parallel_read_cmds");
                else
                    simple_stats_.Increment("num_parallel_write_cmds");
                PrintGreen("SELECT DRAM COMMAND!!", cmd.CommandTypeString());
                return cmd;
            }
            PrintError("TOUCH!!! remain_slack_:", remain_slack_,
                       pim_queue_.empty() ? "pimq:emtpy" : "pimq:exist");
        }
        return cmd;
    }

    return Command();
}

void NeuPIMSCommandQueue::EraseRWCommand(const Command &cmd) {
    auto &queue = GetQueue(cmd.PIMQCommand(), cmd.Rank(), cmd.Bankgroup(), cmd.Bank());
    bool erase_pim_header = cmd.IsPIMHeader();
    // if (cmd.IsPIMCommand()) {
    //     PrintInfo("cid:", channel_id_, "clk:", clk_, "Erase!!", cmd.CommandTypeString(),
    //                    "addr:", HexString(cmd.hex_addr));

    //     PrintQueue(queue);
    // }
    for (auto cmd_it = queue.begin(); cmd_it != queue.end(); cmd_it++) {
        if (cmd.hex_addr == cmd_it->hex_addr && cmd.cmd_type == cmd_it->cmd_type) {
            int before_cnt = queue.size(); // remove // this is for debug

            queue.erase(cmd_it);
            // if (channel_id_ == 0) {
            //     PrintInfo("(EraseRWCommand) q.size():", before_cnt, "->", queue.size());
            //     PrintQueue(queue);
            // }

            return;
        }
    }
    PrintError("Cannot find cmd!", cmd.CommandTypeString(), "channel_id:", channel_id_);
}

int NeuPIMSCommandQueue::QueueUsage() const {
    int usage = 0;
    for (auto i = queues_.begin(); i != queues_.end(); i++) {
        usage += i->size();
    }
    return usage;
}

// gsheo: Read after write dependency check
// since PIM commands are like read commands from the memory's perspective,
// no write operations should occur at the same address before a PIM command is executed.
// -> set isRead = true for pim command
bool NeuPIMSCommandQueue::HasRWDependency(const CMDIterator &cmd_it, const CMDQueue &queue) const {
    // Read after write has been checked in controller so we only
    // check write after read here
    for (auto it = queue.begin(); it != cmd_it; it++) {
        bool is_read = it->IsRead() || it->IsPIMCommand(); // >>> gsheo
        if (is_read && it->Row() == cmd_it->Row() && it->Column() == cmd_it->Column() &&
            it->Bank() == cmd_it->Bank() && it->Bankgroup() == cmd_it->Bankgroup()) {
            return true;
        }
    }
    return false;
}

} // namespace dramsim3
