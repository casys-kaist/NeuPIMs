#include "bankstate.h"

namespace dramsim3 {

BankState::BankState(bool enable_dual_buffer)
    : enable_dual_buffer_(enable_dual_buffer),
      state_(State::CLOSED),
      pim_state_(State::CLOSED),
      pim_open_row_(-1),
      cmd_timing_(static_cast<int>(CommandType::SIZE)),
      open_row_(-1),
      row_hit_count_(0),
      pim_enter_count_(0),
      pim_lock_(false) {
    cmd_timing_[static_cast<int>(CommandType::READ)] = 0;
    cmd_timing_[static_cast<int>(CommandType::READ_PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::WRITE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::WRITE_PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::ACTIVATE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::REFRESH)] = 0;
    cmd_timing_[static_cast<int>(CommandType::SREF_ENTER)] = 0;
    cmd_timing_[static_cast<int>(CommandType::SREF_EXIT)] = 0;
    // >>> gsheo
    cmd_timing_[static_cast<int>(CommandType::GWRITE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::G_ACT)] = 0;
    cmd_timing_[static_cast<int>(CommandType::COMP)] = 0;
    cmd_timing_[static_cast<int>(CommandType::READRES)] = 0;
    cmd_timing_[static_cast<int>(CommandType::PIM_PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::PWRITE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::COMPS_READRES)] = 0;
    // <<< gsheo
}

/* Return the cmd required before issuing the cmd in the current state */
Command BankState::GetReadyCommand(const Command &cmd, uint64_t clk) const {
    if (!enable_dual_buffer_) {
        return GetReadyCommandSingle(cmd, clk);
    }

    // p_header for gwrite = normal buffer cmd
    // p_header for comp-readres = pim buffer cmd
    if (cmd.IsPIMBufferCommand()) {
        return GetReadyPIMCommand(cmd, clk);
    } else if (cmd.IsNormalBufferCommand()) {
        return GetReadyNormalCommand(cmd, clk);
    } else {
        PrintError("(GetReadyCommand) Not Valid Command");
    }
}

void BankState::UpdateState(const Command &cmd) {
    if (cmd.IsPIMHeader()) {
        PrintError("UpdateBankState:PIM_HEADER:");
    }
    if (!enable_dual_buffer_) {
        UpdateStateSingle(cmd);
        return;
    }

    if (cmd.IsPIMBufferCommand()) {
        UpdatePIMState(cmd);
    } else if (cmd.IsNormalBufferCommand())
        UpdateNormalState(cmd);
    else {
        PrintError("(UpdateState) Not Valid Command");
    }
}

Command BankState::GetReadyPIMCommand(const Command &cmd, uint64_t clk) const {
    CommandType required_type = CommandType::SIZE;
    switch (pim_state_) {
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::PWRITE:
                    PrintWarning("Not Ready for PWRITE immediately.., closed");
                    return Command();  // TODO: check it
                    break;
                case CommandType::P_HEADER:
                    if (state_ == State::OPEN && cmd.Row() == open_row_) {
                        required_type = CommandType::PRECHARGE;
                    } else {
                        return Command(cmd);
                    }
                    break;
                case CommandType::G_ACT:
                    if (state_ == State::OPEN && cmd.Row() == open_row_) {
                        PrintError("(GetReadyPIMCommand) Already Normal Buffer open this row");
                    }
                    required_type = cmd.cmd_type;
                    break;
                case CommandType::COMP:
                case CommandType::COMPS_READRES:
                    required_type = CommandType::G_ACT;
                    break;
                case CommandType::READRES:
                case CommandType::PIM_PRECHARGE:  // no need to precharge
                default:
                    std::cerr << "(GetReadyPIMCommand) Channel:" << cmd.Channel()
                              << " Rank:" << cmd.Rank() << " Bankgroup:" << cmd.Bankgroup()
                              << " Bank:" << cmd.Bank() << std::endl;
                    std::cerr << "PIM State: CLOSED, but try " << cmd.CommandTypeString()
                              << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::PWRITE:
                    if (cmd.Row() == pim_open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        PrintWarning("Not Ready for PWRITE immediately.., opened different row",
                                     pim_open_row_, cmd.Row());
                        return Command();  // TODO: check it
                    }
                    break;
                case CommandType::REFRESH:
                case CommandType::REFRESH_BANK:
                case CommandType::SREF_ENTER:
                    required_type = CommandType::PIM_PRECHARGE;
                    break;
                // >>> gsheo
                case CommandType::P_HEADER:
                    if (cmd.Row() == pim_open_row_) {
                        return Command(cmd);
                    } else {
                        required_type = CommandType::PIM_PRECHARGE;
                    }
                    break;
                case CommandType::G_ACT:
                    PrintError("(GetReadyPIMCommand) G_ACT impossible during open");
                    break;
                case CommandType::PIM_PRECHARGE:
                case CommandType::COMP:
                case CommandType::READRES:
                case CommandType::COMPS_READRES:
                    required_type = cmd.cmd_type;
                    break;
                // <<< gsheo
                default:
                    std::cerr << "(GetReadyPIMCommand) Unknown type!" << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::SREF:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:
                case CommandType::GWRITE:
                case CommandType::G_ACT:
                    required_type = CommandType::SREF_EXIT;
                    break;
                case CommandType::COMP:
                case CommandType::READRES:
                case CommandType::COMPS_READRES:
                case CommandType::PIM_PRECHARGE:
                // <<< gsheo
                default:
                    std::cerr << "(GetReadyPIMCommand) Unknown type!" << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::PD:
        case State::SIZE:
            std::cerr << "In unknown state" << std::endl;
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
            break;
    }
    if (required_type != CommandType::SIZE) {
        if (clk >= cmd_timing_[static_cast<int>(required_type)]) {
            // PrintImportant(cmd.CommandTypeString(), "clk:", clk,
            //                "cmd_timing:",
            //                cmd_timing_[static_cast<int>(required_type)]);
            return Command(required_type, cmd.addr, cmd.hex_addr, cmd.is_last_comps, cmd.num_comps);
        }
        // Command fordebug = Command(required_type, cmd.addr, cmd.hex_addr);
        // std::cout << "(GetReadyCommand) cur:" << cmd.CommandTypeString()
        //           << ", required:" << fordebug.CommandTypeString();
        // std::cout << ", clk:" << clk << " , cmd_timing:"
        //           << cmd_timing_[static_cast<int>(required_type)] <<
        //           std::endl;
    }
    return Command();
}
Command BankState::GetReadyNormalCommand(const Command &cmd, uint64_t clk) const {
    CommandType required_type = CommandType::SIZE;
    switch (state_) {
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:  // p_header for gwrite
                    if (pim_state_ == State::OPEN && pim_open_row_ == cmd.Row()) {
                        required_type = CommandType::PIM_PRECHARGE;
                    } else {
                        return Command(cmd);
                    }
                    break;
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::GWRITE:
                    if (pim_state_ == State::OPEN && cmd.Row() == pim_open_row_) {
                        required_type = CommandType::PIM_PRECHARGE;
                        PrintColor(Color::RED,
                                   "(GetReadyPIMCommand) Already PIM Buffer open this row");
                        break;
                    }
                    required_type = CommandType::ACTIVATE;
                    break;
                case CommandType::REFRESH:
                    if (pim_state_ == State::CLOSED) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PIM_PRECHARGE;
                    }
                    break;
                default:
                    std::cerr << "(GetReadyNormalCommand) Unknown type!" << std::endl;
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:
                    if (open_row_ == cmd.Row()) {
                        return Command(cmd);
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::READ:
                case CommandType::WRITE:
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::GWRITE:
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                        // PrintError("Why not handle?");
                    }
                    break;
                case CommandType::REFRESH:
                    required_type = CommandType::PRECHARGE;
                    break;
                default:
                    std::cerr << "(GetReadyNormalCommand) Unknown type!" << std::endl;
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::SREF:
        case State::PD:
        case State::SIZE:
            std::cerr << "In unknown state" << std::endl;
            AbruptExit(__FILE__, __LINE__);
            break;
    }

    if (required_type != CommandType::SIZE) {
        if (clk >= cmd_timing_[static_cast<int>(required_type)]) {
            return Command(required_type, cmd.addr, cmd.hex_addr);
        }
    }
    return Command();
}

void BankState::UpdateNormalState(const Command &cmd) {
    switch (state_) {
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    row_hit_count_++;
                    break;
                case CommandType::GWRITE:
                    break;
                case CommandType::PRECHARGE:
                    state_ = State::CLOSED;
                    open_row_ = -1;
                    row_hit_count_ = 0;
                    break;
                case CommandType::ACTIVATE:
                case CommandType::REFRESH:
                default:
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::REFRESH:
                    if (pim_state_ == State::CLOSED) {
                        break;
                    } else {
                        AbruptExit(__FILE__, __LINE__);
                    }
                case CommandType::REFRESH_BANK:
                    break;
                case CommandType::ACTIVATE:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    break;
                case CommandType::SREF_ENTER:
                    state_ = State::SREF;
                    break;
                case CommandType::GWRITE:
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::READ_PRECHARGE:
                case CommandType::WRITE_PRECHARGE:
                case CommandType::PRECHARGE:
                case CommandType::SREF_EXIT:
                default:
                    std::cout << cmd << std::endl;
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        default:
            AbruptExit(__FILE__, __LINE__);
    }
    return;
}

void BankState::UpdatePIMState(const Command &cmd) {
    // PrintDebug("UpdatePIMState");
    switch (pim_state_) {
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::PWRITE:
                    break;
                case CommandType::COMP:
                case CommandType::READRES:
                case CommandType::COMPS_READRES:
                    break;
                case CommandType::PIM_PRECHARGE:
                    pim_state_ = State::CLOSED;
                    pim_open_row_ = -1;
                    break;
                case CommandType::G_ACT:
                    pim_open_row_ = cmd.Row();
                    break;
                default:
                    std::cout << "(UpdatePIMState) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::PWRITE:
                    PrintError("Cannot PWRITE in PIM_CLOSED state");
                    break;
                case CommandType::G_ACT:
                    pim_state_ = State::OPEN;
                    pim_open_row_ = cmd.Row();
                    break;

                // >>> gsheo
                case CommandType::SREF_ENTER:
                case CommandType::SREF_EXIT:
                case CommandType::REFRESH:  // handle in normal
                case CommandType::REFRESH_BANK:
                case CommandType::COMP:
                case CommandType::READRES:
                case CommandType::COMPS_READRES:
                case CommandType::PIM_PRECHARGE:
                // <<< gsheo
                default:
                    std::cout << "(UpdatePIMState) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::SREF:
            AbruptExit(__FILE__, __LINE__);
            break;
        default:
            std::cout << "(UpdatePIMState) addr: " << HexString(cmd.hex_addr)
                      << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                      << " bank:" << cmd.Bank() << " ";
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
    }
    // PrintDebug("UpdatePIMState Done");
    return;
}

void BankState::UpdateTiming(CommandType cmd_type, uint64_t time) {
    // if (cmd_type == CommandType::COMPS_READRES)
    //     PrintImportant("(UpdateTiming) COMPS_READRES", "time:", time);
    cmd_timing_[static_cast<int>(cmd_type)] =
        std::max(cmd_timing_[static_cast<int>(cmd_type)], time);
    return;
}

/* Return the cmd required before issuing the cmd in the current state */
Command BankState::GetReadyCommandSingle(const Command &cmd, uint64_t clk) const {
    // In case of single buffer, no usage of PIM_PRECHARGE
    CommandType required_type = CommandType::SIZE;
    switch (state_) {
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::GWRITE:
                    required_type = CommandType::ACTIVATE;
                    break;
                case CommandType::P_HEADER:
                    return Command(cmd);
                    break;
                case CommandType::G_ACT:
                case CommandType::REFRESH:
                    required_type = cmd.cmd_type;
                    break;
                case CommandType::COMP:
                    required_type = CommandType::G_ACT;
                    break;
                case CommandType::READRES:
                // <<< gsheo
                default:
                    PrintWarning("(GetReadyCommandSingle) Unknown type! addr: ",
                                 HexString(cmd.hex_addr), "channel:", cmd.Channel(),
                                 "rank:", cmd.Rank(), "bg:", cmd.Bankgroup(), "bank:", cmd.Bank());
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    if (pim_lock_) {
                        PrintStateAndCommand(cmd);
                        std::cerr << "Try to getReadyCommand for READ/WRITE during PIM lock"
                                  << std::endl;
                        AbruptExit(__FILE__, __LINE__);
                    }
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::P_HEADER:
                    if (cmd.Row() == open_row_) {
                        return Command(cmd);
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::GWRITE:  // >>> gsheo
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::G_ACT:
                    PrintError("errrrrr");
                    break;
                case CommandType::REFRESH:
                    required_type = CommandType::PRECHARGE;
                    break;
                case CommandType::COMP:
                case CommandType::READRES:
                    required_type = cmd.cmd_type;
                    break;
                // <<< gsheo
                default:
                    std::cerr << "(GetReadyCommandSingle) Unknown type!" << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::PD:
        case State::SIZE:
            std::cerr << "In unknown state" << std::endl;
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
            break;
    }

    if (required_type != CommandType::SIZE) {
        if (clk >= cmd_timing_[static_cast<int>(required_type)]) {
            return Command(required_type, cmd.addr, cmd.hex_addr, cmd.is_last_comps, cmd.num_comps);
        }
    }
    return Command();
}

void BankState::UpdateStateSingle(const Command &cmd) {
    switch (state_) {
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    row_hit_count_++;
                    break;
                case CommandType::READRES:
                    pim_lock_ = false;
                case CommandType::GWRITE:
                case CommandType::COMP:
                    break;
                case CommandType::PRECHARGE:
                    state_ = State::CLOSED;
                    open_row_ = -1;
                    row_hit_count_ = 0;
                    break;
                case CommandType::G_ACT:  //  check this..
                    // PrintError("G_ACT in OPEN state");
                    break;
                case CommandType::ACTIVATE:
                case CommandType::REFRESH:
                default:
                    std::cout << "(UpdateStateSingle) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::REFRESH:
                    break;
                case CommandType::ACTIVATE:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    break;
                case CommandType::SREF_ENTER:
                    state_ = State::SREF;
                    break;
                // >>> gsheo
                case CommandType::G_ACT:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    pim_lock_ = true;
                    pim_enter_count_++;
                    break;
                // <<< gsheo
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::PRECHARGE:
                case CommandType::GWRITE:
                case CommandType::COMP:
                case CommandType::READRES:
                // <<< gsheo
                default:
                    std::cout << "(UpdateStateSingle) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        default:
            std::cout << "(UpdateStateSingle) addr: " << HexString(cmd.hex_addr)
                      << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                      << " bank:" << cmd.Bank() << " ";
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
    }
    return;
}

}  // namespace dramsim3
