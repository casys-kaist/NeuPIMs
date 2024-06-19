#ifndef __COMMON_H
#define __COMMON_H

#include <stdint.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace dramsim3 {

struct Address {
    Address() : channel(-1), rank(-1), bankgroup(-1), bank(-1), row(-1), column(-1) {}
    Address(int channel, int rank, int bankgroup, int bank, int row, int column)
        : channel(channel), rank(rank), bankgroup(bankgroup), bank(bank), row(row), column(column) {
    }
    Address(const Address &addr)
        : channel(addr.channel), rank(addr.rank), bankgroup(addr.bankgroup), bank(addr.bank),
          row(addr.row), column(addr.column) {}
    int channel;
    int rank;
    int bankgroup;
    int bank;
    int row;
    int column;
};

inline uint32_t ModuloWidth(uint64_t addr, uint32_t bit_width, uint32_t pos) {
    addr >>= pos;
    auto store = addr;
    addr >>= bit_width;
    addr <<= bit_width;
    return static_cast<uint32_t>(store ^ addr);
}

// extern std::function<Address(uint64_t)> AddressMapping;
int GetBitInPos(uint64_t bits, int pos);
// it's 2017 and c++ std::string still lacks a split function, oh well
std::vector<std::string> StringSplit(const std::string &s, char delim);
template <typename Out> void StringSplit(const std::string &s, char delim, Out result);

int LogBase2(int power_of_two);
void AbruptExit(const std::string &file, int line);
bool DirExist(std::string dir);
enum class Color { RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, DEFAULT, RESET };

std::string ColorString(Color color);

enum class CommandType {
    READ,
    READ_PRECHARGE,
    WRITE,
    WRITE_PRECHARGE,
    ACTIVATE,
    PRECHARGE,
    REFRESH_BANK,
    REFRESH,
    SREF_ENTER,
    SREF_EXIT,
    // >>> gsheo
    GWRITE,
    G_ACT,
    COMP,
    READRES,
    PIM_PRECHARGE,
    PWRITE,
    P_HEADER,
    // <<< gsheo
    // >> neupims
    COMPS_READRES,
    // <<
    SIZE
};

struct Command {
    Command() : cmd_type(CommandType::SIZE), hex_addr(0) {}
    Command(CommandType cmd_type, const Address &addr, uint64_t hex_addr)
        : cmd_type(cmd_type), addr(addr), hex_addr(hex_addr), for_gwrite(false),
          is_last_comps(false), num_comps(0), num_readres(0) {}
    Command(const Command &cmd) {
        cmd_type = cmd.cmd_type;
        addr = cmd.addr;
        hex_addr = cmd.hex_addr;

        for_gwrite = cmd.for_gwrite;
        num_comps = cmd.num_comps;
        num_readres = cmd.num_readres;

        is_last_comps = cmd.is_last_comps;
    }
    // for COMPS_READRES
    Command(CommandType cmd_type, const Address &addr, uint64_t hex_addr, bool is_last,
            int num_comps)
        : cmd_type(cmd_type), addr(addr), hex_addr(hex_addr), is_last_comps(is_last),
          for_gwrite(false), num_comps(num_comps) {}
    // for P_HEADER : for_gwrite is deprecated
    Command(CommandType cmd_type, const Address &addr, uint64_t hex_addr, bool for_gwrite,
            int num_comps, int num_readres)
        : cmd_type(cmd_type), addr(addr), hex_addr(hex_addr), for_gwrite(for_gwrite),
          is_last_comps(false), num_comps(num_comps), num_readres(num_readres) {}

    bool IsValid() const { return cmd_type != CommandType::SIZE; }
    bool IsRefresh() const {
        return cmd_type == CommandType::REFRESH || cmd_type == CommandType::REFRESH_BANK;
    }
    bool IsRead() const {
        return cmd_type == CommandType::READ || cmd_type == CommandType ::READ_PRECHARGE;
    }
    bool IsWrite() const {
        return cmd_type == CommandType ::WRITE || cmd_type == CommandType ::WRITE_PRECHARGE;
    }
    bool IsReadWrite() const { return IsRead() || IsWrite(); }
    bool IsRankCMD() const {
        return cmd_type == CommandType::REFRESH || cmd_type == CommandType::SREF_ENTER ||
               cmd_type == CommandType::SREF_EXIT;
    }
    bool IsChannelCMD() const {
        return cmd_type == CommandType::COMP || cmd_type == CommandType::READRES ||
               cmd_type == CommandType::COMPS_READRES;
    }
    bool IsPIMHeader() const { return cmd_type == CommandType::P_HEADER; }

    bool IsPIMCommand() const {
        return cmd_type == CommandType::GWRITE || cmd_type == CommandType::COMP ||
               cmd_type == CommandType::READRES || cmd_type == CommandType::COMPS_READRES;
    }
    bool PIMQCommand() const {
        return cmd_type == CommandType::GWRITE || cmd_type == CommandType::COMP ||
               cmd_type == CommandType::READRES || cmd_type == CommandType::P_HEADER ||
               cmd_type == CommandType::COMPS_READRES;
    }
    bool IsPIMComp() const { return cmd_type == CommandType::COMP; }
    bool IsReadRes() const { return cmd_type == CommandType::READRES; }
    bool IsCompsReadres() const { return cmd_type == CommandType::COMPS_READRES; }
    bool IsGwrite() const { return cmd_type == CommandType::GWRITE; }
    bool NeedRefreshCheck() const { return cmd_type == CommandType::P_HEADER; }
    bool IsPIMPrecharge() const { return cmd_type == CommandType::PIM_PRECHARGE; }
    bool IsNormalBufferCommand() const {
        bool p_header_for_gwrite = for_gwrite && cmd_type == CommandType::P_HEADER;

        return cmd_type == CommandType::READ || cmd_type == CommandType::READ_PRECHARGE ||
               cmd_type == CommandType::WRITE || cmd_type == CommandType::WRITE_PRECHARGE ||
               cmd_type == CommandType::ACTIVATE || cmd_type == CommandType::PRECHARGE ||
               cmd_type == CommandType::REFRESH_BANK || cmd_type == CommandType::REFRESH ||
               cmd_type == CommandType::SREF_ENTER || cmd_type == CommandType::SREF_EXIT ||
               cmd_type == CommandType::GWRITE || p_header_for_gwrite;
    }
    bool IsPIMBufferCommand() const {
        bool p_header_for_gemv = !for_gwrite && cmd_type == CommandType::P_HEADER;

        return cmd_type == CommandType::G_ACT || cmd_type == CommandType::COMP ||
               cmd_type == CommandType::READRES || cmd_type == CommandType::PIM_PRECHARGE ||
               cmd_type == CommandType::PWRITE || p_header_for_gemv ||
               cmd_type == CommandType::COMPS_READRES;
    }
    bool IsLastPIMCmd() const {
        // `is_last_comps` in COMP does not mean last command
        return is_last_comps &&
               (cmd_type == CommandType::READRES || cmd_type == CommandType::COMPS_READRES);
    }
    // <<< gsheo
    CommandType cmd_type;
    Address addr;
    uint64_t hex_addr;

    // for pim_header
    bool for_gwrite; // deprecated
    int num_comps;   // also for comps_readres
    int num_readres;

    bool is_last_comps;

    int Channel() const { return addr.channel; }
    int Rank() const { return addr.rank; }
    int Bankgroup() const { return addr.bankgroup; }
    int Bank() const { return addr.bank; }
    int Row() const { return addr.row; }
    int Column() const { return addr.column; }

    friend std::ostream &operator<<(std::ostream &os, const Command &cmd);

    std::string CommandTypeString() const {
        switch (cmd_type) {
        case CommandType::READ:
            return "READ";
            break;
        case CommandType::READ_PRECHARGE:
            return "READ_PRECHARGE";
            break;
        case CommandType::WRITE:
            return "WRITE";
            break;
        case CommandType::WRITE_PRECHARGE:
            return "WRITE_PRECHARGE";
            break;
        case CommandType::ACTIVATE:
            return "ACTIVATE";
            break;
        case CommandType::PRECHARGE:
            return "PRECHARGE";
            break;
        case CommandType::REFRESH_BANK:
            return "REFRESH_BANK";
            break;
        case CommandType::REFRESH:
            return "REFRESH";
            break;
        case CommandType::SREF_ENTER:
            return "SREF_ENTER";
            break;
        case CommandType::SREF_EXIT:
            return "SREF_EXIT";
            break;
        case CommandType::GWRITE:
            return "GWRITE";
            break;
        case CommandType::G_ACT:
            return "G_ACT";
            break;
        case CommandType::COMP:
            return "COMP";
            break;
        case CommandType::READRES:
            return "READRES";
            break;
        case CommandType::PIM_PRECHARGE:
            return "PIM_PRECHARGE";
            break;
        case CommandType::P_HEADER:
            return "P_HEADER";
            break;
        case CommandType::PWRITE:
            return "PWRITE";
            break;
        case CommandType::SIZE:
            return "SIZE";
            break;
        case CommandType::COMPS_READRES:
            return "COMPS_READRES";
            break;
        default:
            return "UNKNOWN";
            break;
        }
    }
};

// >>> gsheo
// MUST be same order in MemoryAccessType of ONNXim (NPU-side simulator)
enum class TransactionType {
    READ,
    WRITE,
    GWRITE,
    COMP,
    READRES,
    P_HEADER,
    COMPS_READRES, // >> neupims
    PWRITE,        // >> unused
    SIZE
};

// <<< gsheo

struct Transaction {
    Transaction() {}
    Transaction(uint64_t addr, TransactionType req_type)
        : addr(addr), added_cycle(0), complete_cycle(0), req_type(req_type) {}
    Transaction(const Transaction &tran)
        : addr(tran.addr), added_cycle(tran.added_cycle), complete_cycle(tran.complete_cycle),
          req_type(tran.req_type) {}
    uint64_t addr;
    uint64_t added_cycle;
    uint64_t complete_cycle;
    TransactionType req_type;
    bool is_dram_trans() const {
        return req_type == TransactionType::WRITE || req_type == TransactionType::READ;
    }
    bool is_write() const { return req_type == TransactionType::WRITE; }
    bool is_read() const { return req_type == TransactionType::READ; }

    friend std::ostream &operator<<(std::ostream &os, const Transaction &trans);
    friend std::istream &operator>>(std::istream &is, Transaction &trans);

    std::string TransactionTypeString() const {
        switch (req_type) {
        case TransactionType::READ:
            return "READ";
            break;
        case TransactionType::WRITE:
            return "WRITE";
            break;
        case TransactionType::GWRITE:
            return "GWRITE";
            break;
        case TransactionType::COMP:
            return "COMP";
            break;
        case TransactionType::READRES:
            return "READRES";
            break;
        case TransactionType::PWRITE:
            return "PWRITE";
            break;
        case TransactionType::P_HEADER:
            return "P_HEADER";
            break;
        case TransactionType::COMPS_READRES:
            return "COMPS_READRES";
            break;
        default:
            return "UNKNOWN";
        }
    }
};
// #ifndef LOGGING_CONFIG_H
// #define LOGGING_CONFIG_H
// ... (your header content)
namespace LOGGING_CONFIG {
extern bool STATUS_CHECK;
extern uint32_t TROUBLE_ADDR;    // for logging only specific addr
extern uint32_t TROUBLE_CHANNEL; // for logging only specific channel
extern bool PIMSIM_LOGGING;
extern bool PIMSIM_LOGGING_DEBUG;
extern bool LOGGING_ONLY_TROUBLE_ZONE;
} // namespace LOGGING_CONFIG
// #endif // EXAMPLE_H

template <typename T> void Print(T t) { std::cout << t << "\033[0m" << std::endl; }

template <typename T, typename... Args> void Print(T t, Args... args) {
    std::cout << t << " ";
    Print(args...);
}

template <typename... Args> void PrintColor(Color color, Args... args) {
    if (!LOGGING_CONFIG::PIMSIM_LOGGING_DEBUG)
        return;
    std::cout << ColorString(color);
    Print(args...);
}

template <typename... Args> void PrintError(Args... args) {
    std::cout << ColorString(Color::RED);
    Print(args...);
    AbruptExit(__FILE__, __LINE__);
}

template <typename... Args> void PrintWarning(Args... args) {
    if (!LOGGING_CONFIG::PIMSIM_LOGGING_DEBUG)
        return;
    std::cout << ColorString(Color::MAGENTA);
    Print(args...);
}

template <typename... Args> void PrintImportant(Args... args) {
    if (!LOGGING_CONFIG::PIMSIM_LOGGING)
        return;
    std::cout << ColorString(Color::YELLOW);
    Print(args...);
}

template <typename... Args> void PrintDebug(Args... args) {
    // when you fix bugs -> do not use this method
    // if you want to leave log, use other method instead
    std::cout << ColorString(Color::BLUE);
    Print(args...);
}

template <typename... Args> void PrintInfo(Args... args) {
    if (!LOGGING_CONFIG::PIMSIM_LOGGING_DEBUG)
        return;
    std::cout << ColorString(Color::CYAN);
    Print(args...);
}

template <typename... Args> void PrintGreen(Args... args) {
    if (!LOGGING_CONFIG::PIMSIM_LOGGING_DEBUG)
        return;
    std::cout << ColorString(Color::GREEN);
    Print(args...);
}
std::string HexString(uint64_t addr);

void PrintControllerLog(std::string method_name, int channel_id, int clk, const Command &cmd);
void PrintTransactionLog(std::string method_name, int channel_id, int clk,
                         const Transaction &trans);
} // namespace dramsim3
#endif
