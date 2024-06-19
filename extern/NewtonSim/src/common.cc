#include "common.h"

#include <sys/stat.h>

#include <sstream>
#include <unordered_set>

#include "fmt/format.h"

namespace dramsim3 {

namespace LOGGING_CONFIG {
bool STATUS_CHECK = false;
uint32_t TROUBLE_ADDR = 816834560;  // for logging only specific addr
uint32_t TROUBLE_CHANNEL = 0;       // for logging only specific channel
bool PIMSIM_LOGGING = false;
bool PIMSIM_LOGGING_DEBUG = false;
bool LOGGING_ONLY_TROUBLE_ZONE = true;
}  // namespace LOGGING_CONFIG

std::ostream &operator<<(std::ostream &os, const Command &cmd) {
    std::vector<std::string> command_string = {
        "read", "read_p", "write", "write_p", "activate", "precharge",
        "refresh_bank",  // verilog model doesn't distinguish bank/rank refresh
        "refresh", "self_refresh_enter", "self_refresh_exit",
        // >>> gsheo
        "gwrite", "g_act", "comp", "readres",
        // <<< gsheo
        "WRONG"};
    os << fmt::format("{:<20} {:>3} {:>3} {:>3} {:>3} {:>#8x} {:>#8x}",
                      command_string[static_cast<int>(cmd.cmd_type)], cmd.Channel(), cmd.Rank(),
                      cmd.Bankgroup(), cmd.Bank(), cmd.Row(), cmd.Column());
    return os;
}
// gsheo todo: what is the purpose of this function?
std::ostream &operator<<(std::ostream &os, const Transaction &trans) {
    const std::string trans_type = trans.is_write() ? "WRITE" : "READ";
    os << fmt::format("{:<30} {:>8}", trans.addr, trans_type);
    return os;
}
// gsheo todo: what is the purpose of this function?
std::istream &operator>>(std::istream &is, Transaction &trans) {
    std::unordered_set<std::string> write_types = {"WRITE", "write", "P_MEM_WR", "BOFF"};
    std::string mem_op;
    is >> std::hex >> trans.addr >> mem_op >> std::dec >> trans.added_cycle;
    // >>> gsheo
    if (write_types.count(mem_op) == 1) {
        trans.req_type = TransactionType::WRITE;
    }
    // <<< gsheo
    return is;
}

int GetBitInPos(uint64_t bits, int pos) {
    // given a uint64_t value get the binary value of pos-th bit
    // from MSB to LSB indexed as 63 - 0
    return (bits >> pos) & 1;
}

int LogBase2(int power_of_two) {
    int i = 0;
    while (power_of_two > 1) {
        power_of_two /= 2;
        i++;
    }
    return i;
}

std::vector<std::string> StringSplit(const std::string &s, char delim) {
    std::vector<std::string> elems;
    StringSplit(s, delim, std::back_inserter(elems));
    return elems;
}

template <typename Out>
void StringSplit(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            *(result++) = item;
        }
    }
}

void AbruptExit(const std::string &file, int line) {
    std::cerr << "Exiting Abruptly - " << file << ":" << line << std::endl;
    std::exit(-1);
}

bool DirExist(std::string dir) {
    // courtesy to stackoverflow
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        return false;
    } else if (info.st_mode & S_IFDIR) {
        return true;
    } else {  // exists but is file
        return false;
    }
}

std::string ColorString(Color color) {
    std::string color_code;
    switch (color) {
        case Color::RED:
            color_code = "\033[1;31m";
            break;
        case Color::GREEN:
            color_code = "\033[1;32m";
            break;
        case Color::YELLOW:
            color_code = "\033[1;33m";
            break;
        case Color::BLUE:
            color_code = "\033[1;34m";
            break;
        case Color::MAGENTA:
            color_code = "\033[1;35m";
            break;
        case Color::CYAN:
            color_code = "\033[1;36m";
            break;
        case Color::RESET:
            color_code = "\033[0m";
            break;
        default:
            color_code = "";
    }
    return color_code;
}

std::string HexString(uint64_t addr) { return fmt::format("{:#X}", addr); }

void PrintControllerLog(std::string method_name, int channel_id, int clk, const Command &cmd) {
    if (LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE) {
        if (channel_id != LOGGING_CONFIG::TROUBLE_CHANNEL) return;
        // if (cmd.hex_addr != LOGGING_CONFIG::TROUBLE_ADDR)
        //     return;
    }

    Color color = Color::CYAN;
    method_name = "[" + method_name + "]";

    if (cmd.IsPIMBufferCommand()) color = Color::YELLOW;
    if (cmd.IsLastPIMCmd()) color = Color::MAGENTA;

    std::string color_string = ColorString(color);

    if (cmd.cmd_type == CommandType::COMPS_READRES) {
        PrintImportant(color_string, method_name, "cid:", channel_id, "clk:", clk, "|",
                       cmd.CommandTypeString(), "| (address) rank:", cmd.Rank(),
                       "bankgroup:", cmd.Bankgroup(), "bank:", cmd.Bank(),
                       "addr:", HexString(cmd.hex_addr), "#comps:", cmd.num_comps);
        return;
    }

    if (cmd.cmd_type == CommandType::P_HEADER) {
        PrintImportant(color_string, method_name, "cid:", channel_id, "clk:", clk, "|",
                       cmd.CommandTypeString(), "| (packet) #comps:", cmd.num_comps,
                       "#readres:", cmd.num_readres);
        return;
    }
    PrintImportant(color_string, method_name, "cid:", channel_id, "clk:", clk, "|",
                   cmd.CommandTypeString(), "| (address) rank:", cmd.Rank(),
                   "bankgroup:", cmd.Bankgroup(), "bank:", cmd.Bank(),
                   "addr:", HexString(cmd.hex_addr));
    //    std::hex, std::uppercase, cmd.hex_addr, std::dec
}

void PrintTransactionLog(std::string method_name, int channel_id, int clk,
                         const Transaction &trans) {
    if (LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE) {
        if (channel_id != LOGGING_CONFIG::TROUBLE_CHANNEL) return;
        // if (trans.addr != LOGGING_CONFIG::TROUBLE_ADDR)
        //     return;
    }

    Color color = Color::MAGENTA;
    method_name = "[" + method_name + "]";

    std::string color_string = ColorString(color);

    PrintImportant(color_string, method_name, "cid:", channel_id, "clk:", clk, "|",
                   trans.TransactionTypeString(), "| addr:", HexString(trans.addr));
    //    std::hex, std::uppercase, cmd.hex_addr, std::dec
}

}  // namespace dramsim3