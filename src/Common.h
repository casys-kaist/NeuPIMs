#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#undef NDEBUG
#include <robin_hood.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// For backtrace
#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>

#include <csignal>
#include <fstream>

#include "SimulationConfig.h"
#include "Stat.h"
#include "helper/HelperFunctions.h"
#include "nlohmann/json.hpp"

#define SPAD_BASE 0x10000000
#define ACCUM_SPAD_BASE 0x20000000
#define GARBAGE_ADDR 0xFFFFFFFFFFFFFFF
#define KB *1024

#define PAGE_SIZE 4096

#define ADDR_ALIGN 256

using json = nlohmann::json;
template <typename T>
using Ptr = std::shared_ptr<T>;

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

namespace AddressConfig {
extern addr_type alignment;
extern addr_type channel_mask;
extern addr_type channel_offset;

uint32_t mask_channel(addr_type address);
addr_type allocate_address(uint32_t size);
addr_type align(addr_type addr);

uint64_t make_address(int channel, int rank, int bankgroup, int bank, int row, int col);
uint64_t encode_pim_header(int channel, int row, bool for_gwrite, int num_comps, int num_readres);
uint64_t encode_pim_comps_readres(int ch, int row, int num_comps, bool last_cmd);

addr_type switch_co_ch(addr_type addr);
}  // namespace AddressConfig

enum class Color { RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, DEFAULT };

enum class Opcode {
    MOVIN,
    MOVOUT,
    MOVOUT_POOL,
    GEMM_PRELOAD,
    GEMM,
    GEMM_WRITE,
    COMP,
    IM2COL,
    LAYERNORM,
    GELU,
    SOFTMAX,
    ADD,
    BAR,
    PIM_HEADER,
    PIM_GWRITE,
    PIM_COMP,
    PIM_READRES,
    PIM_COMPS_READRES,
    DUMMY,
    SIZE
};

struct Tile;

struct Instruction {
    Opcode opcode;
    cycle_type start_cycle;
    cycle_type finish_cycle;
    std::string id;
    std::vector<std::string> dependent_ids;
    std::string dest_id;
    addr_type dest_addr;
    uint32_t size;
    std::vector<addr_type> src_addrs;
    int spad_id;
    int accum_spad_id;
    uint32_t operand_id = 0;
    addr_type base_addr;

    // for load store instruction operations
    uint32_t tensor_id;

    // for matrix multiplication systolic array utilization
    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;

    bool src_from_accum = false;
    bool valid = true;

    bool is_pim_inst = false;
    bool is_gemv = false;

    std::weak_ptr<Tile> parent_tile;

    std::string repr();
};

struct Tile {
    enum class Status {
        INITIALIZED,
        RUNNING,
        FINISH,
        BAR,
        EMPTY,
    };
    Status status = Status::EMPTY;
    std::string optype;
    uint32_t operation_id;
    uint32_t batch;
    uint32_t Q;
    uint32_t P;
    uint32_t C;
    uint32_t S;
    uint32_t R;

    // assume several matmuls
    std::vector<uint32_t> batches;
    // assume N,K @ K,M
    uint32_t N;
    uint32_t K;
    uint32_t M;

    TileStat stat;
    std::deque<Instruction> instructions;
    bool accum;
    bool skip;
    int spad_id;
    int accum_spad_id;

    // initialized when Tile moves into core.

    // count up when MOVIN op exists,
    // populate accurate memory request when load instruction is decoded
    uint32_t remaining_loads;
    // computation instruction count
    uint32_t remaining_computes;
    // count up when MOVOUT op exists,
    // count up for the compute instruction
    // populate accurate memory request when store instruction is decoded
    uint32_t remaining_accum_io;
    int program_id;  // SA program / PIM program (sub-batch interleaving에 사용)
    std::string repr();
};

enum class MemoryAccessType { READ, WRITE, GWRITE, COMP, READRES, P_HEADER, COMPS_READRES, SIZE };

std::string memAccessTypeString(MemoryAccessType type);
std::string opcodeTypeString(Opcode opcode);

typedef struct MemoryAccess {
    static int req_count;
    static int pre_req_count;

    uint32_t id;
    addr_type dram_address;
    addr_type spad_address;
    uint64_t size;
    MemoryAccessType req_type;
    bool request;
    uint32_t core_id;
    cycle_type start_cycle;
    cycle_type dram_enter_cycle;
    cycle_type dram_finish_cycle;
    int buffer_id;

    static std::vector<MemoryAccess *> from_instruction(Instruction &inst, uint32_t id,
                                                        uint32_t size, MemoryAccessType req_type,
                                                        bool request, uint32_t core_id,
                                                        cycle_type start_cycle, int buffer_id);

    std::weak_ptr<Tile> parent_tile;

    static void log_count() {
        spdlog::info("total pre req count {} / memory request count {}", pre_req_count, req_count);
    }

} MemoryAccess;

uint32_t generate_id();
uint32_t generate_mem_access_id();
json load_config(std::string config_path);
SimulationConfig initialize_config(json config);  // npu config
void initialize_memory_config(std::string mem_config_path);
void initialize_client_config(std::string cli_config_path);
void initialize_model_config(std::string model_config_path);
void initialize_system_config(std::string sys_config_path);

std::string to_hex(uint32_t input);
template <typename... Args>
std::string name_gen(Args... args) {
    std::vector<std::string> strs = {args...};
    assert(!strs.empty());
    std::string ret = "";
    for (auto &str : strs) {
        ret += str + ".";
    }
    ret.resize(ret.size() - 1);
    return ret;
}

class BTensor;
typedef struct {
    // client to scheduler.
    uint32_t id;
    uint32_t arrival_cycle;    // time spend on client == arrival time to scheduler
    uint32_t completed_cycle;  // return time to client

    // request demand
    uint32_t input_size;   // input sequence length
    uint32_t output_size;  // # tokens to generate

    // request status
    bool is_initiated;   // whether initialization phase is done
    uint32_t generated;  // # tokens generated
    // mapped channel
    int channel;

    std::vector<Ptr<BTensor>> K_cache;
    std::vector<Ptr<BTensor>> V_cache;

} InferRequest;

void print_backtrace();
void ast(bool cond);
template <typename T>
std::vector<T> slice(std::vector<T> &inp, int start, int end) {
    if (end <= -1) end = inp.size() + (end + 1);
    return std::vector<T>(inp.begin() + start, inp.begin() + end);
}

template <typename T>
class Singleton {
   protected:
    static T *instance;

   public:
    static T *GetInstance() {
        if (instance == nullptr) instance = new T();

        return instance;
    }
    static void Delete() { delete instance; }
};
template <typename T>
T *Singleton<T>::instance = nullptr;

MemoryAccess *TransToMemoryAccess(Instruction &inst, uint32_t size, uint32_t core_id,
                                  cycle_type start_cycle, int buffer_id);

int LogBase2(int power_of_two);

enum class ModelStageProgramType { SA, PIM };