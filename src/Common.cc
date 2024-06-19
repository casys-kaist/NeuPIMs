#include "Common.h"

uint32_t generate_id() {
    static uint32_t id_counter{0};
    return id_counter++;
}
uint32_t generate_mem_access_id() {
    static uint32_t id_counter{0};
    return id_counter++;
}

namespace AddressConfig {
addr_type alignment =
    Config::global_config.dram_req_size; // BL * dev width / 8 bytes
addr_type channel_mask;                  // not used
addr_type channel_offset;                // not used
} // namespace AddressConfig

int MemoryAccess::req_count = 0;
int MemoryAccess::pre_req_count = 0;

// FIXME: Magic Numbers
uint32_t AddressConfig::mask_channel(addr_type address) {
    const int col_bits = 4;
    const int offset = 6;

    int ch = (address >> (col_bits + offset)) & channel_mask;
    return ch;
}

addr_type AddressConfig::switch_co_ch(addr_type addr) {
    const int num_col_bits = 4;
    const int num_ch_bits = 5;
    const int num_offset = 6;

    const addr_type ch_mask = ((1 << num_ch_bits) - 1)
                              << (num_col_bits + num_offset);
    const addr_type col_mask = ((1 << num_col_bits) - 1) << num_offset;

    const addr_type mask = ch_mask | col_mask;

    addr_type new_col_bits = (addr & (col_mask << num_ch_bits)) >> num_ch_bits;
    addr_type new_ch_bits = (addr & (ch_mask >> num_col_bits)) << num_col_bits;

    addr = addr & (~mask);
    addr = addr | new_col_bits;
    addr = addr | new_ch_bits;

    return addr;
}

// used in NPU-only
// this is creating dram address.
// align cachline size to 4B
// ex) allocate 31 bytes => align to 32 bytes
addr_type AddressConfig::allocate_address(uint32_t size) {
    static addr_type base_addr{0};

    addr_type result = base_addr;
    base_addr += size;
    if (base_addr & (alignment - 1)) {
        base_addr += alignment - (base_addr & (alignment - 1));
    }

    return result;
}

addr_type AddressConfig::align(addr_type addr) {
    addr_type aligned_addr = addr - (addr & (alignment - 1));
    // spdlog::info("align address!! {}", fmt::format("{:#X} to {:#X}", addr,
    // aligned_addr));

    return aligned_addr;
}

std::vector<MemoryAccess *>
MemoryAccess::from_instruction(Instruction &inst, uint32_t id, uint32_t size,
                               MemoryAccessType req_type, bool request,
                               uint32_t core_id, cycle_type start_cycle,
                               int buffer_id, StagePlatform stage_platform) {
    static addr_type const_addr = 0;
    const addr_type max_address = Config::global_config.model_n_embd *
                                  Config::global_config.model_n_embd * 5 * 2 /
                                  Config::global_config.n_tp;

    robin_hood::unordered_set<addr_type> aligned_src_addrs;
    for (auto addr : inst.src_addrs) {
        pre_req_count++;
        const_addr += 2;
        if (const_addr >= max_address) {
            const_addr = 0;
        }
        aligned_src_addrs.insert(
            AddressConfig::align(AddressConfig::switch_co_ch(const_addr)));
    }

    std::vector<MemoryAccess *> ret;
    for (auto &addr : aligned_src_addrs) {
        req_count++;

        MemoryAccess *mem_access = new MemoryAccess{
            .id = id,
            .dram_address = addr,
            .spad_address = inst.dest_addr,
            .size = size,
            .req_type = req_type,
            .request = request,
            .core_id = core_id,
            .start_cycle = start_cycle,
            .buffer_id = buffer_id,
            .parent_tile = inst.parent_tile,
            .stage_platform = stage_platform,
        };
        ret.push_back(mem_access);
    }

    return ret;
}

void PrintColor(Color color, std::string str) {
    return;
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
    default:
        color_code = "";
    }
    std::cout << color_code << str << "\033[0m" << std::endl;
}

SimulationConfig Config::global_config;

SimulationConfig initialize_config(json config) {
    SimulationConfig parsed_config;

    /* Core configs */
    parsed_config.num_cores = config["num_cores"];
    if ((std::string)config["core_type"] == "systolic_os")
        parsed_config.core_type = CoreType::SYSTOLIC_OS;
    else if ((std::string)config["core_type"] == "systolic_ws")
        parsed_config.core_type = CoreType::SYSTOLIC_WS;
    else
        throw std::runtime_error(fmt::format("Not implemented core type {} ",
                                             (std::string)config["core_type"]));
    parsed_config.core_freq = config["core_freq"];
    parsed_config.core_width = config["core_width"];
    parsed_config.core_height = config["core_height"];

    /* Vector configs */
    parsed_config.process_bit = config["process_bit"];

    parsed_config.vector_core_count = config["vector_core_count"];
    parsed_config.vector_core_width = config["vector_core_width"];
    parsed_config.add_latency = config["add_latency"];
    parsed_config.mul_latency = config["mul_latency"];
    parsed_config.exp_latency = config["exp_latency"];
    parsed_config.gelu_latency = config["gelu_latency"];
    parsed_config.add_tree_latency = config["add_tree_latency"];
    parsed_config.scalar_sqrt_latency = config["scalar_sqrt_latency"];
    parsed_config.scalar_add_latency = config["scalar_add_latency"];
    parsed_config.scalar_mul_latency = config["scalar_mul_latency"];

    /* SRAM configs */
    parsed_config.sram_size = config["sram_size"];
    parsed_config.sram_width = config["sram_width"];
    parsed_config.spad_size = config["sram_size"];
    parsed_config.accum_spad_size = config["sram_size"];

    /* log config*/
    parsed_config.operation_log_output_path =
        config["operation_log_output_path"];

    /* Icnt config */
    if ((std::string)config["icnt_type"] == "simple")
        parsed_config.icnt_type = IcntType::SIMPLE;
    else if ((std::string)config["icnt_type"] == "booksim2")
        parsed_config.icnt_type = IcntType::BOOKSIM2;
    else
        throw std::runtime_error(fmt::format("Not implemented icnt type {} ",
                                             (std::string)config["icnt_type"]));
    parsed_config.icnt_freq = config["icnt_freq"];
    if (config.contains("icnt_latency"))
        parsed_config.icnt_latency = config["icnt_latency"];
    if (config.contains("icnt_config_path"))
        parsed_config.icnt_config_path = config["icnt_config_path"];

    parsed_config.precision = config["precision"];
    parsed_config.layout = config["layout"];
    parsed_config.scheduler_type = config["scheduler"];
    return parsed_config;
}

void initialize_memory_config(std::string mem_config_path) {
    json mem_config = load_config(mem_config_path);
    PrintColor(Color::RED, (std::string)mem_config["dram_type"]);
    /* DRAM config */
    if ((std::string)mem_config["dram_type"] == "dram")
        Config::global_config.dram_type = DramType::DRAM;
    else if ((std::string)mem_config["dram_type"] == "newton")
        Config::global_config.dram_type = DramType::NEWTON;
    else if ((std::string)mem_config["dram_type"] == "neupims")
        Config::global_config.dram_type = DramType::NEUPIMS;
    else
        throw std::runtime_error(
            fmt::format("Not implemented dram type {} ",
                        (std::string)mem_config["dram_type"]));
    Config::global_config.dram_freq = mem_config["dram_freq"];

    Config::global_config.dram_channels = mem_config["dram_channels"];
    if (mem_config.contains("dram_req_size"))
        Config::global_config.dram_req_size = mem_config["dram_req_size"];

    /* PIM config */
    if (mem_config.contains("pim_config_path")) {
        Config::global_config.pim_config_path = mem_config["pim_config_path"];
        // DRAM row buffer size (in bytes)
        Config::global_config.dram_page_size = mem_config["dram_page_size"];
        Config::global_config.dram_banks_per_ch =
            mem_config["dram_banks_per_ch"];
        // # params per PIM_COMP command
        Config::global_config.pim_comp_coverage =
            mem_config["pim_comp_coverage"];
    }
    if (mem_config.contains("baseline_exp")) {
        Config::global_config.baseline_exp = mem_config["baseline_exp"];
    } else {
        Config::global_config.baseline_exp = false;
    }

    Config::global_config.HBM_size = (uint64_t)(mem_config["HBM_size"])GB;
    Config::global_config.HBM_act_buf_size =
        (uint64_t)(mem_config["HBM_act_buf_size"])MB;
}

void initialize_client_config(std::string cli_config_path) {
    Config::global_config.request_dataset_path = cli_config_path;

    // json cli_config = load_config(cli_config_path);
    // /* Client config */
    // Config::global_config.request_dataset_path =
    // cli_config["request_dataset_path"];
    // Config::global_config.request_input_seq_len =
    // cli_config["request_input_seq_len"];
    // Config::global_config.request_interval = cli_config["request_interval"];
    // Config::global_config.request_total_cnt =
    // cli_config["request_total_cnt"];
}
void initialize_model_config(std::string model_config_path) {
    json model_config = load_config(model_config_path);
    /* GPT configs */
    Config::global_config.model_name = model_config["model_name"];
    Config::global_config.model_params_b = model_config["model_params_b"];
    Config::global_config.model_vocab_size = model_config["model_vocab_size"];
    Config::global_config.model_n_layer = model_config["model_n_layer"];
    Config::global_config.model_n_head = model_config["model_n_head"];
    Config::global_config.model_n_embd = model_config["model_n_embd"];
    /* parallelism config */
    Config::global_config.n_tp = model_config["n_tp"];
}
void initialize_system_config(std::string sys_config_path) {
    json sys_config = load_config(sys_config_path);
    /* Batch configs */
    if ((std::string)sys_config["run_mode"] == "npu")
        Config::global_config.run_mode = RunMode::NPU_ONLY;
    else if ((std::string)sys_config["run_mode"] == "npu+pim")
        Config::global_config.run_mode = RunMode::NPU_PIM;
    else
        Config::global_config.run_mode = RunMode::NPU_ONLY;

    if (sys_config["sub_batch_mode"]) {
        if (!Config::global_config.baseline_exp)
            assert(Config::global_config.dram_type == DramType::NEUPIMS);
    }
    Config::global_config.sub_batch_mode = sys_config["sub_batch_mode"];

    Config::global_config.kernel_fusion = sys_config["kernel_fusion"];

    Config::global_config.max_seq_len = sys_config["max_seq_len"];
    Config::global_config.max_active_reqs = sys_config["max_active_reqs"];
    Config::global_config.max_batch_size = sys_config["max_batch_size"];
}

json load_config(std::string config_path) {
    json config_json;
    std::ifstream config_file(config_path);
    config_file >> config_json;
    config_file.close();
    return config_json;
}

std::string memAccessTypeString(MemoryAccessType type) {
    switch (type) {
    case (MemoryAccessType::READ):
        return "READ";
    case (MemoryAccessType::WRITE):
        return "WRITE";
    case (MemoryAccessType::GWRITE):
        return "GWRITE";
    case (MemoryAccessType::COMP):
        return "COMP";
    case (MemoryAccessType::READRES):
        return "READRES";
    case (MemoryAccessType::P_HEADER):
        return "P_HEADER";
    case (MemoryAccessType::COMPS_READRES):
        return "COMPS_READRES";
    default:
        assert(0);
    }
    return "Unknown";
}

std::string opcodeTypeString(Opcode opcode) {
    switch (opcode) {
    case (Opcode::MOVIN):
        return "MOVIN";
    case (Opcode::MOVOUT):
        return "MOVOUT";
    case (Opcode::PIM_HEADER):
        return "PIM_HEADER";
    case (Opcode::PIM_COMP):
        return "PIM_COMP";
    case (Opcode::PIM_GWRITE):
        return "PIM_GWRITE";
    case (Opcode::PIM_READRES):
        return "PIM_READRES";
    case (Opcode::PIM_COMPS_READRES):
        return "PIM_COMPS_READRES";
    default:
        return "Unknown";
    }
}

std::string Instruction::repr() {
    std::string ret;
    switch (opcode) {
    case (Opcode::MOVIN):
        ret += "MOVIN";
        break;
    case (Opcode::MOVOUT):
        ret += "MOVOUT";
        break;
    case (Opcode::GEMM_PRELOAD):
        ret += "GEMM_PRELOAD";
        break;
    case (Opcode::GEMM):
        ret += "GEMM";
        break;
    case (Opcode::BAR):
        ret += "BAR";
        break;
    case (Opcode::LAYERNORM):
        ret += "LAYERNORM";
        break;
    case (Opcode::GELU):
        ret += "GELU";
        break;
    case (Opcode::SOFTMAX):
        ret += "SOFTMAX";
        break;
    case (Opcode::ADD):
        ret += "ADD";
        break;
    case (Opcode::DUMMY):
        ret += "DUMMY";
        break;
    }
    ret += " / src_addrs.size() : ";
    ret += std::to_string(src_addrs.size());
    ret += " / dest_addrs : ";
    ret += to_hex(dest_addr);
    return ret;
}

std::string Tile::repr() {
    std::string ret;
    switch (status) {
    case Status::INITIALIZED:
        ret += "init ";
        break;
    case Status::RUNNING:
        ret += "run ";
        break;
    case Status::FINISH:
        ret += "fin ";
        break;
    case Status::BAR:
        ret += "bar ";
        break;
    case Status::EMPTY:
        ret += "emp ";
        break;
    }
    ret += optype + " ";
    ret += std::to_string(operation_id) + " ";

    return ret;
}

std::string to_hex(uint32_t input) {
    std::stringstream addr_as_hex;
    addr_as_hex << std::hex << input;
    return addr_as_hex.str();
}

void print_backtrace() {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    std::raise(SIGINT);
    exit(1);
}

void ast(bool cond) {
    if (cond)
        return;

    print_backtrace();
    spdlog::error("assertion failed");
    std::raise(SIGSEGV);
    // exit(-1);
}

MemoryAccess *TransToMemoryAccess(Instruction &inst, uint32_t size,
                                  uint32_t core_id, cycle_type start_cycle,
                                  int buffer_id, StagePlatform stage_platform) {
    MemoryAccessType req_type;
    switch (inst.opcode) {
    case Opcode::PIM_HEADER:
        req_type = MemoryAccessType::P_HEADER;
        break;
    case Opcode::PIM_GWRITE:
        req_type = MemoryAccessType::GWRITE;
        break;
    case Opcode::PIM_COMP:
        req_type = MemoryAccessType::COMP;
        break;
    case Opcode::PIM_READRES:
        req_type = MemoryAccessType::READRES;
        break;
    case Opcode::PIM_COMPS_READRES:
        req_type = MemoryAccessType::COMPS_READRES;
        break;
    case Opcode::MOVIN:
        req_type = MemoryAccessType::READ;
        break;
    case Opcode::MOVOUT:
        req_type = MemoryAccessType::WRITE;
        break;
    default:
        req_type = MemoryAccessType::SIZE;
        spdlog::error(
            "Fail to translate unknown Instruction to MemoryAccessType");
        break;
    }

    auto it = inst.src_addrs.begin();
    assert(it != inst.src_addrs.end());
    addr_type dram_addr = *it;

    MemoryAccess *mem_request = new MemoryAccess{
        .id = generate_mem_access_id(),
        .dram_address = dram_addr,
        .spad_address = inst.dest_addr,
        .size = size,         //
        .req_type = req_type, //
        .request = true,
        .core_id = core_id,
        .start_cycle = start_cycle,
        .buffer_id = buffer_id,
        .parent_tile = inst.parent_tile,
        .stage_platform = stage_platform,
    };
    return mem_request;
}

int LogBase2(int power_of_two) {
    int i = 0;
    while (power_of_two > 1) {
        power_of_two /= 2;
        i++;
    }
    return i;
}

uint64_t AddressConfig::make_address(int channel, int rank, int bankgroup,
                                     int bank, int row, int col) {
    // rorabgbachco
    // HBM2_8Gb_s128_pim.ini
    uint64_t addr = 0;

    int row_bits = 15;
    int rank_bits = 1;
    int bankgroup_bits = 2;
    int bank_bits = 2;
    int channel_bits = LogBase2(Config::global_config.dram_channels);
    int col_bits = 4;
    int offset = 6;

    addr |= row;

    addr <<= rank_bits;
    addr |= rank;

    addr <<= bankgroup_bits;
    addr |= bankgroup;

    addr <<= bank_bits;
    addr |= bank;

    addr <<= channel_bits;
    addr |= channel;

    addr <<= col_bits;
    addr |= (col & 15);

    addr <<= offset;

    // spdlog::info("(make_address) hexaddr: {}, ch:{}, ra:{}, bg:{}, ba:{},
    // ro:{}, co:{}",
    //              fmt::format("{:#X}", addr), channel, rank, bankgroup, bank,
    //              row, col);
    // uint64_t address =
    //     memory_system_.config_->MakeAddress(channel, rank, bankgroup, bank,
    //     row, col);
    // if (address == addr)
    //     PrintError("SUCCESS");
    // else
    //     PrintError("FAIL");

    return addr;
}

uint64_t AddressConfig::encode_pim_header(int channel, int row, bool for_gwrite,
                                          int num_comps, int num_readres) {
    int gwrite_bit = for_gwrite ? 1 : 0;

    // we can use only 4 bits for column bit
    // use it to shift_amount
    int log_comps = (gwrite_bit << 3) + LogBase2(num_comps);
    int log_readres = LogBase2(num_readres);

    return make_address(channel, log_readres / 16, (log_readres / 4) & 3,
                        log_readres % 4, row, log_comps);
}

uint64_t AddressConfig::encode_pim_comps_readres(int ch, int row, int num_comps,
                                                 bool last_cmd) {
    int ra_bits = 1;
    int bg_bits = 2;
    int ba_bits = 2;

    int bg_mask = (1 << bg_bits) - 1;
    int ba_mask = (1 << ba_bits) - 1;

    num_comps -= 1;

    assert(num_comps < (1 << (ra_bits + bg_bits + ba_bits)));

    int rank = num_comps >> (bg_bits + ba_bits);
    int bankgroup = (num_comps >> ba_bits) & bg_mask;
    int bank = num_comps & ba_mask;
    int col = last_cmd ? 1 : 0;

    return make_address(ch, rank, bankgroup, bank, row, col);
}

// used for sub-batch interleaving
std::string stageToString(Stage stage) {
    static const std::map<Stage, std::string> stageMap = {
        {Stage::A, "A"},           {Stage::B, "B"}, {Stage::C, "C"},
        {Stage::D, "D"},           {Stage::E, "E"}, {Stage::F, "F"},
        {Stage::Finish, "Finish"},
    };

    auto it = stageMap.find(stage);
    return (it != stageMap.end()) ? it->second : "unknown";
}

std::string stagePlatformToString(StagePlatform sp) {
    static const std::map<StagePlatform, std::string> spMap = {
        {StagePlatform::SA, "SA"},
        {StagePlatform::PIM, "PIM"},
    };

    auto it = spMap.find(sp);
    return (it != spMap.end()) ? it->second : "unknown";
}