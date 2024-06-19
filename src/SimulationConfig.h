#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

typedef uint64_t cycle_type;

enum class CoreType { SYSTOLIC_OS, SYSTOLIC_WS };

enum class DramType { DRAM, NEWTON, NEUPIMS };

enum class IcntType { SIMPLE, BOOKSIM2 };

enum class RunMode { NPU_ONLY, NPU_PIM };

struct SimulationConfig {
    // gpt model config
    std::string model_name;
    uint32_t model_params_b;
    uint32_t model_block_size;
    uint32_t model_vocab_size;
    uint32_t model_n_layer;
    uint32_t model_n_head;
    uint32_t model_n_embd;

    /* Custom Config */
    RunMode run_mode;  // NPU
    bool sub_batch_mode;
    bool ch_load_balancing;
    bool kernel_fusion;
    uint32_t max_batch_size;
    uint32_t max_active_reqs;  // max size of (ready_queue + running_queue) in scheduler
    uint32_t max_seq_len;
    uint64_t HBM_size;          // HBM size in bytes
    uint64_t HBM_act_buf_size;  // HBM activation buffer size in bytes

    /* Core config */
    uint32_t num_cores;
    CoreType core_type;
    uint32_t core_freq;
    uint32_t core_width;
    uint32_t core_height;

    uint32_t n_tp;

    uint32_t vector_core_count;
    uint32_t vector_core_width;

    /* Vector config*/
    uint32_t process_bit;

    cycle_type layernorm_latency;
    cycle_type softmax_latency;
    cycle_type add_latency;
    cycle_type mul_latency;
    cycle_type exp_latency;
    cycle_type gelu_latency;
    cycle_type add_tree_latency;
    cycle_type scalar_sqrt_latency;
    cycle_type scalar_add_latency;
    cycle_type scalar_mul_latency;

    /* SRAM config */
    uint32_t sram_width;
    uint32_t sram_size;
    uint32_t spad_size;
    uint32_t accum_spad_size;

    /* DRAM config */
    DramType dram_type;
    uint32_t dram_freq;
    uint32_t dram_channels;
    uint32_t dram_req_size;

    /* PIM config */
    std::string pim_config_path;
    uint32_t dram_page_size;  // DRAM row buffer size (in bytes)
    uint32_t dram_banks_per_ch;
    uint32_t pim_comp_coverage;  // # params per PIM_COMP command

    /* Log config */
    std::string operation_log_output_path;
    std::string log_dir;

    /* Client config */
    uint32_t request_input_seq_len;
    uint32_t request_interval;
    uint32_t request_total_cnt;
    std::string request_dataset_path;

    /* ICNT config */
    IcntType icnt_type;
    std::string icnt_config_path;
    uint32_t icnt_freq;
    uint32_t icnt_latency;

    /* Sheduler config */
    std::string scheduler_type;

    /* Other configs */
    uint32_t precision;
    std::string layout;

    uint64_t align_address(uint64_t addr) { return addr - (addr % dram_req_size); }
};

namespace Config {
extern SimulationConfig global_config;
}