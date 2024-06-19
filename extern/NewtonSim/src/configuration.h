#ifndef __CONFIG_H
#define __CONFIG_H

#include <fstream>
#include <string>

#include "INIReader.h"
#include "common.h"

namespace dramsim3 {

enum class DRAMProtocol {
    DDR3,
    DDR4,
    GDDR5,
    GDDR5X,
    GDDR6,
    LPDDR,
    LPDDR3,
    LPDDR4,
    HBM,
    HBM2, // gsheo: remove HMC
    SIZE
};

enum class MemoryType { DRAM, NEWTON, NEUPIMS };

enum class RefreshPolicy {
    RANK_LEVEL_SIMULTANEOUS, // impractical due to high power requirement
    RANK_LEVEL_STAGGERED,
    BANK_LEVEL_STAGGERED,
    SIZE
};

class Config {
  public:
    Config(std::string config_file, std::string out_dir);
    Address AddressMapping(uint64_t hex_addr) const;
    uint64_t MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col);
    uint64_t EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps, int num_readres);
    // DRAM physical structure
    DRAMProtocol protocol;
    MemoryType memory_type;
    int channel_size;
    int channels;
    int ranks;
    int banks;
    int bankgroups;
    int banks_per_group;
    int rows;
    int columns;
    int device_width;
    int bus_width;
    int devices_per_rank;
    int BL;

    // Address mapping numbers
    int shift_bits;
    int ch_pos, ra_pos, bg_pos, ba_pos, ro_pos, co_pos;
    uint64_t ch_mask, ra_mask, bg_mask, ba_mask, ro_mask, co_mask;

    // Generic DRAM timing parameters
    double tCK;
    int burst_cycle; // seperate BL with timing since for GDDRx it's not BL/2
    int AL;
    int CL;
    int CWL;
    int RL;
    int WL;
    int tCCD_L; // column-to-column delay
    int tCCD_S;
    int tRTRS;  // rank-to-rank switching time
    int tRTP;   // read to precharge
    int tWTR_L; // write-to-read delay time
    int tWTR_S;
    int tWR;    // write recovery time
    int tRP;    // row precharge
    int tRRD_L; // row_act-to-row_act delay
    int tRRD_S;
    int tRAS; // row access strobe
    int tRCD; // row-to-column command delay
    int tRFC;
    int tRC; // row cycle
    // tCKSRE and tCKSRX are only useful for changing clock freq after entering
    // SRE mode we are not doing that, so tCKESR is sufficient
    int tCKE;
    int tCKESR;
    int tXS;
    int tXP;
    int tRFCb;
    int tREFI;
    int tREFIb;
    int tFAW;  // four-bank activation window
    int tRPRE; // read preamble and write preamble are important
    int tWPRE;
    int read_delay;
    int write_delay;
    int gwrite_delay; // for GWRITE command

    // LPDDR4 and GDDR5
    int tPPD;
    // GDDR5
    int t32AW;
    int tRCDRD;
    int tRCDWR;

    // pre calculated power parameters
    double act_energy_inc;
    double pre_energy_inc;
    double read_energy_inc;
    double write_energy_inc;
    double ref_energy_inc;
    double refb_energy_inc;
    double act_stb_energy_inc;
    double pre_stb_energy_inc;
    double pre_pd_energy_inc;
    double sref_energy_inc;

    // >>> gsheo: power parameters related to PIM
    double gwrite_energy_inc;
    double gact_energy_inc;
    double comp_energy_inc;
    double readres_energy_inc;
    // pim power parameters for background
    double pim_act_stb_energy_inc;
    double pim_pre_stb_energy_inc;

    // HMC
    int num_links;
    int num_dies;
    int link_width;
    int link_speed;
    int num_vaults;
    int block_size; // block size in bytes
    int xbar_queue_depth;

    // System
    std::string address_mapping;
    std::string queue_structure;
    std::string row_buf_policy;
    RefreshPolicy refresh_policy;
    int cmd_queue_size;
    bool unified_queue;
    int trans_queue_size;
    int write_buf_size;
    bool enable_self_refresh;
    int sref_threshold;
    bool aggressive_precharging_enabled;
    bool enable_hbm_dual_cmd;
    bool enable_dual_buffer;

    int epoch_period;
    int output_level;
    std::string output_dir;
    std::string output_prefix;
    std::string json_stats_name;
    std::string json_epoch_name;
    std::string txt_stats_name;

    // Computed parameters
    int request_size_bytes;

    bool IsGDDR() const {
        return (protocol == DRAMProtocol::GDDR5 || protocol == DRAMProtocol::GDDR5X ||
                protocol == DRAMProtocol::GDDR6);
    }
    bool IsHBM() const { return (protocol == DRAMProtocol::HBM || protocol == DRAMProtocol::HBM2); }
    bool IsHMC() const { return false; } // gsheo: remove HMC support
    // yzy: add another function
    bool IsDDR4() const { return (protocol == DRAMProtocol::DDR4); }

    int ideal_memory_latency;

  private:
    INIReader *reader_;
    void CalculateSize();
    DRAMProtocol GetDRAMProtocol(std::string protocol_str);
    MemoryType GetMemoryType(std::string mem_type);
    int GetInteger(const std::string &sec, const std::string &opt, int default_val) const;
    void InitDRAMParams();
    void InitOtherParams();
    void InitPowerParams();
    void InitSystemParams();
    void InitTimingParams();
    void SetAddressMapping();
};

} // namespace dramsim3
#endif
