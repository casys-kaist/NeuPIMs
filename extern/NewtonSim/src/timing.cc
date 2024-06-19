#include "timing.h"

#include <algorithm>
#include <utility>

namespace dramsim3 {

Timing::Timing(const Config &config)
    : same_bank(static_cast<int>(CommandType::SIZE)),
      other_banks_same_bankgroup(static_cast<int>(CommandType::SIZE)),
      other_bankgroups_same_rank(static_cast<int>(CommandType::SIZE)),
      other_ranks(static_cast<int>(CommandType::SIZE)),
      same_rank(static_cast<int>(CommandType::SIZE)) {
    int read_to_read_l = std::max(config.burst_cycle, config.tCCD_L);
    int read_to_read_s = std::max(config.burst_cycle, config.tCCD_S);
    int read_to_read_o = config.burst_cycle + config.tRTRS;
    int read_to_write = config.RL + config.burst_cycle - config.WL + config.tRTRS;
    int read_to_write_o =
        config.read_delay + config.burst_cycle + config.tRTRS - config.write_delay;
    int read_to_precharge = config.AL + config.tRTP;
    int readp_to_act = config.AL + config.burst_cycle + config.tRTP + config.tRP;

    int write_to_read_l = config.write_delay + config.tWTR_L;
    int write_to_read_s = config.write_delay + config.tWTR_S;
    int write_to_read_o =
        config.write_delay + config.burst_cycle + config.tRTRS - config.read_delay;
    int write_to_write_l = std::max(config.burst_cycle, config.tCCD_L);
    int write_to_write_s = std::max(config.burst_cycle, config.tCCD_S);
    int write_to_write_o = config.burst_cycle;
    int write_to_precharge = config.WL + config.burst_cycle + config.tWR;

    int precharge_to_activate = config.tRP;
    int precharge_to_precharge = config.tPPD;
    int read_to_activate = read_to_precharge + precharge_to_activate;
    int write_to_activate = write_to_precharge + precharge_to_activate;

    int activate_to_activate = config.tRC;
    int activate_to_activate_l = config.tRRD_L;
    int activate_to_activate_s = config.tRRD_S;
    int activate_to_precharge = config.tRAS;
    int activate_to_read, activate_to_write;
    if (config.IsGDDR() || config.IsHBM()) {
        activate_to_read = config.tRCDRD;
        activate_to_write = config.tRCDWR;
    } else {
        activate_to_read = config.tRCD - config.AL;
        activate_to_write = config.tRCD - config.AL;
    }
    int activate_to_refresh = config.tRC; // need to precharge before ref, so it's tRC

    // todo: deal with different refresh rate
    int refresh_to_refresh = config.tREFI; // refresh intervals (per rank level)
    int refresh_to_activate = config.tRFC; // tRFC is defined as ref to act
    int refresh_to_activate_bank = config.tRFCb;

    int self_refresh_entry_to_exit = config.tCKESR;
    int self_refresh_exit = config.tXS;
    // int powerdown_to_exit = config.tCKE;
    // int powerdown_exit = config.tXP;

    if (config.bankgroups == 1) {
        // for a bankgroup can be disabled, in that case
        // the value of tXXX_S should be used instead of tXXX_L
        // (because now the device is running at a lower freq)
        // we overwrite the following values so that we don't have
        // to change the assignement of the vectors
        read_to_read_l = std::max(config.burst_cycle, config.tCCD_S);
        write_to_read_l = config.write_delay + config.tWTR_S;
        write_to_write_l = std::max(config.burst_cycle, config.tCCD_S);
        activate_to_activate_l = config.tRRD_S;
    }

    // same bank
    int gwrite_latency = config.gwrite_delay;
    int pim_act_to_act_same_bk = activate_to_activate;
    int act_to_pim_act_same_bk = activate_to_activate;
    int read_to_pim_read_same_bk = read_to_read_l;
    int write_to_pim_read_same_bk = write_to_read_l;

    // other banks in same bankgroup
    int read_to_pim_read_same_bg = read_to_read_l;
    int write_to_pim_read_same_bg = write_to_read_l;
    int gact_to_act_same_bg = activate_to_activate;

    // same rank
    int pim_to_precharge = read_to_precharge;

    if (config.enable_dual_buffer) {
        // same bank
        pim_act_to_act_same_bk = activate_to_activate_l;
        act_to_pim_act_same_bk = activate_to_activate_l;
        read_to_pim_read_same_bk = 0;  // read_to_read_s;
        write_to_pim_read_same_bk = 0; // write_to_read_s;

        // same bankgroup
        read_to_pim_read_same_bg = 0;  // read_to_read_s;
        write_to_pim_read_same_bg = 0; // write_to_read_s;
        gact_to_act_same_bg = activate_to_activate_l;

        // same rank
        pim_to_precharge = 0;
    }

    // command READ
    same_bank[static_cast<int>(CommandType::READ)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_l},
        {CommandType::WRITE, read_to_write},
        {CommandType::READ_PRECHARGE, read_to_read_l},
        {CommandType::WRITE_PRECHARGE, read_to_write},
        {CommandType::PRECHARGE, read_to_precharge},
        {CommandType::GWRITE, read_to_read_l},
        {CommandType::COMP, read_to_pim_read_same_bk},
        {CommandType::COMPS_READRES, read_to_pim_read_same_bk},
        {CommandType::READRES, read_to_pim_read_same_bk},
        {CommandType::PWRITE, read_to_write},
    };
    other_banks_same_bankgroup[static_cast<int>(CommandType::READ)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_l},
            {CommandType::WRITE, read_to_write},
            {CommandType::READ_PRECHARGE, read_to_read_l},
            {CommandType::WRITE_PRECHARGE, read_to_write},
            {CommandType::GWRITE, read_to_read_l},
            {CommandType::COMP, read_to_pim_read_same_bg},
            {CommandType::COMPS_READRES, read_to_pim_read_same_bg},
            {CommandType::READRES, read_to_pim_read_same_bg},
        };
    other_bankgroups_same_rank[static_cast<int>(CommandType::READ)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_s},
            {CommandType::WRITE, read_to_write},
            {CommandType::READ_PRECHARGE, read_to_read_s},
            {CommandType::WRITE_PRECHARGE, read_to_write},
            {CommandType::GWRITE, read_to_read_s},
            {CommandType::COMP, read_to_read_s},
            {CommandType::COMPS_READRES, read_to_read_s},
            {CommandType::READRES, read_to_read_s},
        };
    other_ranks[static_cast<int>(CommandType::READ)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_o},
        {CommandType::WRITE, read_to_write_o},
        {CommandType::READ_PRECHARGE, read_to_read_o},
        {CommandType::WRITE_PRECHARGE, read_to_write_o},
        {CommandType::READRES, read_to_read_o},
    };

    // command WRITE
    same_bank[static_cast<int>(CommandType::WRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, write_to_read_l},
        {CommandType::WRITE, write_to_write_l},
        {CommandType::READ_PRECHARGE, write_to_read_l},
        {CommandType::WRITE_PRECHARGE, write_to_write_l},
        {CommandType::PRECHARGE, write_to_precharge},
        {CommandType::GWRITE, write_to_read_l},
        {CommandType::PWRITE, write_to_write_l},
        {CommandType::COMP, write_to_pim_read_same_bk},
        {CommandType::COMPS_READRES, write_to_pim_read_same_bk},
        {CommandType::READRES, write_to_pim_read_same_bk},
    };
    other_banks_same_bankgroup[static_cast<int>(CommandType::WRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, write_to_read_l},
            {CommandType::WRITE, write_to_write_l},
            {CommandType::READ_PRECHARGE, write_to_read_l},
            {CommandType::WRITE_PRECHARGE, write_to_write_l},
            {CommandType::GWRITE, write_to_read_l},
            {CommandType::PWRITE, write_to_write_l},
            {CommandType::COMP, write_to_pim_read_same_bg},
            {CommandType::COMPS_READRES, write_to_pim_read_same_bg},
            {CommandType::READRES, write_to_pim_read_same_bg},
        };
    other_bankgroups_same_rank[static_cast<int>(CommandType::WRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, write_to_read_s},
            {CommandType::WRITE, write_to_write_s},
            {CommandType::READ_PRECHARGE, write_to_read_s},
            {CommandType::WRITE_PRECHARGE, write_to_write_s},
            {CommandType::GWRITE, write_to_read_s},
            {CommandType::COMP, write_to_read_s},
            {CommandType::COMPS_READRES, write_to_read_s},
            {CommandType::READRES, write_to_read_s},
            {CommandType::PWRITE, write_to_write_s},
        };
    other_ranks[static_cast<int>(CommandType::WRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, write_to_read_o},
        {CommandType::WRITE, write_to_write_o},
        {CommandType::READ_PRECHARGE, write_to_read_o},
        {CommandType::WRITE_PRECHARGE, write_to_write_o},
        {CommandType::READRES, write_to_read_o},
        {CommandType::PWRITE, write_to_write_o},
    };

    // command READ_PRECHARGE
    same_bank[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::ACTIVATE, readp_to_act},
                                                 {CommandType::REFRESH, read_to_activate},
                                                 {CommandType::REFRESH_BANK, read_to_activate},
                                                 {CommandType::SREF_ENTER, read_to_activate}};
    other_banks_same_bankgroup[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, read_to_read_l},
                                                 {CommandType::WRITE, read_to_write},
                                                 {CommandType::READ_PRECHARGE, read_to_read_l},
                                                 {CommandType::WRITE_PRECHARGE, read_to_write}};
    other_bankgroups_same_rank[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, read_to_read_s},
                                                 {CommandType::WRITE, read_to_write},
                                                 {CommandType::READ_PRECHARGE, read_to_read_s},
                                                 {CommandType::WRITE_PRECHARGE, read_to_write}};
    other_ranks[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, read_to_read_o},
                                                 {CommandType::WRITE, read_to_write_o},
                                                 {CommandType::READ_PRECHARGE, read_to_read_o},
                                                 {CommandType::WRITE_PRECHARGE, read_to_write_o}};

    // command WRITE_PRECHARGE
    same_bank[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::ACTIVATE, write_to_activate},
                                                 {CommandType::REFRESH, write_to_activate},
                                                 {CommandType::REFRESH_BANK, write_to_activate},
                                                 {CommandType::SREF_ENTER, write_to_activate}};
    other_banks_same_bankgroup[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, write_to_read_l},
                                                 {CommandType::WRITE, write_to_write_l},
                                                 {CommandType::READ_PRECHARGE, write_to_read_l},
                                                 {CommandType::WRITE_PRECHARGE, write_to_write_l}};
    other_bankgroups_same_rank[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, write_to_read_s},
                                                 {CommandType::WRITE, write_to_write_s},
                                                 {CommandType::READ_PRECHARGE, write_to_read_s},
                                                 {CommandType::WRITE_PRECHARGE, write_to_write_s}};
    other_ranks[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, write_to_read_o},
                                                 {CommandType::WRITE, write_to_write_o},
                                                 {CommandType::READ_PRECHARGE, write_to_read_o},
                                                 {CommandType::WRITE_PRECHARGE, write_to_write_o}};

    // command ACTIVATE
    same_bank[static_cast<int>(CommandType::ACTIVATE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::ACTIVATE, activate_to_activate},
                                                 {CommandType::READ, activate_to_read},
                                                 {CommandType::WRITE, activate_to_write},
                                                 {CommandType::READ_PRECHARGE, activate_to_read},
                                                 {CommandType::WRITE_PRECHARGE, activate_to_write},
                                                 {CommandType::PRECHARGE, activate_to_precharge},
                                                 {CommandType::G_ACT, act_to_pim_act_same_bk},
                                                 {CommandType::GWRITE, activate_to_read}};

    other_banks_same_bankgroup[static_cast<int>(CommandType::ACTIVATE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, activate_to_activate_l},
            {CommandType::REFRESH_BANK, activate_to_refresh},
            {CommandType::G_ACT, activate_to_activate_l},
        };

    other_bankgroups_same_rank[static_cast<int>(CommandType::ACTIVATE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, activate_to_activate_s},
            {CommandType::REFRESH_BANK, activate_to_refresh},
            {CommandType::G_ACT, activate_to_activate_s},
        };

    // command PRECHARGE
    same_bank[static_cast<int>(CommandType::PRECHARGE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, precharge_to_activate},
        {CommandType::REFRESH, precharge_to_activate},
        {CommandType::REFRESH_BANK, precharge_to_activate},
        {CommandType::SREF_ENTER, precharge_to_activate},
    };

    // for those who need tPPD
    if (config.IsGDDR() || config.protocol == DRAMProtocol::LPDDR4) {
        other_banks_same_bankgroup[static_cast<int>(CommandType::PRECHARGE)] =
            std::vector<std::pair<CommandType, int>>{
                {CommandType::PRECHARGE, precharge_to_precharge},
            };

        other_bankgroups_same_rank[static_cast<int>(CommandType::PRECHARGE)] =
            std::vector<std::pair<CommandType, int>>{
                {CommandType::PRECHARGE, precharge_to_precharge},
            };
    }

    // command REFRESH_BANK
    same_rank[static_cast<int>(CommandType::REFRESH_BANK)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, refresh_to_activate_bank},
            {CommandType::REFRESH, refresh_to_activate_bank},
            {CommandType::REFRESH_BANK, refresh_to_activate_bank},
            {CommandType::SREF_ENTER, refresh_to_activate_bank}};

    other_banks_same_bankgroup[static_cast<int>(CommandType::REFRESH_BANK)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, refresh_to_activate},
            {CommandType::REFRESH_BANK, refresh_to_refresh},
        };

    other_bankgroups_same_rank[static_cast<int>(CommandType::REFRESH_BANK)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, refresh_to_activate},
            {CommandType::REFRESH_BANK, refresh_to_refresh},
        };

    // REFRESH, SREF_ENTER and SREF_EXIT are isued to the entire
    // rank  command REFRESH
    same_rank[static_cast<int>(CommandType::REFRESH)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, refresh_to_activate},
        {CommandType::REFRESH, refresh_to_activate},
        {CommandType::SREF_ENTER, refresh_to_activate},
        {CommandType::G_ACT, refresh_to_activate},
    };

    // command SREF_ENTER
    // todo: add power down commands
    same_rank[static_cast<int>(CommandType::SREF_ENTER)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::SREF_EXIT, self_refresh_entry_to_exit}};

    // command SREF_EXIT
    same_rank[static_cast<int>(CommandType::SREF_EXIT)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, self_refresh_exit},
        {CommandType::REFRESH, self_refresh_exit},
        {CommandType::REFRESH_BANK, self_refresh_exit},
        {CommandType::SREF_ENTER, self_refresh_exit},
        {CommandType::G_ACT, self_refresh_exit} // >>> gsheo
    };

    // command PIM_PRECHARGE
    same_bank[static_cast<int>(CommandType::PIM_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::REFRESH, precharge_to_activate},
            {CommandType::G_ACT, precharge_to_activate},
        };

    // command PWRITE
    same_bank[static_cast<int>(CommandType::PWRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, write_to_read_l},         {CommandType::WRITE, write_to_write_l},
        {CommandType::PRECHARGE, write_to_precharge}, {CommandType::PWRITE, write_to_write_l},
        {CommandType::COMP, config.tWTR_L},           {CommandType::COMPS_READRES, config.tWTR_L},
    };
    other_banks_same_bankgroup[static_cast<int>(CommandType::PWRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, write_to_read_l},
            {CommandType::WRITE, write_to_write_l},
            {CommandType::PWRITE, write_to_write_l},
            {CommandType::COMP, write_to_pim_read_same_bg},
            {CommandType::COMPS_READRES, write_to_pim_read_same_bg},
        };
    other_bankgroups_same_rank[static_cast<int>(CommandType::PWRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, write_to_read_s},          {CommandType::WRITE, write_to_write_s},
            {CommandType::PWRITE, write_to_write_s},       {CommandType::COMP, write_to_read_s},
            {CommandType::COMPS_READRES, write_to_read_s},
        };
    other_ranks[static_cast<int>(CommandType::PWRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, write_to_read_o},
        {CommandType::WRITE, write_to_write_o},
        {CommandType::READRES, write_to_read_o},
        {CommandType::PWRITE, write_to_write_o},
    };

    // command GWRITE
    same_bank[static_cast<int>(CommandType::GWRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, gwrite_latency},
        {CommandType::WRITE, gwrite_latency},
        {CommandType::PRECHARGE, gwrite_latency},
        {CommandType::COMP, gwrite_latency},          // for double buffer
        {CommandType::COMPS_READRES, gwrite_latency}, // for double buffer
    };

    other_banks_same_bankgroup[static_cast<int>(CommandType::GWRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_l},          {CommandType::WRITE, read_to_write},
            {CommandType::GWRITE, gwrite_latency},        {CommandType::COMP, gwrite_latency},
            {CommandType::COMPS_READRES, gwrite_latency}, {CommandType::READRES, gwrite_latency},
        };

    other_bankgroups_same_rank[static_cast<int>(CommandType::GWRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_s},          {CommandType::WRITE, read_to_write},
            {CommandType::GWRITE, gwrite_latency},        {CommandType::COMP, gwrite_latency},
            {CommandType::COMPS_READRES, gwrite_latency}, {CommandType::READRES, gwrite_latency},
        };

    // command G_ACT
    // actually same_bankgroup
    other_banks_same_bankgroup[static_cast<int>(CommandType::G_ACT)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, gact_to_act_same_bg},
            {CommandType::G_ACT, activate_to_activate},
            {CommandType::COMP, activate_to_read},
            {CommandType::COMPS_READRES, activate_to_read},
            {CommandType::READRES, activate_to_read},
            {CommandType::PIM_PRECHARGE, activate_to_precharge},
            {CommandType::PWRITE, activate_to_write},
        };

    same_rank[static_cast<int>(CommandType::G_ACT)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, config.tFAW},
        {CommandType::G_ACT, config.tFAW},
        {CommandType::COMP, config.tRCDRD},
        {CommandType::COMPS_READRES, config.tRCDRD},
    };

    // command COMP
    same_rank[static_cast<int>(CommandType::COMP)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_s},        {CommandType::WRITE, read_to_write},
        {CommandType::GWRITE, read_to_read_s},      {CommandType::COMP, read_to_read_s},
        {CommandType::READRES, read_to_read_s},     {CommandType::PIM_PRECHARGE, read_to_precharge},
        {CommandType::PRECHARGE, pim_to_precharge},
    };

    // command READRES
    same_rank[static_cast<int>(CommandType::READRES)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_s},
        {CommandType::WRITE, read_to_write},
        {CommandType::GWRITE, read_to_read_s},
        {CommandType::COMP, read_to_read_s},
        {CommandType::READRES, read_to_read_s},
        {CommandType::G_ACT, read_to_activate},
        {CommandType::PIM_PRECHARGE, read_to_precharge},
        {CommandType::PRECHARGE, pim_to_precharge},
    };

    other_ranks[static_cast<int>(CommandType::READRES)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_o},
        {CommandType::WRITE, read_to_write_o},
    };

    // command COMPS_READRES
    same_rank[static_cast<int>(CommandType::COMPS_READRES)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_o},
            {CommandType::WRITE, read_to_write_o},
            {CommandType::GWRITE, read_to_read_s},
            {CommandType::G_ACT, read_to_activate}, // maybe not used
            {CommandType::PIM_PRECHARGE, read_to_precharge},
            {CommandType::PRECHARGE, pim_to_precharge},
            {CommandType::COMP, read_to_read_s},    // deprecated
            {CommandType::READRES, read_to_read_s}, // deprecated
            // timing for COMPS_READRES is decided by num_comps of corresponding command
        };
}

} // namespace dramsim3
