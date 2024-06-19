#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "SimulationConfig.h"

// TODO: num_cycles is a magic number. it counts memory load store for 50 core-cycles
// TODO: convert global cycle to core cycle
//  global time / core freq -> core cycle
// memory_reads, memory_writes: bytes
typedef struct NPUStat {
    NPUStat() = default;
    NPUStat(uint64_t core_cycle_)
        : start_cycle(core_cycle_), num_calculations(0) {}

    uint64_t start_cycle;
    uint64_t num_cycles;
    uint64_t num_calculations;

    enum class StatType {
        StartCycle,
        NumCalculations,
    };

    static std::vector<StatType> get_stat_types() {
        return {
            StatType::StartCycle,
            StatType::NumCalculations,
        };
    }

    static std::string enum_to_string(StatType stat_type) {
        switch (stat_type) {
            case StatType::StartCycle:
                return "StartCycle";
            case StatType::NumCalculations:
                return "NumCalculations";
            default:
                assert(0);
        }
    }

    std::string get_by_enum(StatType stat_type) {
        uint64_t core_cycle = Config::global_config.core_freq * 1000000;  // Mhz

        switch (stat_type) {
            case StatType::StartCycle:
                return std::to_string(start_cycle);
            case StatType::NumCalculations:
                return std::to_string(num_calculations);
            default:
                assert(0);
        }
    }

    static std::string get_columns() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += enum_to_string(type) + "\t";
        }
        return ret + "\n";
    }

    std::string repr() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += get_by_enum(type) + "\t";
        }
        return ret + "\n";
    }
} NPUStat;

// TODO: num_cycles is a magic number. it counts memory load store for 50 core-cycles
// TODO: convert global cycle to core cycle
//  global time / core freq -> core cycle
// memory_reads, memory_writes: bytes
typedef struct MemoryIOStat {
    MemoryIOStat() = default;
    MemoryIOStat(uint64_t core_cycle_, uint64_t channel_id_, uint64_t num_cycles_)
        : start_cycle(core_cycle_),
          channel_id(channel_id_),
          num_cycles(num_cycles_),
          memory_reads(0),
          memory_writes(0),
          pim_reads(0) {}

    uint64_t start_cycle;
    uint64_t channel_id;
    uint64_t num_cycles;
    uint64_t memory_reads;
    uint64_t memory_writes;
    uint64_t pim_reads;

    enum class StatType {
        StartCycle,
        ChannelID,
        MemoryReads,
        MemoryWrites,
        MemoryReadBandwidth,
        MemoryWriteBandwidth,
        PIMReads,
        PIMReadBandwidth,
    };

    static std::vector<StatType> get_stat_types() {
        return {
            StatType::StartCycle,   StatType::ChannelID,           StatType::MemoryReads,
            StatType::MemoryWrites, StatType::MemoryReadBandwidth, StatType::MemoryWriteBandwidth,
            StatType::PIMReads,     StatType::PIMReadBandwidth,
        };
    }

    static std::string enum_to_string(StatType stat_type) {
        switch (stat_type) {
            case StatType::StartCycle:
                return "StartCycle";
            case StatType::ChannelID:
                return "ChannelID";
            case StatType::MemoryReads:
                return "MemoryReads";
            case StatType::MemoryWrites:
                return "MemoryWrites";
            case StatType::MemoryReadBandwidth:
                return "MemoryReadBandwidth";
            case StatType::MemoryWriteBandwidth:
                return "MemoryWriteBandwidth";
            case StatType::PIMReads:
                return "PIMReads";
            case StatType::PIMReadBandwidth:
                return "PIMReadBandwidth";
            default:
                assert(0);
        }
    }

    std::string get_by_enum(StatType stat_type) {
        uint64_t core_cycle = Config::global_config.core_freq * 1000000;  // Mhz

        switch (stat_type) {
            case StatType::StartCycle:
                return std::to_string(start_cycle);
            case StatType::ChannelID:
                return std::to_string(channel_id);
            case StatType::MemoryReads:
                return std::to_string(memory_reads);
            case StatType::MemoryWrites:
                return std::to_string(memory_writes);
            case StatType::MemoryReadBandwidth:
                return std::to_string(memory_reads * core_cycle / num_cycles);
            case StatType::MemoryWriteBandwidth:
                return std::to_string(memory_writes * core_cycle / num_cycles);
            case StatType::PIMReads:
                return std::to_string(pim_reads);
            case StatType::PIMReadBandwidth:
                return std::to_string(pim_reads * core_cycle / num_cycles);
            default:
                assert(0);
        }
    }

    static std::string get_columns() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += enum_to_string(type) + "\t";
        }
        return ret + "\n";
    }

    std::string repr() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += get_by_enum(type) + "\t";
        }
        return ret + "\n";
    }
} MemoryIOStat;

typedef struct TileStat {
    TileStat() = default;
    TileStat(uint64_t core_cycle)
        : start_cycle(core_cycle),
          end_cycle(0),
          compute_cycles(0),
          weight_load_cycles(0),
          memory_stalls(0),
          systolic_memory_stalls(0),
          memory_reads(0),
          memory_writes(0),
          sram_reads(0),
          sram_writes(0),
          num_calculation(0) {}

    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t compute_cycles;
    uint64_t weight_load_cycles;
    uint64_t memory_stalls;
    // stall cycles between weight-resued GEMMs
    uint64_t systolic_memory_stalls;

    // unit: bytes
    uint64_t memory_reads;
    uint64_t memory_writes;

    // todo
    uint64_t sram_reads;
    uint64_t sram_writes;

    // todo: vector
    uint64_t num_calculation;

    uint64_t dependency_stall;
} TileStat;

typedef struct OperationStat {
    OperationStat() = default;
    OperationStat(std::string name)
        : op_name(name),
          start_cycle(-1),
          end_cycle(0),
          compute_cycles(0),
          weight_load_cycles(0),
          memory_stalls(0),
          systolic_memory_stalls(0),
          memory_reads(0),
          memory_writes(0),
          sram_reads(0),
          sram_writes(0),
          num_calculation(0) {}

    void update_stat(TileStat tile_stat) {
        std::cout << "tile stat: " << repr() << std::endl;
        if (tile_stat.start_cycle < start_cycle) start_cycle = tile_stat.start_cycle;
        end_cycle = end_cycle > tile_stat.end_cycle ? end_cycle : tile_stat.end_cycle;
        compute_cycles += tile_stat.compute_cycles;
        memory_stalls += tile_stat.memory_stalls;

        memory_reads += tile_stat.memory_reads;
        memory_writes += tile_stat.memory_writes;

        num_calculation += tile_stat.num_calculation;
    }

    enum class StatType {
        OpName,
        StartCycle,
        EndCycle,
        TotalCycle,
        ComputeCycles,
        WeightLoadCycles,
        MemoryStalls,
        SystolicMemoryStalls,
        MemoryReads,
        MemoryWrites,
        ReadBandwidth,
        WriteBandwidth,
        TotalMemoryBandwidth,
        NumCalculation,
        NpuUtilization,
    };

    static std::vector<StatType> get_stat_types() {
        return {
            StatType::OpName,
            StatType::StartCycle,
            StatType::EndCycle,
            StatType::TotalCycle,
            StatType::ComputeCycles,
            StatType::MemoryStalls,
            StatType::WeightLoadCycles,
            StatType::SystolicMemoryStalls,
            StatType::MemoryReads,
            StatType::MemoryWrites,
            StatType::ReadBandwidth,
            StatType::WriteBandwidth,
            StatType::TotalMemoryBandwidth,
            StatType::NumCalculation,
            StatType::NpuUtilization,
        };
    }

    static std::string enum_to_string(StatType stat_type) {
        switch (stat_type) {
            case StatType::OpName:
                return "OpName";
            case StatType::StartCycle:
                return "StartCycle";
            case StatType::EndCycle:
                return "EndCycle";
            case StatType::TotalCycle:
                return "TotalCycle";
            case StatType::ComputeCycles:
                return "ComputeCycles";
            case StatType::WeightLoadCycles:
                return "WeightLoadCycles";
            case StatType::MemoryStalls:
                return "MemoryStalls";
            case StatType::SystolicMemoryStalls:
                return "SystolicMemoryStalls";
            case StatType::MemoryReads:
                return "MemoryReads";
            case StatType::MemoryWrites:
                return "MemoryWrites";
            case StatType::ReadBandwidth:
                return "ReadBandwidth";
            case StatType::WriteBandwidth:
                return "WriteBandwidth";
            case StatType::TotalMemoryBandwidth:
                return "TotalMemoryBandwidth";
            case StatType::NumCalculation:
                return "NumCalculation";
            case StatType::NpuUtilization:
                return "NpuUtilization";
            default:
                assert(0);
        }
    }

    std::string get_by_enum(StatType stat_type) {
        double npu_util;
        uint64_t total_cycle = end_cycle - start_cycle;
        uint64_t core_cycle = Config::global_config.core_freq * 1000000;

        switch (stat_type) {
            case StatType::OpName:
                return op_name;
            case StatType::StartCycle:
                return std::to_string(start_cycle);
            case StatType::EndCycle:
                return std::to_string(end_cycle);
            case StatType::TotalCycle:
                return std::to_string(total_cycle);
            case StatType::ComputeCycles:
                return std::to_string(compute_cycles);
            case StatType::WeightLoadCycles:
                return std::to_string(weight_load_cycles);
            case StatType::MemoryStalls:
                return std::to_string(memory_stalls);
            case StatType::SystolicMemoryStalls:
                return std::to_string(systolic_memory_stalls);
            case StatType::MemoryReads:
                return std::to_string(memory_reads);
            case StatType::MemoryWrites:
                return std::to_string(memory_writes);
            case StatType::ReadBandwidth:
                return std::to_string(memory_reads * core_cycle / total_cycle);
            case StatType::WriteBandwidth:
                return std::to_string(memory_writes * core_cycle / total_cycle);
            case StatType::TotalMemoryBandwidth:
                return std::to_string((memory_reads + memory_writes) * core_cycle / total_cycle);
            case StatType::NumCalculation:
                return std::to_string(num_calculation);
            case StatType::NpuUtilization:
                npu_util = (double)num_calculation /
                           (double)(compute_cycles * Config::global_config.core_width *
                                    Config::global_config.core_height);
                return std::to_string(npu_util);
            default:
                assert(0);
        }
    }

    static std::string get_columns() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += enum_to_string(type) + "\t";
        }
        return ret + "\n";
    }

    std::string repr() {
        std::string ret = "";
        if (end_cycle == 0) {
            return ret;
        }

        for (auto type : get_stat_types()) {
            ret += get_by_enum(type) + "\t";
        }
        return ret + "\n";
    }

    std::string op_name;

    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t compute_cycles;
    uint64_t weight_load_cycles;
    uint64_t memory_stalls;
    uint64_t systolic_memory_stalls;

    uint64_t memory_reads;
    uint64_t memory_writes;

    // TODO
    uint64_t sram_reads;
    uint64_t sram_writes;

    // TODO: count num_calculation for vector operations
    uint64_t num_calculation;

    uint64_t dependency_stall;
} OperationStat;

// xxx not used
typedef struct {
    uint64_t op_cycles;
    std::vector<TileStat> tile_stats;
} OpStat;

// xxx not used
typedef struct {
    uint64_t total_cycles;
    std::vector<OpStat> op_stats;
} ModelStat;
