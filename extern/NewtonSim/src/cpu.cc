#include "cpu.h"
#include "common.h"

namespace dramsim3 {

// TODO: Make this function in common.h
uint64_t CPU::MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col) {
    // rorabgbachco (HBM2_8Gb_s128_pim.ini)
    uint64_t addr = 0;

    int row_bits = 15;
    int rank_bits = 1;
    int bankgroup_bits = 2;
    int bank_bits = 2;
    int channel_bits = 3;
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
    addr |= col;

    addr <<= offset;

    // uint64_t address =
    //     memory_system_.config_->MakeAddress(channel, rank, bankgroup, bank, row, col);
    // if (address == addr)
    //     PrintError("SUCCESS");
    // else
    //     PrintError("FAIL");

    return addr;
}
uint64_t CPU::MakePHeaderPacket(int channel, int row, bool for_gwrite, int num_comps,
                                int num_readres) {
    int gwrite_bit = for_gwrite ? 1 : 0;

    // we can have only 4 bits for column bit, so use as shift_amount
    int log_comps = (gwrite_bit << 3) + LogBase2(num_comps);
    int log_readres = LogBase2(num_readres);

    return MakeAddress(channel, log_readres / 16, (log_readres / 4) & 3, log_readres % 4, row,
                       log_comps);
}

void RandomCPU::ClockTick() {
    // Create random CPU requests at full speed
    // this is useful to exploit the parallelism of a DRAM protocol
    // and is also immune to address mapping and scheduling policies
    memory_system_.ClockTick();
    if (get_next_) {
        last_addr_ = gen();
        last_write_ = (gen() % 3 == 0);
    }

    int row = 0;
    int channel = 0;
    // uint64_t addr = MakeAddress(0, 0, 0, 0, row, 0);

    TransactionType req_type = TransactionType::P_HEADER;
    if (clk_ == 30) {
        num_comps_ = 1;
        num_readres_ = 1;

        bool for_gwrite = false;


        uint64_t p_header_addr =
            MakePHeaderPacket(channel, row, for_gwrite, num_comps_, num_readres_);
        uint64_t weired_addr = MakeAddress(channel, 0, 0, 0, row + 1, 0);

        get_next_ = memory_system_.WillAcceptTransaction(p_header_addr, req_type);
        if (get_next_) {
            memory_system_.AddTransaction(p_header_addr, TransactionType::WRITE);
            memory_system_.AddTransaction(p_header_addr + 256, TransactionType::WRITE);
            memory_system_.AddTransaction(p_header_addr + 128, TransactionType::WRITE);
            memory_system_.AddTransaction(weired_addr, TransactionType::WRITE);
            memory_system_.AddTransaction(p_header_addr, req_type);
            memory_system_.AddTransaction(p_header_addr, TransactionType::COMP);
        }
        // get_next_ = memory_system_.WillAcceptTransaction(addr, TransactionType::COMP);
        // if (get_next_) {
        //     memory_system_.AddTransaction(addr, TransactionType::COMP);
        //     num_comps_--;
        //     loop_cnt_--;
        //     gemv_turn_ = true;
        // }
        gemv_turn_ = false;
    }


    clk_++;

    return;
}

void StreamCPU::ClockTick() {
    // stream-add, read 2 arrays, add them up to the third array
    // this is a very simple approximate but should be able to produce
    // enough buffer hits

    // moving on to next set of arrays
    memory_system_.ClockTick();
    if (offset_ >= array_size_ || clk_ == 0) {
        addr_a_ = gen();
        addr_b_ = gen();
        addr_c_ = gen();
        offset_ = 0;
    }

    if (!inserted_a_ &&
        memory_system_.WillAcceptTransaction(addr_a_ + offset_, TransactionType::READ)) {
        memory_system_.AddTransaction(addr_a_ + offset_, TransactionType::READ);
        inserted_a_ = true;
    }
    if (!inserted_b_ &&
        memory_system_.WillAcceptTransaction(addr_b_ + offset_, TransactionType::READ)) {
        memory_system_.AddTransaction(addr_b_ + offset_, TransactionType::READ);
        inserted_b_ = true;
    }
    if (!inserted_c_ &&
        memory_system_.WillAcceptTransaction(addr_c_ + offset_, TransactionType::WRITE)) {
        memory_system_.AddTransaction(addr_c_ + offset_, TransactionType::WRITE);
        inserted_c_ = true;
    }
    // moving on to next element
    if (inserted_a_ && inserted_b_ && inserted_c_) {
        offset_ += stride_;
        inserted_a_ = false;
        inserted_b_ = false;
        inserted_c_ = false;
    }
    clk_++;
    return;
}

TraceBasedCPU::TraceBasedCPU(const std::string &config_file, const std::string &output_dir,
                             const std::string &trace_file)
    : CPU(config_file, output_dir) {
    trace_file_.open(trace_file);
    if (trace_file_.fail()) {
        std::cerr << "Trace file does not exist" << std::endl;
        AbruptExit(__FILE__, __LINE__);
    }
}

void TraceBasedCPU::ClockTick() {
    memory_system_.ClockTick();
    if (!trace_file_.eof()) {
        if (get_next_) {
            get_next_ = false;
            trace_file_ >> trans_;
        }
        if (trans_.added_cycle <= clk_) {
            get_next_ = memory_system_.WillAcceptTransaction(trans_.addr, trans_.req_type);
            if (get_next_) {
                memory_system_.AddTransaction(trans_.addr, trans_.req_type);
                processed_++;
            }
        }
    }
    clk_++;
    return;
}

} // namespace dramsim3
