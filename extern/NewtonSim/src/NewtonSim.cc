#include "NewtonSim.h"

#include "common.h"
#include "configuration.h"
#include "dram_system.h"

namespace dramsim3 {
NewtonSim::NewtonSim(const std::string &config_file, const std::string &output_dir)
    : config_(new Config(config_file, output_dir)) {
    std::function<void(uint64_t)> read_callback = [&](uint64_t addr) {
        PrintInfo("(NewtonSim) read_callback");
        int channel = GetChannel(addr);
        assert(pending_read_q_.count(addr) > 0);
        if (pending_read_q_.count(addr) == 0)
            exit(1);

        auto it = pending_read_q_.find(addr);       // search
        response_queues_[channel].push(it->second); // push
        pending_read_q_.erase(it);                  // pop
    };
    std::function<void(uint64_t)> write_callback = [&](uint64_t addr) {
        PrintInfo("(NewtonSim) write_callback");
        int channel = GetChannel(addr);
        assert(pending_write_q_.count(addr) > 0); // assert does not work
        if (pending_write_q_.count(addr) == 0)
            exit(1);

        auto it = pending_write_q_.find(addr);      // search
        response_queues_[channel].push(it->second); // push
        pending_write_q_.erase(it);                 // pop
    };

    dram_system_ = new JedecDRAMSystem(*config_, output_dir, read_callback, write_callback);

    // printf("Newtonsim: # of channel= %d\n", config_->channels);
    // # channels = # reponse queues,
    // size of response queue = trans_queue_size
    int res_q_size;

    if (config_->memory_type == MemoryType::DRAM) {
        res_q_size = config_->trans_queue_size * 3;
    } else
        res_q_size = 4352; // original: config_->trans_queue_size
    // int res_q_size = config_->trans_queue_size * 3;
    for (int ch = 0; ch < config_->channels; ++ch) {
        response_queues_.push_back(ResponseQueue(res_q_size));
    }
    // for (int ch = 0; ch < config_->channels; ++ch) {
    //     response_queues_rd_.push_back(ResponseQueue(res_q_size));
    //     response_queues_wr_.push_back(ResponseQueue(res_q_size));
    //     response_queues_pim_.push_back(ResponseQueue(res_q_size));
    // }
}

uint64_t NewtonSim::GetAvgPIMCycles() { return dram_system_->GetAvgPIMCycles(); }
void NewtonSim::ResetPIMCycle() { dram_system_->ResetPIMCycle(); }

NewtonSim::~NewtonSim() {
    // std::cout << "NewtonSim delete" << std::endl;
    delete (dram_system_);
    delete (config_);
}

void NewtonSim::ClockTick() { dram_system_->ClockTick(); }

double NewtonSim::GetTCK() const { return config_->tCK; }

int NewtonSim::GetBusBits() const { return config_->bus_width; }

int NewtonSim::GetBurstLength() const { return config_->BL; }

int NewtonSim::GetQueueSize() const {
    exit(-1);
    // unused method
    return config_->trans_queue_size;
}

int NewtonSim::GetChannel(uint64_t hex_addr) const { return dram_system_->GetChannel(hex_addr); };

uint64_t NewtonSim::MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col) {
    return config_->MakeAddress(channel, rank, bankgroup, bank, row, col);
}
uint64_t NewtonSim::EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps,
                                    int num_readres) {
    return config_->EncodePIMHeader(channel, row, for_gwrite, num_comps, num_readres);
}

bool NewtonSim::WillAcceptTransaction(uint64_t hex_addr, int req_type) const {
    TransactionType type = static_cast<TransactionType>(req_type);
    // check response queue size
    int channel = GetChannel(hex_addr);
    bool available = response_queues_[channel].isAvailable(1);

    if (req_type == (int)TransactionType::P_HEADER)
        available = true;
    // std::cout << "NewtonSim::WillAcceptTransaction(" << channel << ") : "
    //           << std::to_string(
    //                  available &&
    //                  dram_system_->WillAcceptTransaction(hex_addr, type))
    //           << std::endl;
    return available && dram_system_->WillAcceptTransaction(hex_addr, type);
}

bool NewtonSim::AddTransaction(uint64_t hex_addr, int req_type, void *original_req) {
    TransactionType type = static_cast<TransactionType>(req_type);
    int channel = GetChannel(hex_addr);
    if (type != TransactionType::P_HEADER) {
        response_queues_[channel].reserve();
        PushToPendingQueue(hex_addr, type, original_req);
        income_req_cnt_++;
    }

    return dram_system_->AddTransaction(hex_addr, type);
}

void NewtonSim::PushToPendingQueue(uint64_t addr, TransactionType req_type, void *original_req) {
    switch (req_type) {
    case TransactionType::GWRITE:
    case TransactionType::COMP:
    case TransactionType::READRES:
    case TransactionType::READ:
    case TransactionType::COMPS_READRES:
        pending_read_q_.insert(std::make_pair(addr, original_req));
        break;
    case TransactionType::WRITE:
        pending_write_q_.insert(std::make_pair(addr, original_req));
        break;
    default: // P_HEADER
        break;
    }
    return;
}

bool NewtonSim::IsEmpty(uint32_t channel) const {
    bool empty = response_queues_[channel].isEmpty();
    // std::cout << "NewtonSim::IsEmpty(" << channel
    //           << ") : " << std::to_string(empty) << std::endl;
    return empty;
};
void *NewtonSim::Top(uint32_t channel) const {
    // printf("TOP channel= %d\n", channel);
    return response_queues_[channel].top();
};
void NewtonSim::Pop(uint32_t channel) {
    response_queues_[channel].pop();
    outcome_req_cnt_++;
};

void NewtonSim::PrintStats() const { dram_system_->PrintStats(); }

void NewtonSim::ResetStats() { dram_system_->ResetStats(); }

NewtonSim::ResponseQueue::ResponseQueue(int Size) : Size(Size), NumReserved(0) {}

bool NewtonSim::ResponseQueue::isAvailable() const {
    return NumReserved + OutputQueue.size() < Size;
}

bool NewtonSim::ResponseQueue::isAvailable(uint32_t count) const {
    return NumReserved + OutputQueue.size() + count - 1 < Size;
}

void NewtonSim::ResponseQueue::reserve() {
    assert(NumReserved < Size);
    NumReserved++;
}

void NewtonSim::ResponseQueue::push(void *original_req) {
    // std::cout << "ResponseQueue::push" << std::endl;
    OutputQueue.push_back(original_req);
    assert(NumReserved > 0);
    NumReserved--;
}

bool NewtonSim::ResponseQueue::isEmpty() const { return OutputQueue.empty(); }

void NewtonSim::ResponseQueue::pop() {
    auto it = OutputQueue.begin();
    OutputQueue.erase(it);
}
void *NewtonSim::ResponseQueue::top() const { return OutputQueue.front(); }

} // namespace dramsim3
