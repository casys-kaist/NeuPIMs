#ifndef __NEWTONSIM__H
#define __NEWTONSIM__H

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace dramsim3 {
class Config;
class BaseDRAMSystem;
enum class TransactionType;
// This should be the interface class that deals with CPU
class NewtonSim {
  public:
    NewtonSim(const std::string &config_file, const std::string &output_dir);
    ~NewtonSim();
    void ClockTick();
    void RegisterCallbacks() { return; }
    double GetTCK() const;
    int GetBusBits() const;
    int GetBurstLength() const;
    int GetQueueSize() const;
    int GetChannel(uint64_t hex_addr) const;
    void PrintStats() const;
    void ResetStats();

    uint64_t GetAvgPIMCycles();
    void ResetPIMCycle();

    uint64_t MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col);
    uint64_t EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps, int num_readres);

    bool WillAcceptTransaction(uint64_t hex_addr, int req_type) const;
    bool AddTransaction(uint64_t hex_addr, int req_type, void *original_req);

    // added for ONNXim
    bool IsEmpty(uint32_t channel) const;
    void *Top(uint32_t channel) const;
    void Pop(uint32_t channel);

  private:
    // These have to be pointers because Gem5 will try to push this object
    // into container which will invoke a copy constructor, using pointers
    // here is safe
    class ResponseQueue;
    Config *config_;
    BaseDRAMSystem *dram_system_;
    std::vector<ResponseQueue> response_queues_;
    std::vector<ResponseQueue> response_queues_rd_;
    std::vector<ResponseQueue> response_queues_wr_;
    std::vector<ResponseQueue> response_queues_pim_;
    std::multimap<uint64_t, void *> pending_read_q_;
    std::multimap<uint64_t, void *> pending_write_q_;

    int income_req_cnt_ = 0;
    int outcome_req_cnt_ = 0;

    void PushToPendingQueue(uint64_t addr, TransactionType req_type, void *original_req);
};
class NewtonSim::ResponseQueue {
  public:
    ResponseQueue(int Size);
    bool isAvailable() const;
    bool isAvailable(uint32_t count) const;
    bool isEmpty() const;
    void reserve();
    void push(void *original_req);
    void *top() const;
    void pop();

  private:
    const uint32_t Size;
    uint32_t NumReserved;
    std::vector<void *> OutputQueue;
};

} // namespace dramsim3

#endif
