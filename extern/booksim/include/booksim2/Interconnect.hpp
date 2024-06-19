#ifndef INTERCONNECT_H_
#define INTERCONNECT_H_

#include <map>
#include <memory>
#include <string>
#include <queue>
#include <cstdint>

class Network;
class Flit;
class GNNTrafficManager;
class NetworkRequest;
class Stats;
class Router;
class OutputSet;
namespace booksim2 {
class Interconnect {
public:
  enum class Type {
    READ, WRITE, READ_REPLY, WRITE_REPLY, ANY
  };

  //Interconnect(BookSimConfig offchip_config);
  Interconnect(const std::string ConfigFilePath, int NumNodes);
  ~Interconnect();
  //
  void run();
  // bool done() const override;
  uint32_t get_flit_size() { return flit_size; }

  bool is_full(uint32_t nid, uint32_t subnet, uint32_t size) const;
  void push(void* packet, uint32_t subnet, 
            uint64_t addr, int bytes, Type type, int src, int dst);
  bool is_empty(uint32_t nid, uint32_t subnet) const;
  const void* top(uint32_t nid, uint32_t subnet) const;
  void pop(uint32_t nid, uint32_t subnet);

  void Transfer2BoundaryBuffer(uint32_t subnet, uint32_t output);
  void WriteOutBuffer(uint32_t subnet, int output, Flit* flit);
  Flit* GetEjectedFlit(uint32_t subnet, uint32_t nid);

  // TODO: check whether this function works correctly
  uint64_t get_cycle() const { return clk; }
  bool print_activity() { return gPrintActivity; }

  Stats* GetStats(const std::string &name);

  // void register_stat(std::shared_ptr<CycleObjectStats> stats) override;

  const int HEADER_SIZE;

  bool gPrintActivity;
  int gK;
  int gN;
  int gC;
  int gNodes;
  bool gTrace;
  std::ofstream *gWatchOut;

  // Anynet
  std::map<int, int> *anynet_global_routing_table;

  int gNumVCs;
  int gReadReqBeginVC;
  int gReadReqEndVC;
  int gWriteReqBeginVC;
  int gWriteReqEndVC;
  int gReadReplyBeginVC;
  int gReadReplyEndVC;
  int gWriteReplyBeginVC;
  int gWriteReplyEndVC;
  using tRoutingFunction = void(*)(const Router*,
                                   const Flit*,
                                   int,
                                   OutputSet*,
                                   bool);
  std::map<std::string, tRoutingFunction> gRoutingFunctionMap;
private:

  class BoundaryBufferItem {
  public:
    BoundaryBufferItem(): num_packets(0) {}
    inline uint32_t size(void) const { return buffer.size(); }
    inline bool is_empty() const { return num_packets == 0; }
    void pop();
    const void* top() const;
    void push(void* packet, bool is_tail);
    typedef struct Buffer {
      void* packet;
      bool is_tail;
    } Buffer;
  private:
    std::vector<Buffer> buffer;
    //std::queue<void*> buffer;
    //std::queue<bool> tail_flag;
    uint32_t num_packets;
  };

  const uint32_t REQUEST_VC = 0;
  const uint32_t RESPONSE_VC = 1;

  typedef std::queue<Flit*> EjectionBufferItem;

  int num_nodes;
  int num_subnets;
  int vcs;
  uint32_t flit_size;
  std::vector<Network*> nets;
  uint32_t input_buffer_capacity;

  uint32_t boundary_buffer_capacity;
  // size: [subnets][nodes][vcs]
  std::vector<std::vector<std::vector<BoundaryBufferItem>>> boundary_buffer;
  uint32_t ejection_buffer_capacity;
  std::vector<std::vector<std::vector<EjectionBufferItem>>> ejection_buffer;

  std::vector<std::vector<std::queue<Flit*>>> ejected_flit_queue;

  std::vector<std::vector<int>> round_robin_turn;

  uint64_t clk;

  GNNTrafficManager* traffic_manager;
  //std::unique_ptr<GNNSimConfig> BookSimConfig;
  // using ConnectionType = DomainInterface::Type;
  // int getSrcID(int InterfaceID, ConnectionType Type) const;
  // int getDstID(int InterfaceID, ConnectionType Type) const;

  void initParameters();
  // std::shared_ptr<IcntStats> stats;
};
}
#endif
