#include <fstream>
#include <sstream>
#include <algorithm>

#include "Interconnect.hpp"

#include "routefunc.hpp"
#include "networks/network.hpp"
#include "GNNSimConfig.hpp"
#include "GNNTrafficManager.hpp"
#include "routefunc.hpp"
namespace booksim2 {
Interconnect::Interconnect(const std::string ConfigFilePath,
                           int NumNodes)
    : HEADER_SIZE(0/*TODO:*/),
      num_nodes(NumNodes),
      clk(0) {
  GNNSimConfig Config;
  Config.ParseFile(ConfigFilePath);
  // TODO: automatically describe network topology
  // std::ofstream anynet_file("booksim_configs/anynet/anynet_file");
  // // connect compute nodes
  // for (int n = 0; n < num_compute_nodes; ++n) {
  //   anynet_file << "router " << (n * 2) << " "
  //               << "router " << (n * 2) + 1 << " ";
  //   if (num_compute_nodes > 1) {
  //     anynet_file << "router " << ((n * 2) + 2) % (num_compute_nodes*2) << " ";
  //   }
  //   anynet_file << "node " << n << std::endl;
  // }
  // // connect memory nodes
  // for (int n = 0; n < num_compute_nodes; ++n) {
  //   anynet_file << "router " << (n * 2) + 1 << " "
  //               << "node " << n + num_compute_nodes << std::endl;
  // }
  // anynet_file.close();
  // offchip_icnt_config.AddStrField("network_file", 
  //                                 "booksim_configs/anynet/anynet_file");
  // package_icnt_config.AddStrField("network_file", 
  //                                 "booksim_configs/anynet/anynet_file");
  // offchip_icnt_config.AddIntField("num_nodes", num_compute_nodes * 2);
  // package_icnt_config.AddIntField("num_nodes", num_compute_nodes * 2);
  initParameters();
  num_subnets = Config.GetInt("subnets");
  assert(num_subnets);
  nets.resize(num_subnets);
  InitializeRoutingMap(this, Config);
  for (uint32_t n = 0; n < num_subnets; ++n) {
    std::ostringstream name;
    name << "network_" << n;
    nets[n] = Network::New(Config, name.str(), this);
  }

  flit_size = Config.GetInt("flit_size");
  if (Config.GetInt("ejection_buffer_size")) {
    ejection_buffer_capacity = Config.GetInt("ejection_buffer_size");
  } else {
    ejection_buffer_capacity = Config.GetInt("vc_buf_size");
  }

  boundary_buffer_capacity = Config.GetInt("boundary_buffer_size");
  assert(boundary_buffer_capacity);
  if (Config.GetInt("input_buffer_size")) {
    input_buffer_capacity = Config.GetInt("input_buffer_size");
  } else {
    input_buffer_capacity = 9;
  }

  std::string watch_file = Config.GetStr("watch_out");
  if (watch_file == "")
    gWatchOut = nullptr;
  else if (watch_file == "-")
    //gWatchOut = &cout;
    gWatchOut = nullptr;
  else
    gWatchOut = new std::ofstream(watch_file);


  vcs = Config.GetInt("num_vcs");

  // Init buffer
  boundary_buffer.resize(num_subnets);
  ejection_buffer.resize(num_subnets);
  round_robin_turn.resize(num_subnets);
  ejected_flit_queue.resize(num_subnets);

  for (int s = 0; s < num_subnets; ++s) {
    boundary_buffer[s].resize(num_nodes);
    ejection_buffer[s].resize(num_nodes);
    round_robin_turn[s].resize(num_nodes);
    ejected_flit_queue[s].resize(num_nodes);

    for (int n = 0; n < num_nodes; ++n) {
      boundary_buffer[s][n].resize(vcs);
      ejection_buffer[s][n].resize(vcs);
    }
  }
  traffic_manager =
    static_cast<GNNTrafficManager*>(TrafficManager::New(Config, nets, this));
  traffic_manager->Init();
}

Stats* Interconnect::GetStats(const std::string &name) {
  Stats* test = traffic_manager->getStats(name);
  if(test == 0){
    cout<<"warning statistics "<<name<<" not found"<<endl;
  }
  return test;
}

// bool Interconnect::done() const {
//   bool empty = true;
//   for (int subnet = 0; subnet < num_subnets; ++subnet) {
//     for (int n = 0; n < num_nodes; ++n) {
//       for (int vc = 0; vc < vcs; ++vc) {
//         empty &= boundary_buffer[subnet][n][vc].is_empty();
//         empty &= ejection_buffer[subnet][n][vc].empty();
//       }
//     }
//   }
//   return empty;
// }

// void Interconnect::register_stat(std::shared_ptr<CycleObjectStats> _stats) {
//   stats = std::static_pointer_cast<IcntStats>(_stats);
// }
void Interconnect::run() {
  clk++;
  traffic_manager->_Step();
}

bool Interconnect::is_full(uint32_t nid, uint32_t subnet, uint32_t size) const {
  // WARNING: Must append header to the payload
  size += HEADER_SIZE;
  uint32_t num_flits = size / flit_size + ((size % flit_size) ? 1 : 0);
  // TODO: select input_queue depending on the node (memory and compute)
  // currently, set to 0
  // [subnets][nodes][vcs]
  uint32_t expected_size = 
    traffic_manager->_input_queue[subnet][nid][0].size() + num_flits;

  return expected_size > input_buffer_capacity;
}

void Interconnect::push(void* packet, uint32_t subnet, 
                        uint64_t addr, int bytes, Type type, int src, int dst) {
  // TODO: Get srouce from the Queue ID
  // TODO: Need to calculate the destination
  // _GeneratePacket will calculate packet type and the number of flits
  traffic_manager->_GeneratePacket(packet, addr, bytes, type, HEADER_SIZE, subnet, 
                                   0, traffic_manager->_time, src, dst);
}

void Interconnect::Transfer2BoundaryBuffer(uint32_t subnet, uint32_t output) {
  Flit* flit;
  for (uint32_t vc = 0; vc < vcs; ++vc) {
    if (!ejection_buffer[subnet][output][vc].empty() && 
        boundary_buffer[subnet][output][vc].size() < boundary_buffer_capacity) {
      flit = ejection_buffer[subnet][output][vc].front();
      assert(flit);
      ejection_buffer[subnet][output][vc].pop();
      boundary_buffer[subnet][output][vc].push(flit->data, flit->tail);
      // Indicates this flit is already popped from the ejection buffer and
      // ready for credit return
      ejected_flit_queue[subnet][output].push(flit);
      if (flit->head) {
        assert(flit->dest == output);
      }
    }
  }
}

void Interconnect::WriteOutBuffer(uint32_t subnet, int output, Flit* flit) {
  int vc = flit->vc;
  assert(ejection_buffer[subnet][output][vc].size() < ejection_buffer_capacity);
  ejection_buffer[subnet][output][vc].push(flit);

  // if (flit->tail) {
  //   if (flit->type == Flit::READ_REPLY) {
  //     stats->read_reply_bytes[output] += flit->payload_size;
  //   } else if (flit->type == Flit::READ_REQUEST) {
  //     stats->read_request_bytes[output] += flit->payload_size;
  //   } else if (flit->type == Flit::WRITE_REPLY) {
  //     stats->write_reply_bytes[output] += flit->payload_size;
  //   } else if (flit->type == Flit::WRITE_REQUEST) {
  //     stats->write_request_bytes[output] += flit->payload_size;
  //   } else {
  //     assert(false);
  //   }
  // }
}

Flit* Interconnect::GetEjectedFlit(uint32_t subnet, uint32_t nid) {
  Flit* flit = NULL;
  if (!ejected_flit_queue[subnet][nid].empty()) {
    flit = ejected_flit_queue[subnet][nid].front();
    ejected_flit_queue[subnet][nid].pop();
  }
  return flit;
}

bool Interconnect::is_empty(uint32_t nid, uint32_t subnet) const {
  return std::find_if(
      std::begin(boundary_buffer[subnet][nid]),
      std::end(boundary_buffer[subnet][nid]),
      [](const BoundaryBufferItem &item) {
        return !item.is_empty();
      }) == std::end(boundary_buffer[subnet][nid]);
}

const void* Interconnect::top(uint32_t nid, uint32_t subnet) const {
  int turn = round_robin_turn[subnet][nid];
  for (int vc = 0; vc < vcs; ++vc) {
    if (!boundary_buffer[subnet][nid][turn].is_empty()) {
      return boundary_buffer[subnet][nid][turn].top();
    }
    turn = (turn + 1) % vcs;
  }
  assert(false);
  return nullptr;
}

// Pop from compute node clock domain
void Interconnect::pop(uint32_t nid, uint32_t subnet) {
  int turn = round_robin_turn[subnet][nid];
  void* data = nullptr;
  // for (int vc = 0; (vc < vcs) && (data == NULL); ++vc) {
  int vc = 0;
  for (vc = 0; vc < vcs; ++vc) {
    if (!boundary_buffer[subnet][nid][turn].is_empty()) {
      boundary_buffer[subnet][nid][turn].pop();
      break;
      //data = boundary_buffer[subnet][nid][turn].pop();
    } else {
      turn = (turn + 1) % vcs;
    }
  }
  if (vc == vcs) {
    round_robin_turn[subnet][nid] = turn;
  }
  // //assert(data != nullptr);
  // if (data) {
  //   round_robin_turn[subnet][nid] = turn;
  // }
}

void Interconnect::BoundaryBufferItem::pop() {
  assert(!is_empty());
  auto it = std::find_if(buffer.begin(), buffer.end(), [](const Buffer& buf) {
      return buf.is_tail;
  });
  assert(it != buffer.end());
  assert(it->packet != nullptr);
  // delete it->packet;
  // std::unique_ptr<DomainCrossRequest> data = std::move(it->data);
  buffer.erase(buffer.begin(), it + 1);
  num_packets--;
  // return data;
}

const void* Interconnect::BoundaryBufferItem::top() const {
  assert(!is_empty());
  // Find first occurence of tail flag
  auto it = std::find_if(
      std::begin(buffer), 
      std::end(buffer), 
      [](const Buffer &buf) {
        return buf.is_tail;
      });
  assert(it != std::end(buffer));
  assert(it->packet!= nullptr);
  return it->packet;
}

void Interconnect::BoundaryBufferItem::push(void* packet, bool is_tail) {
  buffer.push_back({packet, is_tail});
  //buffer.push(data);
  //tail_flag.push(is_tail);
  if (is_tail) {
    num_packets++;
  }
}

void Interconnect::initParameters() {
  gPrintActivity = false;
  gK = 0;
  gN = 0;
  gC = 0;
  gNodes = 0;
  gTrace = false;

  gNumVCs = 0;
  gReadReqBeginVC = 0;
  gReadReqEndVC = 0;
  gWriteReqBeginVC = 0;
  gWriteReqEndVC = 0;
  gReadReplyBeginVC = 0;
  gReadReplyEndVC = 0;
  gWriteReplyBeginVC = 0;
  gWriteReplyEndVC = 0;
}

Interconnect::~Interconnect() = default;
}