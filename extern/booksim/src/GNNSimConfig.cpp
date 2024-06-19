#include "GNNSimConfig.hpp"

GNNSimConfig::GNNSimConfig() {
  _int_map["perfect_icnt"] = 0; // if set overrides fixed_lat_per_hop setting
  _int_map["fixed_lat_per_hop"] = 0; // if set icnt is NOT simulated instead packets are sent into destination based on a fixed_lat_per_hop

  _int_map["use_map"] = 1;
  // config SMs and memory nodes map
  AddStrField("memory_node_map", "");

  _int_map["flit_size"] = 32;
  // The header size of NVLINK is 16B
  _int_map["header_size"] = 16;
  
  _int_map["input_buffer_size"] = 0;
  _int_map["ejection_buffer_size"] = 0; // if left zero the simulator will use the vc_buf_size instead
  _int_map["boundary_buffer_size"] = 16;
  

  // FIXME: obsolete, unsupport configs
  _int_map["output_extra_latency"] = 0;
  _int_map["network_count"] = 2; // number of independent interconnection networks (if it is set to 2 then 2 identical networks are created: sh2mem and mem2shd )
  _int_map["enable_link_stats"]    = 0;     // show output link and VC utilization stats

  //stats
  _int_map["MATLAB_OUTPUT"]        = 0;     // output data in MATLAB friendly format
  _int_map["DISPLAY_LAT_DIST"]     = 0; // distribution of packet latencies
  _int_map["DISPLAY_HOP_DIST"]     = 0;     // distribution of hop counts
  _int_map["DISPLAY_PAIR_LATENCY"] = 0;     // avg. latency for each s-d pair
}
