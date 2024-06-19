#pragma once
#include "Common.h"

class Model;
class Operation;

class Tensor {
  public:
    Tensor(std::string name, std::vector<uint32_t> dims, bool produced);
    Tensor(const Tensor &tensor);

    void add_child_node(std::shared_ptr<Operation> op);

    uint32_t get_id() { return _id; }
    std::string get_name() { return _name; }
    std::vector<uint32_t> get_dims() { return _dims; }
    void set_produced() { _produced = true; }
    bool get_produced() { return _produced; }
    uint32_t num_child_nodes() { return _child_nodes.size(); }
    std::shared_ptr<Operation> get_child_node(uint32_t id) { return _child_nodes[id]; }
    std::vector<std::shared_ptr<Operation>> get_child_nodes() { return _child_nodes; }

    addr_type calculate_relative_address(std::vector<uint32_t> indexes);
    addr_type calculate_dram_address(std::vector<uint32_t> indexes);
    std::set<addr_type> calculate_dram_addresses(std::vector<uint32_t> indexes);
    std::vector<uint32_t> calculate_batch_indexes(uint32_t batch_index, size_t emb_dim_size);
    std::set<addr_type> calculate_batch_addresses(uint32_t batch_index, size_t emb_dim_size);

    addr_type get_address() { return _address; }
    uint32_t get_size() { return _size; }

  private:
    bool _produced;
    uint32_t _id;
    std::string _name;
    std::vector<uint32_t> _dims;
    std::shared_ptr<Operation> _src_node;
    std::vector<std::shared_ptr<Operation>> _child_nodes;
    addr_type _address;
    uint32_t _size;
    int _precision;

    void reserve_address();
    friend Model;
};