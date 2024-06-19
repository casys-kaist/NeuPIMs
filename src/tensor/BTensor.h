#pragma once

#include "../Common.h"

class Operation;

class BTensor {
   public:
    BTensor() = default;
    ~BTensor() = default;

    void add_child_node(Ptr<Operation> op);
    void clear_child_nodes();

    uint32_t get_id() { return _id; }
    std::string get_name() { return _name; }
    std::vector<uint32_t> get_dims() { return _dims; }
    void set_produced() { _produced = true; }
    bool get_produced() { return _produced; }
    Ptr<Operation> get_src_node() { return _src_node; }
    uint32_t num_child_nodes() { return _child_nodes.size(); }
    Ptr<Operation> get_child_node(uint32_t id) { return _child_nodes[id]; }
    std::vector<Ptr<Operation>> get_child_nodes() { return _child_nodes; }

    virtual addr_type get_addr(std::vector<uint32_t> indexes) = 0;
    virtual std::vector<addr_type> get_all_addrs() = 0;
    virtual void add_token() = 0;

    bool _produced;
    uint32_t _id;
    std::string _name;
    std::vector<uint32_t> _dims;
    Ptr<Operation> _src_node;
    std::vector<Ptr<Operation>> _child_nodes;
    uint32_t _precision;
};