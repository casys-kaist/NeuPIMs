#include "Operation.h"

#include <memory>

SimulationConfig Operation::_config;

Operation::Operation(MappingTable mapping_table) {
    _id = generate_id();
    _finish = false;
    spdlog::trace("Node {} op_type {}", _name.c_str(), _optype.c_str());
    if (_config.layout == "NCHW") {
        Ndim = 0;
        Cdim = 1;
        Hdim = 2;
        Wdim = 3;
    } else if (_config.layout == "NHWC") {
        Ndim = 0;
        Cdim = 3;
        Hdim = 1;
        Wdim = 2;
    }
    Mdim = 0;
    Cdim_w = 1;
    Sdim = 2;
    Rdim = 3;
}

Operation::Operation(std::string name) {
    _id = generate_id();
    _optype = name;
    _name = name;
    _finish = false;

    // spdlog::info("operation {} generated", name);

    _stat = OperationStat(_name);
    _acc_spad_addr = ACCUM_SPAD_BASE;
    _spad_addr = SPAD_BASE;
}

std::pair<addr_type, uint32_t> Operation::allocate_sram_addr(uint32_t size, bool accum) {
    // size means number of elements
    addr_type spad_addr = accum ? _acc_spad_addr : _spad_addr;
    uint32_t size_in_byte = size * _config.precision;
    if (accum) {
        _acc_spad_addr += size_in_byte;
    } else {
        _spad_addr += size_in_byte;
    }

    return std::make_pair(spad_addr, size_in_byte);
}

std::vector<Ptr<BTensor>> Operation::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    spdlog::info("parent");

    return {};
}

void Operation::set_as_parent_tensor(std::vector<Ptr<BTensor>> inputs) {
    for (auto input : inputs) {
        input->add_child_node(shared_from_this());
    }
}

void Operation::reduce_tile(Tile &tile) {
    // xxx is it necessary?
    // _op_stat.tile_stats.push_back(tile.stat);
    _stat.update_stat(tile.stat);
}

// xxx no need for this operation. pass it to _stat->repr()
// std::string Operation::repr() {
//     return _stat.repr();
// }

Operation::Operation(const Operation &operation) {
    _id = operation._id;
    _optype = operation._optype;
    _name = operation._name;
    _finish = operation._finish;
    _attributes = operation._attributes;
    _tiles = operation._tiles;
    _inputs = operation._inputs;
    _outputs = operation._outputs;
}

void Operation::set_finish() {
    for (auto output : _outputs) {
        output->set_produced();
    }
    _finish = true;
    spdlog::trace("layer {} finish", _name.c_str());
}

std::vector<std::shared_ptr<Operation>> Operation::get_child_nodes() {
    std::vector<std::shared_ptr<Operation>> result;
    for (auto output : _outputs) {
        spdlog::trace("num child nodes {}", output->num_child_nodes());
        for (auto child : output->get_child_nodes()) {
            result.push_back(child);
        }
    }
    return result;
}

bool Operation::check_executable() {
    // Execution is possible only after the creation of all tensors that have this operation as a child.
    bool result = true;
    for (auto input : _inputs) {
        result = result && input->get_produced();
        spdlog::trace("Layer {}: Input {} Produced {}", _name.c_str(), input->get_name().c_str(),
                      input->get_produced());
    }
    return result;
}

std::deque<Tile> Operation::get_tiles() { return _tiles; }