#include "Tensor.h"

#include "Model.h"
#include "SimulationConfig.h"
#include "allocator/AddressAllocator.h"
#include "operations/Operation.h"

Tensor::Tensor(std::string name, std::vector<uint32_t> dims, bool produced = false) {
    _id = generate_id();
    _name = name;
    for (int dim : dims) {
        _dims.push_back(dim);
    }
    spdlog::trace("Tensor: {} {}", _name, dims);
    _precision = Config::global_config.precision;
    reserve_address();
    _produced = produced;
}

Tensor::Tensor(const Tensor &tensor) {
    _produced = tensor._produced;
    _id = tensor._id;
    _name = tensor._name;
    _dims = tensor._dims;
    _child_nodes = tensor._child_nodes;
    _address = tensor._address;
    _size = tensor._size;
}

void Tensor::add_child_node(std::shared_ptr<Operation> op) { _child_nodes.push_back(op); }

void Tensor::reserve_address() {
    _size = _precision;
    for (auto dim : _dims) {
        _size *= dim;
    }
    _address = AddressConfig::allocate_address(_size);
    // spdlog::info("{} allocated, dims: {} / size: {} / precision: {} / dram
    // address: {:x}", get_name(), _dims, _size, _precision, _address);
}

// this operation is used to create sram destination address.
addr_type Tensor::calculate_dram_address(std::vector<uint32_t> indexes) {
    auto relative_address = calculate_relative_address(indexes);
    if (relative_address == GARBAGE_ADDR) {
        return GARBAGE_ADDR;
    }
    return _address + relative_address;
}

// given index, return all elements' dram address 
// example: Request indices [1,3] from a tensor of dimension [2,4,6]
// Return the DRAM addresses for [1, 3, 0], [1, 3, 1], [1, 3, 2], [1, 3, 3], [1, 3, 4], [1, 3, 5]
// (6 elements * precision bytes)
std::set<addr_type> Tensor::calculate_dram_addresses(std::vector<uint32_t> indexes) {
    // iterate thru size and precisions
    auto base_addr = _address + calculate_relative_address(indexes);
    auto size = _size;
    for (size_t i = 0; i < indexes.size(); ++i) {
        size /= _dims[i];
    }

    std::set<addr_type> ret;
    for (uint32_t addr = 0; addr < size; addr += _precision) {
        ret.insert(base_addr + addr);
    }
    return ret;
}

// Split the tensor dimensions into operand dimensions and batch dimensions,
// and compute the indices for the batch dimensions.
// (Note that this 'batch' is different from the request's batch size)
// ex: [2, 12, 1, 64] = [batch_size, num_head, seq_len, d_k]
// calculate_batch_indexes(2, 2) => [0, 2]
// 0 => [0, 0]
// 1 => [0, 1]
// 2 => [0, 2]
// 3 => [0, 3]
// 12 => [1, 0]
// Converts the batch index into an n-dimensional index with dimensions (_dim.size() - emb_dim_size).
std::vector<uint32_t> Tensor::calculate_batch_indexes(uint32_t batch_index, size_t emb_dim_size) {
    if (_dims.size() == emb_dim_size) {
        return {};
    }
    assert(_dims.size() > emb_dim_size);
    std::vector<uint32_t> indexes(_dims.size() - emb_dim_size, 0);

    auto dims_riter = _dims.rbegin();
    auto indexes_riter = indexes.rbegin();
    for (size_t i = 0; i < _dims.size(); i++) {
        if (i < emb_dim_size) {
            dims_riter++;
            continue;
        }
        *indexes_riter = batch_index % (*dims_riter);
        batch_index /= (*dims_riter);

        dims_riter++;
        indexes_riter++;
    }

    return indexes;
}

// Split the tensor's dimensions into operand dimensions and batch dimensions,
// and return the DRAM addresses for the operand area.
// ex: [2, 12, 1, 64], calculate_batch_addresses(12, 2)
// Return all DRAM addresses corresponding to [1, 0, x, x].
std::set<addr_type> Tensor::calculate_batch_addresses(uint32_t batch_index, size_t emb_dim_size) {
    return calculate_dram_addresses(calculate_batch_indexes(batch_index, emb_dim_size));
}

// size is relative address.
// example: Request index [1,2] from a tensor of dimension [1,2,3]
// Return tensor base addr + (1*3 + 2)
// tensor [] operator.
addr_type Tensor::calculate_relative_address(std::vector<uint32_t> indexes) {
    assert(indexes.size() <= _dims.size());

    uint32_t offset = _dims.size() - indexes.size(); // 1
    uint64_t size = 0;
    for (int i = 0; i < indexes.size(); ++i) {
        size *= _dims[offset + i];
        size += indexes[i];
    }
    size *= _precision;

    if (size >= _size) {
        // In MatMul, when generating tile instruction addresses, 
        // if it's a GARBAGE_ADDR, handle the exception.
        // Therefore, do not generate an error here.
        // spdlog::info("index out of range {} {}, {} {}", indexes, _dims, size,
        //  _size);
        // print_backtrace();
        return GARBAGE_ADDR;
    }

    return size;
}