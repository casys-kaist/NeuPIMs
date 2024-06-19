#include "NPUTensor.h"

#include <memory>

#include "NPUTensor2D.h"

/**
 * NPUTensor with NPUTensor2D: Tensor object for weight, activation
 * Note that NPUTensorKV should be initialized with kv_type, not buf_type
 *  Weight:
 *      buf_type: NPUTensorBufType::WGT
 *      dimension: 2D (Fully connected)
 *
 *  Activation:
 *      buf_type: NPUTensorBufType::ACT
 *      dimension: 3D (including batch?)
 */
NPUTensor::NPUTensor(std::string name, std::vector<uint32_t> dims, NPUTensorBufType buf_type,
                     bool produced) {
    ast(buf_type != NPUTensorBufType::KV);

    _id = generate_id();
    _name = name;
    _dims = dims;
    _produced = produced;
    _precision = Config::global_config.precision;

    uint32_t num_inners = 1;
    std::vector<uint32_t> inner_dims = dims;
    if (dims.size() == 3) {
        num_inners = dims[0];
        inner_dims = slice(dims, 1, -1);
    }
    for (int i = 0; i < num_inners; ++i) {
        _inners.push_back(std::make_shared<NPUTensor2D>(inner_dims, buf_type));
    }

    _is_transposed = false;
}

/**
 * NPUTensor with NPUTensorKV: Tensor object for Key and Value
 *  Key:
 *      kv_type: NPUTensorKVType::Key
 *      dimension: 3D (nh,dk,T)
 *
 *  Value:
 *      kv_type: NPUTensorKVType::Value
 *      dimension: 3D (nh,T,dk)
 */
NPUTensor::NPUTensor(std::string name, std::vector<uint32_t> dims, NPUTensorKVType kv_type,
                     bool produced) {
    _id = generate_id();
    _name = name;
    _dims = dims;
    _produced = produced;
    _precision = Config::global_config.precision;

    uint32_t num_inners = 1;
    std::vector<uint32_t> inner_dims = dims;

    // XXX: can dims size be 2, without num_heads?
    if (dims.size() == 2) {
        ast(0);
    }
    if (dims.size() == 3) {
        num_inners = dims[0];  // num_heads
        inner_dims = slice(dims, 1, -1);
    }
    for (int i = 0; i < num_inners; ++i) {
        _inners.push_back(std::make_shared<NPUTensorKV>(inner_dims, kv_type));
    }
}

NPUTensor::NPUTensor(std::string name, Ptr<NPUTensor2D> tensor, bool produced) {
    _id = generate_id();
    _name = name;
    _dims = tensor->_dims;
    _produced = produced;
    _precision = Config::global_config.precision;
    _inners = {tensor};
}

void NPUTensor::set_transposed() {
    assert(_inners[0]->_buf_type == NPUTensorBufType::ACT ||
           _inners[0]->_buf_type == NPUTensorBufType::WGT);
    _is_transposed = true;
}

void NPUTensor::unset_transposed() {
    assert(_inners[0]->_buf_type == NPUTensorBufType::ACT ||
           _inners[0]->_buf_type == NPUTensorBufType::WGT);
    _is_transposed = false;
}

std::vector<uint32_t> NPUTensor::get_dims() {
    if (_is_transposed) {
        std::vector<uint32_t> ret(_dims.size());
        std::reverse_copy(_dims.begin(), _dims.end(), ret.begin());
        return ret;
    }
    return _dims;
}

addr_type NPUTensor::get_addr(std::vector<uint32_t> indexes) {
    // spdlog::info("(NPUTensor::get_addr) indexes:{}, inners.size:{}", indexes, _inners.size());
    // spdlog::info("_inners[0]->dims.size:{}", _inners[0]->_dims.size());
    // spdlog::info("_inners.size() + _inners[0]->_dims.size(): {}",
    //              _inners.size() + _inners[0]->_dims.size());
    // int idx_size = indexes.size();
    // ast(_inners.size() > 0);
    // ast(_dims.size() == idx_size);
    // ast(0 <= idx_size && idx_size <= 3);

    ast(_dims.size() == indexes.size());

    std::vector<uint32_t> dims(_dims.begin(), _dims.end());

    if (_is_transposed) {
        std::copy(_dims.rbegin(), _dims.rend(), dims.begin());
    }

    for (size_t i = 0; i < dims.size(); ++i) {
        if (indexes[i] >= dims[i]) {
            return GARBAGE_ADDR;
        }
    }

    return 0;

    if (indexes.size() <= 2) {  // bias, wgt
        return _inners[0]->get_addr(indexes);
    }
    return _inners[indexes[0]]->get_addr(slice(indexes, 1, -1));
}

std::vector<addr_type> NPUTensor::get_all_addrs() {
    ast(_inners.size() > 0);
    std::vector<addr_type> res;
    for (int i = 0; i < _inners.size(); i++) {
        auto addrs = _inners[i]->get_all_addrs();
        for (auto addr : addrs) {
            res.push_back(addr);
        }
    }
    return res;
}

void NPUTensor::add_token() {
    for (auto inner : _inners) {
        std::static_pointer_cast<NPUTensorKV>(inner)->add_token();
    }
}

// get_row_addrs: row_idx -> [addr]
// Used when invoking a 2D tensor by 1D row units in LayerNorm,
// or when invoking a 3D tensor by 1D row units in Softmax.
// Should only be used when inner is NPUTensor2D.
std::vector<addr_type> NPUTensor::get_row_addrs(uint32_t row_idx) {
    // ast(_inners.size() == 1);
    // ast(_dims.size() == 2);
    if (_dims.size() == 2) {
        // ln
        return std::static_pointer_cast<NPUTensor2D>(_inners[0])->get_row_addrs(row_idx);
    } else if (_dims.size() == 3) {
        // Softmax
        auto l = _dims[1];
        return std::static_pointer_cast<NPUTensor2D>(_inners[row_idx / l])
            ->get_row_addrs(row_idx % l);
    }
    ast(0);
}

std::vector<Ptr<NPUTensor>> NPUTensor::split_by_row(std::vector<uint32_t> row_dims) {
    ast(_inners.size() == 1);
    ast(_dims.size() == 2);
    ast(_inners[0]->_buf_type == NPUTensorBufType::ACT);

    std::vector<Ptr<NPUTensor>> ret;
    Ptr<NPUTensor2D> inner = std::static_pointer_cast<NPUTensor2D>(_inners[0]);
    std::vector<Ptr<NPUTensor2D>> splited_inners = inner->split_by_row(row_dims);
    int i = 0;
    for (auto inner : splited_inners) {
        std::string new_name = _name + "_" + std::to_string(i);
        Ptr<NPUTensor> tensor = std::make_shared<NPUTensor>(new_name, inner, true);
        ret.push_back(tensor);
    }
    return ret;
}