#include "SplitEncoding.h"

// Not used anymore
SplitEncoding::SplitEncoding(std::string name) : Operation(name) { _inputs.resize(1); }

/**
 * QKVSplit function called at initialization phase
 *  input:
 *      QKV concatenated (l,3E)
 *  output:
 *      Q(nh,l,dk)[NPUTensor2D], K(nh,dk,l)[NPUTensorKV], V(nh,l,dk)[NPUTensorKV]
 */
std::vector<Ptr<BTensor>> SplitEncoding::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];

    Ptr<NPUTensor> input = std::static_pointer_cast<NPUTensor>(_inputs[0]);

    auto input_dim = input->get_dims();
    auto nh = Config::global_config.model_n_head;
    auto l = input_dim[0];
    auto dk = Config::global_config.model_n_embd / Config::global_config.model_n_head;
    ast((*input_dim.rbegin()) == 3 * nh * dk);
    spdlog::info("SplitEnc input dim : {}", input_dim);

    _outputs.resize(3);

    std::vector<uint32_t> output_dim0 = {nh, l, dk};
    std::vector<uint32_t> output_dim1 = {nh, dk, l};

    _outputs[0] = std::make_shared<NPUTensor>(_name + "_output" + std::to_string(0), output_dim0,
                                              NPUTensorBufType::ACT, false);
    _outputs[1] = std::make_shared<NPUTensor>(_name + "_output" + std::to_string(1), output_dim1,
                                              NPUTensorKVType::KEY, false);
    _outputs[2] = std::make_shared<NPUTensor>(_name + "_output" + std::to_string(2), output_dim0,
                                              NPUTensorKVType::VALUE, false);

    spdlog::info("SplitEnc ouput dim 0,2 : {} / output dim 1 : {}", output_dim0, output_dim1);

    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "SplitEncoding",
        .operation_id = _id,
        .batch = 1,
        .skip = true,
    });

    return _outputs;
}