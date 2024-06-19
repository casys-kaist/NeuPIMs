#include "Reshape.h"

Reshape::Reshape(std::string name, std::vector<uint32_t> shape) : Operation(name) {
    _inputs.resize(1);

    _shape.assign(shape.begin(), shape.end());
}

// reshape can do unsqueeze
std::vector<Ptr<BTensor>> Reshape::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _outputs.resize(1);

    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];

    auto input = std::static_pointer_cast<NPUTensor>(_inputs[0]);
    auto input_dims = input->get_dims();

    uint32_t acc_in = 1;
    for (auto &i : input_dims) {
        acc_in *= i;
    }
    uint32_t acc_out = 1;
    for (auto &i : _shape) {
        acc_out *= i;
    }

    assert(acc_in == acc_out);

    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", _shape, NPUTensorBufType::ACT, false);

    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "Reshape",
        .operation_id = _id,
        .batch = 1,
        .skip = true,
    });

    return _outputs;
}