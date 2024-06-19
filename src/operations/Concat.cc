#include "Concat.h"

Concat::Concat(std::string name, uint32_t dim) : Operation(name) { _dim = dim; }

std::vector<Ptr<BTensor>> Concat::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _outputs.resize(1);

    _inputs.assign(inputs.begin(), inputs.end());

    std::vector<uint32_t> output_dim(inputs[0]->get_dims());
    for (size_t i = 1; i < inputs.size(); ++i) {
        output_dim[_dim] += inputs[i]->get_dims()[_dim];
        for (size_t j = 0; j < inputs[0]->get_dims().size(); ++j) {
            if (j == _dim) {
                continue;
            }
            assert(inputs[0]->get_dims()[j] == inputs[i]->get_dims()[j]);
        }
    }

    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", output_dim, NPUTensorBufType::ACT, false);

    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "Concat",
        .operation_id = _id,
        .batch = 1,
        .skip = true,
    });

    return _outputs;
}
