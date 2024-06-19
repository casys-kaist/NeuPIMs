#include "Split.h"
#include "spdlog/spdlog.h"

// n,e -> n (1(unit),e) => _inputs.size() == 1, _outputs.size() == n
// Split(1, 0) => n (1,e)
Split::Split(std::string name, std::vector<uint32_t> units, uint32_t dim)
    : Operation(name), _units(units), _dim(dim), _sum(0) {
    _inputs.resize(1);
    for (auto d : _units) {
        _sum += d;
    }
}

/**
 * Split function called when dividing batched request
 *  input:
 *      token embeddings for batched request (N,3E)
 *  output:
 *      n 2DMatrix for each request.
 *      (T1,3E),(T2,3E), ...,(Tn,3E)
 */
std::vector<Ptr<BTensor>> Split::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];

    Ptr<NPUTensor> input = std::static_pointer_cast<NPUTensor>(_inputs[0]);

    auto input_dim = input->get_dims();
    ast(input_dim[_dim] == _sum);

    _outputs.resize(_units.size());

    std::vector<uint32_t> output_dim(input_dim);

    spdlog::info("Split input dim: {}", input->get_dims());
    for (int i = 0; i < _units.size(); ++i) {
        auto output_dim_buf = output_dim;
        output_dim_buf[_dim] = _units[i];
        spdlog::info("Split output dim: {}", output_dim_buf);
        _outputs[i] = std::make_shared<NPUTensor>(_name + "_output" + std::to_string(i),
                                                  output_dim_buf, NPUTensorBufType::ACT, false);
    }
    

    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "Split",
        .operation_id = _id,
        .batch = 1,
        .skip = true,
    });

    return _outputs;
}