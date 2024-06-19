// #include "Transpose.h"

// #include "../Tensor.h"

// Transpose::Transpose(std::string name, std::vector<uint32_t> perm) : Operation(name) {
//     assert(perm.size() == 4);
//     for (uint32_t i = 0; i < 4; ++i) {
//         assert(std::find(perm.begin(), perm.end(), i) != perm.end());
//     }
//     _perm.assign(perm.begin(), perm.end());

//     _inputs.resize(1);
// }

// // Transpose does not change shapes.
// std::vector<std::shared_ptr<BatchedTensor>>
// Transpose::get_outputs(std::vector<std::shared_ptr<BatchedTensor>> inputs) {
//     set_as_parent_tensor(inputs);

//     _outputs.resize(1);

//     assert(inputs.size() == 1);
//     _inputs[0] = inputs[0];

//     auto dims = inputs[0]->get_dims();
//     std::vector<uint32_t> output_dims(4);
//     for (uint32_t i = 0; i < 4; ++i) {
//         output_dims[i] = dims[_perm[i]];
//     }

//     _outputs[0] = std::make_shared<BatchedTensor>(_name + "_output", output_dims, false);

//     _tiles.push_back(Tile{
//         .status = Tile::Status::INITIALIZED,
//         .optype = "Transpose",
//         .operation_id = _id,
//         .batch = 1,
//         .skip = true,
//     });

//     return _outputs;
// }