// #pragma once
// #include "Operation.h"

// class Transpose : public Operation {
//   public:
//     Transpose(std::string name, std::vector<uint32_t> perm);

//     std::vector<std::shared_ptr<BatchedTensor>>
//     get_outputs(std::vector<std::shared_ptr<BatchedTensor>> inputs);

//   private:
//     std::vector<uint32_t> _perm;
// };