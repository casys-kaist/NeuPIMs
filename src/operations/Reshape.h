#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

class Reshape : public Operation {
   public:
    Reshape(std::string name, std::vector<uint32_t> shape);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);

   private:
    std::vector<uint32_t> _shape;
};