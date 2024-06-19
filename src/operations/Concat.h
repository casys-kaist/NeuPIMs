#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

class Concat : public Operation {
   public:
    Concat(std::string name, uint32_t dim);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);

   private:
    uint32_t _dim;
};