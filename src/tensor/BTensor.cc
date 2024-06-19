#include "BTensor.h"

void BTensor::add_child_node(Ptr<Operation> op) { _child_nodes.push_back(op); }
void BTensor::clear_child_nodes() { _child_nodes.clear(); }