#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include "Common.h"
#include "Mapping.h"
#include "Tensor.h"
#include "helper/HelperFunctions.h"
#include "operations/Add.h"
#include "operations/Attention.h"
#include "operations/Concat.h"
#include "operations/FusedMHA.h"
#include "operations/Gelu.h"
#include "operations/LayerNorm.h"
#include "operations/MatMul.h"
#include "operations/Microbench.h"
#include "operations/NeuPIMSAttend.h"
#include "operations/NeuPIMSLogitSoftmax.h"
#include "operations/Operation.h"
#include "operations/PIMGEMV.h"
#include "operations/PIMGEMVAdd.h"
#include "operations/PIMGEMVSoftmax.h"
#include "operations/Reshape.h"
#include "operations/Softmax.h"
#include "operations/Split.h"
#include "operations/SplitDecoding.h"
// #include "operations/SplitEncoding.h"
#include "operations/Transpose.h"
#include "tensor/NPUTensor.h"

#define LAYER(i) ("layer" + std::to_string(i))

namespace BlockType {
extern std::string Attention;
extern std::string FeedForward;
}  // namespace BlockType

namespace OperationType {
extern std::string LayerNorm;
extern std::string QKVGen;
extern std::string Projection;
extern std::string FullyConnected1;
extern std::string FullyConnected2;
extern std::string QKVSplit;
extern std::string QKMatMul;
extern std::string SoftMax;
extern std::string LsVMatMul;
extern std::string AReshape;
extern std::string Residual;
extern std::string Gelu;
extern std::string BatchSplit;
extern std::string BatchConcat;

extern std::string KCacheConcat;
extern std::string VCacheConcat;
extern std::string VConcat;

extern std::string PIMGEMVSoftmax;
extern std::string PIMGEMVAdd;
extern std::string Microbench;
extern std::string NeuPIMSLogitSoftmax;
extern std::string Attention;
extern std::string NeuPIMSAttend;
extern std::string FusedMHA;
extern std::string PIMGEMV;
}  // namespace OperationType

namespace ParameterType {
extern std::string Weight;
extern std::string Bias;
}  // namespace ParameterType

class Model {
   public:
    Model(SimulationConfig config, std::string name);
    Model(const Model &model);

    void init_params();  // new
    Ptr<NPUTensor> find_tensor(std::string name);
    std::vector<Ptr<NPUTensor>> get_params(int layer_idx, std::string block_type,
                                           std::string operation_type);

    std::shared_ptr<Tensor> get_tensor(uint32_t id);
    void add_tensor(std::shared_ptr<Tensor> tensor);
    uint32_t add_tensor(uint32_t src_node, std::string name, std::vector<uint32_t> dims);
    // std::vector<std::shared_ptr<Tensor>> get_param(int layer, std::string block_type,
    //                                                std::string operation_type);

    void initialize_model(std::string input_names, std::vector<uint32_t> &input_dims,
                          MappingTable mapping_table);
    // void initialize_model(std::vector<std::string> &input_names,
    // std::vector<std::vector<uint32_t>> &input_dims);
    void finish_operation(uint32_t id);
    void finish_operation_tile(uint32_t id, Tile &tile);

    std::shared_ptr<Tensor> create_tensor(std::string name, std::vector<uint32_t> dims);
    std::shared_ptr<NPUTensor> create_weight(std::string name, std::vector<uint32_t> dims);

    std::shared_ptr<Operation> create_and_add_gpt_operation(Ops op_type, std::string name);
    std::shared_ptr<Operation> create_and_add_gpt_operation(Ops op_type, std::string name,
                                                            uint32_t attribute);
    std::shared_ptr<Operation> create_and_add_gpt_operation(Ops op_type, std::string name,
                                                            std::vector<uint32_t> attribute);
    // std::vector<std::shared_ptr<Tensor>> get_outputs(std::shared_ptr<Operation> op,
    //  std::vector<std::shared_ptr<Tensor>> inputs);

    std::shared_ptr<Operation> register_operation(std::shared_ptr<Operation>);
    void find_executable_node(std::shared_ptr<Tensor> tensor);

    std::string get_name() { return _name; }
    uint32_t get_id() { return _root_node_id; }
    std::shared_ptr<Tensor> get_input_tensor() { return _input_tensor; }
    std::vector<std::shared_ptr<Operation>> get_executable_operations();
    bool check_finish();
    uint64_t get_weight_size();
    addr_type get_weight_top_addr();  // Return the top address of the aligned weight + alignment.
    void log_model();

   private:
    std::string _name;
    std::string _input_name;
    std::vector<uint32_t> _input_dim;
    uint32_t _root_node_id;
    std::shared_ptr<Tensor> _input_tensor;
    std::map<uint32_t, std::shared_ptr<Operation>> _operation_map;
    std::map<uint32_t, std::shared_ptr<Tensor>> _tensor_map;
    robin_hood::unordered_map<std::string, Ptr<NPUTensor>> _wgt_map;
    std::vector<std::shared_ptr<Operation>> _executable_operations;
    SimulationConfig _config;
    bool _is_decode;
    uint64_t _wgt_size;  // bytes

    // for gpt
    uint32_t _num_batch;
    uint32_t _num_token;
    uint32_t _target_token;

    bool check_exist_in_executable(uint32_t id);

    void initialize_gpt2_params();  // old
    void gpt2_encode();
    void gpt2_decode();
    std::shared_ptr<Tensor> load_cache(uint32_t layer, std::string type);
    void log_inputs(std::vector<uint32_t> input_ids);
};

#endif