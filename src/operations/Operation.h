#pragma once

#include "../Common.h"
#include "../Mapping.h"
#include "../Tensor.h"
#include "../tensor/BTensor.h"

class Model;
class OpParser;

enum class Ops {
    LayerNorm,  // start of the FFN and attn layer
    MatMul,     // QKV generation, attention score calculation, rescoring value,
                // projection, FFN * 2
    Split,
    // Mask,
    Softmax,
    Add,  // for Residual Connection
    Gelu,
    Reshape,
    Transpose,
    Concat,
    /* PIM fused operation */
    GEMV_Softmax,  // L
    GEMV_Add,      // A
};

// Graph Node
class Operation : public std::enable_shared_from_this<Operation> {
   public:
    Operation(MappingTable mapping_table);
    Operation(std::string name);
    Operation(const Operation &operation);

    static void initialize(SimulationConfig config) { _config = config; }

    virtual void set_finish();

    virtual std::string get_name() { return _name; }
    virtual std::string get_optype() { return _optype; }
    virtual uint32_t get_id() { return _id; }
    virtual uint32_t num_inputs() { return _inputs.size(); }
    virtual std::vector<std::shared_ptr<BTensor>> get_inputs() { return _inputs; }
    virtual uint32_t num_outputs() { return _outputs.size(); }
    virtual std::vector<std::shared_ptr<Operation>> get_child_nodes();
    virtual std::deque<Tile> get_tiles();
    virtual bool check_executable();
    virtual std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);

    void set_as_parent_tensor(std::vector<Ptr<BTensor>> inputs);
    bool check_finish() { return _finish; };

    void reduce_tile(Tile &tile);
    OperationStat get_stat() { return _stat; };
    std::string repr();

   protected:
    virtual void initialize_instructions(Tile &tile, Mapping mapping) {}

    static const uint32_t _NO_OPERAND = 0;
    static const uint32_t _INPUT_OPERAND = 100;
    static const uint32_t _OUTPUT_OPERAND = 200;
    uint32_t _id;
    std::string _name;
    std::string _optype;
    static SimulationConfig _config;
    std::vector<Ptr<BTensor>> _inputs;
    std::vector<Ptr<BTensor>> _outputs;
    std::map<std::string, std::string> _attributes;
    std::deque<Tile> _tiles;
    std::vector<std::vector<std::vector<addr_type>>> _weight_addrs;
    std::vector<std::vector<std::vector<std::vector<addr_type>>>> _input_addrs;
    std::vector<std::vector<std::vector<std::vector<addr_type>>>> _output_addrs;

    int Ndim;    // Batch dimension of activation tensor (commoly 0)
    int Hdim;    // Height dimension of activation tensor
    int Wdim;    // Width dimension of activation tensor
    int Cdim;    // Channel dimension of activation tensor
    int Cdim_w;  // Channel dimension of weight tensor
    int Mdim;    // Output channel dimension of weight tensor
    int Sdim;    // Height dimension of weight tensor
    int Rdim;    // Width dimension of weight tensor

    OpStat _op_stat;
    OperationStat _stat;

    bool _finish;
    friend Model;
    addr_type _spad_addr;
    addr_type _acc_spad_addr;

    std::pair<addr_type, uint32_t> allocate_sram_addr(uint32_t size, bool accum);
};