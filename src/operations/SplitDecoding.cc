#include "SplitDecoding.h"

// split and concat 1,3E to nh,1,dk(2D) / nh,dk,T+1 (tp,KV) / nh,T+1,dk (KV)
SplitDecoding::SplitDecoding(std::string name, std::pair<Ptr<BTensor>, Ptr<BTensor>> kv_cache,
                             bool is_initiated)
    : Operation(name) {
    _inputs.resize(3);
    _inputs[1] = kv_cache.first;
    _inputs[2] = kv_cache.second;
    _is_initiated = is_initiated;
}

/**
 * QKVSplit function called at incremental phase
 * for key and value cache, add_token method is called to increment the token count
 *  input:
 *      QKV concatenated (1,3E), k_cache (nh,dk,l), v_cache(nh,l,dk)
 *  output:
 *      Q(nh,l,dk)[NPUTensor2D], K(nh,dk,l)[NPUTensorKV], V(nh,l,dk)[NPUTensorKV]
 */
std::vector<Ptr<BTensor>> SplitDecoding::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];

    _outputs.resize(3);

    Ptr<NPUTensor> input = std::static_pointer_cast<NPUTensor>(_inputs[0]);

    uint32_t nh = Config::global_config.model_n_head / Config::global_config.n_tp;
    uint32_t E = Config::global_config.model_n_embd;
    uint32_t dk = Config::global_config.model_n_embd / Config::global_config.model_n_head;

    // Perform write operations based on the DRAM addresses received from split and concat breq->req->cacheload.
    // _inputs[1]->add_token(); 
    // _inputs[2]->add_token();

    ast(input->get_dims().back() == 3 * E);
    spdlog::info("(SplitDecoding) inputs[0] dimension: {}", _inputs[0]->get_dims()[0]);

    uint32_t l = _is_initiated ? 1 : _inputs[0]->get_dims()[0];
    std::vector<uint32_t> output_dim = {nh, l, dk};
    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", output_dim, NPUTensorBufType::ACT, false);

    _outputs[1] = _inputs[1];
    _outputs[2] = _inputs[2];

    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "SplitDecoding",
        .operation_id = _id,
        .batch = 1,
        .skip = true,
    });

    return _outputs;
}
