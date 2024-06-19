#include "Attention.h"
// Fused Operations QKV gen + Multi-head Attention
Attention::Attention(std::string name, std::shared_ptr<BatchedRequest> breq) : Operation(name) {
    // requests
    _breq = breq;
    _N = _breq->get_num_rows();
    _seq_lens = _breq->get_num_rows_breakdown();
    _batch_size = _breq->get_num_reqs();

    // model config
    _E = _config.model_n_embd;
    _nh = _config.model_n_head / _config.n_tp;
    _dk = _config.model_n_embd / _config.model_n_head;
}

std::vector<Ptr<BTensor>> Attention::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    return _outputs;
}

void Attention::initialize_tiles() {
    uint32_t num_tiles = ceil((double)_nh / _k);
    uint32_t covered_heads = 0;
    int k;
    for (auto i = 0; i < num_tiles; i++) {
        if (i == num_tiles - 1)
            k = _nh - covered_heads;
        else
            k = _k;
        _tiles.push_back(initialize_instructions(k));
        covered_heads += k;
    }
}

Tile Attention::initialize_instructions(int k) {
    // k: num heads in this tile
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };

    // LOAD (Input)

    for (auto h_idx = 0; h_idx < k; h_idx++) {
    }
    // LOAD (Wq for k heads)
    // GEMM (Wq * Input = Q for k heads * batch_size)
    // MOVOUT (Q)

    // LOAD (Wk for k heads)
    // GEMM (Wk * Input = K)
    // MOVOUT (K)

    // PIM_GEMV (Q*K = L)
    // SOFTMAX (L -> Ls)
    // MOVOUT (Ls)

    // LOAD (Wv for k heads)
    // GEMM (Wv * Input = V)
    // MOVOUT (V)

    // PIM_GEMV (Ls*V = A)
    // MOVOUT (A)

    return tile;
}

void Attention::calculate_loops() {
    int layer_idx = 0;  // todo: modify real layer

    uint32_t weight_per_head = 3 * _dk * _E;
    uint32_t qkv_per_head = 3 * _dk * _N;
    uint32_t attend_per_head = _N * _dk;

    uint32_t logit_per_head = 0;
    for (auto i = 0; i < _batch_size; i++) {
        bool incr = _breq->is_initiated(i);
        uint32_t pi =
            _breq->get_cache(layer_idx, i).first->get_dims()[0];  // previous input seq len
        if (incr) {
            logit_per_head += pi;
        } else {
            logit_per_head += _seq_lens[i] * _seq_lens[i];
        }
    }
    logit_per_head *= 2;

    uint32_t input_size = _N * _E;

    uint32_t sram_capacity = _config.spad_size KB / 2;
    uint32_t sram_size = sram_capacity / _config.precision;

    uint32_t remain_size = sram_size - input_size;

    uint32_t need_sram_per_head = weight_per_head + qkv_per_head + attend_per_head + logit_per_head;

    _k = remain_size / need_sram_per_head;

    spdlog::info("k: {}", _k);
}

uint32_t Attention::sram_size_needed() { return 0; }