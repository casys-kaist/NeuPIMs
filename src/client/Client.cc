#include "Client.h"

Client::Client(SimulationConfig config)
    : _config(config),
      _cycles(0),
      _last_request_cycle(0),
      _issued_cnt(0),
      _completed_cnt(0),
      _need_wait_cycles(0) {  // 30
    // arguments:
    // - request_interval (mean),
    // - total number of requests
    // - request size (input,output)

    std::random_device rd;
    std::mt19937 gen(rd());
    _gen = gen;

    // todo: get from config
    _imin = 10;
    _imax = 256;
    _omin = 2;
    _omax = 4;

    uint32_t answer_index = 1;
    RequestGenerator::init(config.request_dataset_path, answer_index);

    // _total_cnt = _config.request_total_cnt;
    _total_cnt = RequestGenerator::get_total_req_cnt();
    spdlog::info("Client total request cnt: {}", _total_cnt);
    _request_interval = _config.request_interval;

    std::poisson_distribution<> d(_request_interval);
    _distribution = d;
    _touch = false;
}

int Client::rand_input_size() { return rand() % (_imax - _imin) + _imin; }
int Client::rand_output_size() { return rand() % (_omax - _omin) + _omin; }

void Client::cycle() {
    uint32_t idle_cycles = _cycles - _last_request_cycle;
    // return; // FIXME: comment
    // FIXME: change while to if
    while (!_touch) {
        // todo: send request to scheduler
        uint32_t rid = generate_rid();

        // TODO: from benchmark dataset
        // uint32_t input_size = rand_input_size();  // 10;
        // uint32_t output_size = rand_output_size();  // 2;
        std::pair<uint32_t, uint32_t> input_output_size;
        if (RequestGenerator::has_data()) {
            input_output_size = RequestGenerator::get_qa_length();
        } else {
            spdlog::info("RequestGenerator has no data!");
            _touch = true;
            break;
            // exit(-1);
        }
        uint32_t input_size = input_output_size.first;
        uint32_t output_size = 1;  // input_output_size.second;  // 1;
        uint32_t channel = input_output_size.second;
        std::shared_ptr<InferRequest> request =
            std::make_shared<InferRequest>(InferRequest{.id = rid,
                                                        .arrival_cycle = _cycles,
                                                        .completed_cycle = 0,
                                                        .input_size = input_size,
                                                        .output_size = output_size,
                                                        .is_initiated = false,
                                                        .generated = 0,
                                                        .channel = channel});
        _waiting_queue.push(request);

        _issued_cnt++;
        spdlog::info("issued cnt:{} total: {}", _issued_cnt, _total_cnt);
        spdlog::info("idle:{}, need_wait:{}", idle_cycles, _need_wait_cycles);
        _last_request_cycle = _cycles;

        // set next request interval
        spdlog::info("xyz");
        _need_wait_cycles = 0;  // _distribution(_gen);
        spdlog::info("xyz");

        spdlog::info("Client Request Departure!! now:{} next wait: {}", _cycles, _need_wait_cycles);
        spdlog::info("Request #{}, input size:{}, output size:{}", rid, input_size, output_size);
    }

    _cycles++;

    if (_completed_cnt == _total_cnt) {
        spdlog::info("Client completed!");
        exit(-1);
    }
}

bool Client::running() {
    return _completed_cnt < _total_cnt;  // FIXME: comment
    return false;
}

bool Client::has_request() { return !_waiting_queue.empty(); }

std::shared_ptr<InferRequest> Client::pop_request() {
    std::shared_ptr<InferRequest> top = _waiting_queue.front();
    _waiting_queue.pop();
    return top;
}

void Client::receive_response(std::shared_ptr<InferRequest> response) {
    ast(response->generated == response->output_size);

    response->completed_cycle = _cycles;
    _completed_cnt++;

    // spdlog::info("Receive response! spend_cycles: {}",
    //              response->completed_cycle - response->arrival_cycle);

    // todo stat.
    // delete response;
}

uint32_t generate_rid() {
    static uint32_t rid{0};
    return rid++;
}
