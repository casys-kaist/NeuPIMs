#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../Common.h"
#include "../RequestGenerator.h"

uint32_t generate_rid();
class Client {
   public:
    Client(SimulationConfig config);
    void cycle();

    bool running();
    bool has_request();
    std::shared_ptr<InferRequest> pop_request();
    void receive_response(std::shared_ptr<InferRequest> response);

   private:
    SimulationConfig _config;
    uint32_t _cycles;
    uint32_t _last_request_cycle;
    uint32_t _need_wait_cycles;

    uint32_t _total_cnt;
    uint32_t _issued_cnt;
    uint32_t _completed_cnt;

    uint32_t _request_interval;  // send a request per (core_freq/qps) cycles
    std::queue<std::shared_ptr<InferRequest>> _waiting_queue;

    /* Random generate from poisson disribution (request arrival time)*/
    std::mt19937 _gen;
    std::poisson_distribution<> _distribution;

    /* Random generate from uniform d (input, output size) [min, max)*/
    int _imin;
    int _imax;
    int _omin;
    int _omax;
    int rand_input_size();
    int rand_output_size();
    bool _touch;
};
