import csv
import random
import math
import time
import copy

def ceil(a, b):
    return (a+b-1) // b


def GB2B(a):
    return int(a*2**30)


def MB2B(a):
    return int(a*2**20)


def KB2B(a):
    return int(a*2**10)


def import_tsv(fname):
    num_otk_zero = 0
    itks = []
    otks = []
    with open(fname, 'r') as f:
        tsv = csv.reader(f, delimiter='\t')
        row_index = -1
        for row in tsv:
            # skip the first column names
            row_index += 1
            if row_index == 0:
                continue
            # if int(row[1]) == 0 or int(row[0]) + int(row[1]) > ModelConfig.max_seq_len:
            if int(row[1]) == 0:
                num_otk_zero += 1
                row_index -= 1
                continue
            itks.append(int(row[0]))
            otks.append(int(row[1]))

        avg_itks = sum(itks) / len(itks)
        avg_otks = sum(otks) / len(otks)

        print(f"imported dataset {fname}. {row_index} rows")
        print(f"excluding {num_otk_zero} rows")
        print(f"average input tokens {avg_itks} / average output tokens {avg_otks}")

    return itks, otks


class MemoryConfig:
    num_channels = 32
    dram_page_size = 512
    dram_banks_per_ch = 32
    gwrite_latency = 100
    gemv_latency = 184
    pim_tile_size_b = dram_page_size * dram_banks_per_ch * 2


class ModelConfig:
    max_seq_len = 2048

    @classmethod
    def init(cls, model_size=7, n_tp=4, n_pp=1):
        if model_size == 7:
            # 7B
            cls.num_params = 7
            cls.E = 4096
            cls.nh = 32
            cls.nl = 32
        elif model_size == 13:
            # 13B
            cls.num_params = 13
            cls.E = 5120
            cls.nh = 40
            cls.nl = 40
        elif model_size == 30:
            # 30B
            cls.num_params = 30
            cls.E = 7168
            cls.nh = 56
            cls.nl = 48
        elif model_size == 175:
            # 175B
            cls.num_params = 175
            cls.E = 12288
            cls.nh = 96
            cls.nl = 96
        else:
            assert 0

        cls.n_pp = n_pp
        cls.n_tp = n_tp
        cls.dk = cls.E / cls.nh
    

def num_tiles_per_channel():
    # precision 
    model_params_per_ch_b = GB2B(ModelConfig.num_params / ModelConfig.n_pp * 2) // ModelConfig.n_tp
    available_kv_b = GB2B(MemoryConfig.num_channels) - model_params_per_ch_b
    available_tiles = ceil(available_kv_b, MemoryConfig.pim_tile_size_b)
    return available_tiles // MemoryConfig.num_channels


class Request:
    def __init__(self, input_tok, output_tok):
        self.input_tok = input_tok
        self.output_tok = output_tok
        self.generated_tok = 0

        self.is_allocated = False
        self.channel = 0
    

    def __lt__(self, other):
        return self.get_seq_len() < other.get_seq_len()


    def set_channel(self, channel):
        assert self.is_allocated == False
        self.is_allocated = True
        self.channel = channel


    def get_seq_len(self):
        return min(self.input_tok + self.generated_tok, ModelConfig.max_seq_len)


    def is_done(self):
        return self.generated_tok == self.output_tok


    def increment(self):
        assert self.generated_tok < self.output_tok
        self.generated_tok += 1


    # return unit : tiles
    def tile_used(self):
        effective_e = ModelConfig.E // ModelConfig.n_tp

        key_period = MemoryConfig.dram_banks_per_ch
        value_period = MemoryConfig.dram_page_size

        key_pages = ceil(self.get_seq_len(), key_period)
        value_pages = ceil(self.get_seq_len(), value_period)

        key_tiles = key_pages * ceil(effective_e, value_period)
        value_tiles = value_pages * ceil(effective_e, key_period)

        #print(self.get_seq_len(), key_tiles, value_tiles)

        return (key_tiles + value_tiles) * ModelConfig.nl // ModelConfig.n_pp
       

    def estimate_latency(self):
        effective_e = ModelConfig.E / ModelConfig.n_tp
        latency = 0

        chunks = math.ceil(effective_e / MemoryConfig.dram_page_size)
        tiles = math.ceil(self.get_seq_len() / MemoryConfig.dram_banks_per_ch)
        latency += chunks * MemoryConfig.gwrite_latency
        latency += chunks * tiles * MemoryConfig.gemv_latency

        chunks = math.ceil(self.get_seq_len() / MemoryConfig.dram_page_size) * ModelConfig.nh
        tiles = math.ceil(ModelConfig.dk / MemoryConfig.dram_banks_per_ch)
        latency += chunks * MemoryConfig.gwrite_latency
        latency += chunks * tiles * MemoryConfig.gemv_latency

        return latency


    def __repr__(self):
        return f"itk: {self.input_tok} / otk: {self.output_tok} / generated: {self.generated_tok}"


# algorithm should be "rr" or "clb"
class BatchedRequest:
    def __init__(self, dataset, num_channels=32, algorithm="rr", max_batch_size=256):
        self.initiated_requests = []
        for itk, otk in zip(dataset[0], dataset[1]):
            self.initiated_requests.append(Request(itk, otk))

        self.ongoing_requests = []
        # self.finished_requests = []
        self.queued_requests = []

        self.cycle_count = 0

        self.num_channels = num_channels

        if algorithm not in ["rr", "clb", "rrn"]:
            raise NotImplemented(f"{self.algorithm} not implemented")
        self.algorithm = algorithm
        self.rr_ch_idx = 0

        self.max_batch_size = max_batch_size

        self.cycle(0)
        

    def _toks_per_channel(self):
        toks_per_channel = [0] * self.num_channels

        for req in self.ongoing_requests:
            toks_per_channel[req.channel] += req.get_seq_len()

        return toks_per_channel


    def _loads_per_channel(self):
        loads_per_channel = [0] * self.num_channels

        for req in self.ongoing_requests:
            loads_per_channel[req.channel] += req.estimate_latency()

        return loads_per_channel


    def _tiles_left_per_channel(self):
        tiles_left_per_channel = [num_tiles_per_channel()] * self.num_channels

        for req in self.ongoing_requests:
            tiles_left_per_channel[req.channel] -= req.tile_used()

        for num_tiles in tiles_left_per_channel:
            if num_tiles < 0:
                print(tiles_left_per_channel)
                print(self._toks_per_channel())
                assert 0

        return tiles_left_per_channel
    

    def _move_init_reqs_to_queued(self):
        assert len(self.queued_requests) == 0 
        n = self.max_batch_size - len(self.ongoing_requests)
        assert n >= 0 

        # this deepcopy is shit
        # self.queued_requests = copy.deepcopy(random.choices(self.initiated_requests, k=n))

        for req in random.choices(self.initiated_requests, k=n):
            self.queued_requests.append(copy.deepcopy(req))


    def _rr_naive(self):
        return self.rr_ch_idx


    def _rr_get_ch_idx_first_fit(self, req):
        tiles_left_per_channel = self._tiles_left_per_channel()
        for i in range(MemoryConfig.num_channels):
            ch_idx = (self.rr_ch_idx + i) % MemoryConfig.num_channels
            if req.tile_used() <= tiles_left_per_channel[ch_idx]:
                return ch_idx

        assert "out of memory" and 0
        return -1

        
    def _channel_load_balancing(self):
        assert len(self.queued_requests) + len(self.ongoing_requests) == self.max_batch_size
        loads_per_channel = self._loads_per_channel()
        for req in reversed(sorted(self.queued_requests)):
            min_idx = min(range(self.num_channels), key=lambda i: loads_per_channel[i])
            loads_per_channel[min_idx] += req.estimate_latency()
            req.set_channel(min_idx)
            self.ongoing_requests.append(req)

        # to check the validity
        self._tiles_left_per_channel()

        
    def _round_robin(self):
        assert len(self.queued_requests) + len(self.ongoing_requests) == self.max_batch_size
        for req in self.queued_requests:
            ch_idx = self._rr_get_ch_idx_first_fit(req)
            req.set_channel(ch_idx)
            self.rr_ch_idx = (ch_idx+1) % MemoryConfig.num_channels
            self.ongoing_requests.append(req)

    def _round_robin_naive(self):
        assert len(self.queued_requests) + len(self.ongoing_requests) == self.max_batch_size
        for req in self.queued_requests:
            ch_idx = self._rr_naive()
            req.set_channel(ch_idx)
            self.rr_ch_idx = (ch_idx+1) % MemoryConfig.num_channels
            self.ongoing_requests.append(req)


    def cycle(self, num_iter=1):
        self.cycle_count += num_iter
        indexes_to_remove = []
        for _ in range(num_iter):
            for i, request in enumerate(self.ongoing_requests):
                if request.is_done():
                    continue
                request.increment()
                if request.is_done():
                    indexes_to_remove.append(i)
        for i in reversed(sorted(indexes_to_remove)):
            req = self.ongoing_requests.pop(i)
            assert req.is_done()

        # to check the validity
        self._tiles_left_per_channel()
       
        self._move_init_reqs_to_queued()
        if self.algorithm == "rr":
            self._round_robin()
        elif self.algorithm == "rrn":
            self._round_robin_naive()
        elif self.algorithm == "clb":
            self._channel_load_balancing()

        # is done at _move_init_reqs_to_queued()
        self.queued_requests = []


    def snapshot(self):
        return self.cycle_count, self.ongoing_requests 

def red_string(s):
    return f"\033[31m{s}\033[0m"

# format float
def ff(f):
    digits = 3 
    
    # Calculate the exponent
    exponent = int(math.floor(math.log10(abs(f))))
    f /= 10**exponent
    
    # Format the output
    return f"{f:.{digits - 1}f}e{exponent}" if exponent <= 5 else red_string(f"{f:.{digits - 1}f}e{exponent}")


def write_output(fname, idx, ongoing_requests):
    fname += f"{idx}.csv"
    
    data = [["seq_len", "ch_idx"]]
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        data.extend([[r.get_seq_len(), r.channel] for r in ongoing_requests])
        writer.writerows(data)


def generate_traces(dataset_name, batch_size, algorithm):
    fname = f"{dataset_name}/stats.tsv"

    dataset = import_tsv(fname)
    batched_request = BatchedRequest(dataset=dataset, num_channels=MemoryConfig.num_channels, algorithm=algorithm, max_batch_size=batch_size)

    cycle_unit = 10

    loop = 50000
    log_interval = 50
    average_sequence_length = 0
    latency_diff = 0


    print(f"available number of tiles per channel {num_tiles_per_channel()}")
    print(f"looping {cycle_unit} * {loop} times ..")

    time.sleep(5)

    # add {idx}.csv at last
    ofname = f"traces/{dataset_name}-bs{batch_size}-ms{ModelConfig.num_params}B-tp{ModelConfig.n_tp}-pp{ModelConfig.n_pp}-{algorithm}-"
    warmup_loop = 10000
    num_output_log = 10
    output_idx = 0

    for i in range(loop):
        batched_request.cycle(cycle_unit)
        loads_per_channel = batched_request._loads_per_channel()
        max_latency = max(loads_per_channel)
        min_latency = min(loads_per_channel)

        latency_diff += max_latency - min_latency

        if i % log_interval == 0:
            num_cycles, ongoing_requests = batched_request.snapshot()
            assert len(ongoing_requests) == batch_size
            sequences = [request.get_seq_len() for request in ongoing_requests]
            if average_sequence_length == sum(sequences) / len(sequences):
                print(sequences)
                print([req.is_done() for req in ongoing_requests])


            if i > warmup_loop:
                write_output(ofname, output_idx, ongoing_requests)
                output_idx += 1
                if output_idx == num_output_log:
                    return


            average_sequence_length = sum(sequences)/len(sequences)
            # print(f"cycle count {num_cycles}")
            # print(f"average sequence length: {ff(sum(sequences)/len(sequences))}")
            # print(f"avg latency diff: {ff(latency_diff / log_interval)}")
            # print(f"minimum tiles left: {min(batched_request._tiles_left_per_channel())}")
            latency_diff = 0


if __name__=="__main__":
    # dataset_name = "share-gpt"
    # dataset_name = "alpaca"
    # batch_size = 512
    # algorithm = "clb"
    # model_size = 7

    batch_size = 256
    algorithm = "clb"

    for dataset_name in ["alpaca", "share-gpt"]:
        ModelConfig.init(7, 4, 1)
        generate_traces(dataset_name, batch_size, algorithm)


    for dataset_name in ["share-gpt"]:
            # model_size = 7
            # for n_tp, n_pp in [(4, 1), (2, 2)]:
            #     ModelConfig.init(model_size, n_tp, n_pp)
            #     generate_traces(dataset_name, batch_size, algorithm)

            # model_size = 13
            # for n_tp, n_pp in [(8, 1), (4, 2), (2, 4)]:
            #     ModelConfig.init(model_size, n_tp, n_pp)
            #     generate_traces(dataset_name, batch_size, algorithm)

            # model_size = 30
            # for n_tp, n_pp in [(16, 1), (8, 2), (4, 4)]:
            #     ModelConfig.init(model_size, n_tp, n_pp)
            #     generate_traces(dataset_name, batch_size, algorithm)

            model_size = 175
            for n_tp, n_pp in [(16, 4), (8, 8)]:
                ModelConfig.init(model_size, n_tp, n_pp)
                generate_traces(dataset_name, batch_size, algorithm)

