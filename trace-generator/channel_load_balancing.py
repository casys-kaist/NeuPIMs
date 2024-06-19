import math
from functools import reduce

# memory spec parameters
_dram_page_size = 512
_dram_banks_per_ch = 32
_gwrite_latency = 100
_gemv_latency = 184

# model spec
E = 4096
n_tp = 4
_nh = 32
_dk = E/_nh


def estimate_mha_latency(seq_len):
    _effective_e = E / n_tp
    # calculate MHA latency with sequence length
    latency = 0

    # key * query
    chunks = math.ceil(_effective_e / _dram_page_size)
    tiles = math.ceil(seq_len / _dram_banks_per_ch)
    latency += chunks * _gwrite_latency
    latency += chunks * tiles * _gemv_latency

    # logit * value
    chunks = math.ceil(seq_len / _dram_page_size) * _nh
    tiles = math.ceil(_dk / _dram_banks_per_ch)
    latency += chunks * _gwrite_latency
    latency += chunks * tiles * _gemv_latency
    return latency

def sum_load(seqlens):
    load = reduce(lambda acc, seq_len: acc + estimate_mha_latency(seq_len), seqlens, 0)
    return load

# distribute requests takes (1)new requests, (2)previous channel distributions, (3)total channel size
# channel에 있는 seqlen에 대한 load들의 합을 구함
# channel load가 가장 작은 idx를 선택 후,
def distribute_requests(new_seq_lens, channels_seqlen, k):
    # Create a list to store the sum of values in each channel
    channels_load = [sum_load(seqlens) for seqlens in channels_seqlen]
    
    for element in sorted(new_seq_lens, reverse=True):
        min_sum_index = min(range(k), key=lambda i: channels_load[i])
        channels_seqlen[min_sum_index].append(element)
        channels_load[min_sum_index] += estimate_mha_latency(element)
        
    return channels_seqlen 

# Example usage:
request_lengths = [5, 8, 3, 2, 7]
channels_seqlen = [[4, 6], [8, 1], [3, 9]]
k = 3

result = distribute_requests(request_lengths, channels_seqlen, k)
print(result)
print([sum_load(seqlens) for seqlens in channels_seqlen])
