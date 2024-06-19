import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

commands = ['gemv', 'write', 'read']

with open('dramsim3.json', 'r') as f:
    data = json.load(f)

channel = "0"
ch_stat = data[channel]
for cmd in commands:
    latencys = []
    counts = []
    avg_latency = "None"
    for key in ch_stat:
        if (f"average_{cmd}_latency" in key):
            avg_latency = ch_stat[key]
        elif (f"{cmd}_latency[" in key):
            cnt = ch_stat[key]
            key = key.replace(f"{cmd}_latency[", "")
            key = key.replace("]", "")
            start, end = key.split('-')
            
            if end == '0': continue
            start = int(start)
            
            latencys.append(start)
            counts.append(cnt)
    # todo: bug here, latencys and counts are not sorted
    x = np.arange(len(latencys))
    plt.xticks(x, latencys)
    plt.bar(x, counts)
    plt.title(f"{cmd} latency, average: {avg_latency}")
    plt.show()