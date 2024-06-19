### Core Configuration
systolic_ws_128x128_dev.json

### Memory Configuration
|config|type|description|
|:---:|:---|:---|
|`dram_type`|string|Memory type. `dram`:HBM, `newton`:PIM, `neupims`:Dual row buffered PIM|
|`dram_freq`|int|DRAM frequency|
|`pim_config_path`|string|DRAM or PIM hardware specification|
|`dram_channels`|int|Number of DRAM channels|
|`dram_req_size`|int|DRAM access granularity (unit:Byte)|
|`dram_page_size`|int|DRAM row size (unit:Byte)|
|`dram_banks_per_ch`|int|Number of DRAM banks in channel|
|`pim_comp_coverage`|int|Number of multipliers per bank|

### Model Configuration
|config|type|description|
|:---:|:---|:---|
|`model_name`|string|Model name. It is just used to print log.|
|`model_params_b`|int|Number of model parameters (unit:B)|
|`vocab_size`|int|Vocabulary size (Unused)|
|`n_layer`|int|Number of layers (Unused)|
|`n_head`|int|Number of heads|
|`n_embd`|int|Embedding size|
|`n_tp`|int|Degree of Tensor parallelism|
|`n_pp`|int|Degree of Pipeline parallelism|

### System Configuration
|config|type|description|
|:---:|:---|:---|
|`run_mode`|string|`npu` or `npu+pim`|
|`sub_batch_mode`|boolean|Sub-batch interleaving mode on/off, sub-batch-on only available for neupims|
|`kernel_fusion`|boolean|Indicate whether kernel fusion is applied|
|`max_batch_size`|int|Maximum batch size|
|`max_active_reqs`|int|Maximum number of active requests|
|`max_seq_len`|int|Maximum sequence length|

### Request Traces
- (seq_len, pim_ch_idx) of each request
- channel load balancing algorithm: (rr, clb)
    - rr: round-robin algorithm
    - clb: greedy min-load bin packing algorithm
- Refer to `/trace-generator`. You can make your own trace corresponding the distribution of dataset (alpaca, share-gpt2)