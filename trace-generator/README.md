# trace generator

1. `tokenize_stat.py` creates tsv files, consists of input_toks, output_toks pair from real data. Note that real data is not uploaded due to the filesize constraints.
2. `get_distributions.py` creates traces from the tsv files generated at stage 1.

`channel_load_balancing.py` is a reference code.