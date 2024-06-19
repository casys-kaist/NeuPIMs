import gc
import json
import csv

import pandas as pd
import jsonlines
from transformers import AutoTokenizer

def flatten_data(j):
    input_toks = []
    output_toks = []
    for individual in j:
        conversations = individual['conversations']
        keys = conversations[0].keys()
        if 'from' in keys:
            key = 'from'
        elif 'user' in keys:
            key = 'user'
        else:
            assert False
        if 'value' in keys:
            value = 'value'
        elif 'text' in keys:
            value = 'text'
        else:
            assert False

        for conversation in conversations:
            if conversation[key] == 'human':
                input_toks.append(conversation[value])
            if conversation[key] == 'gpt':
                output_toks.append(conversation[value])
    
    return input_toks, output_toks 

def process_file(fname):
    if 'jsonl' in fname:
        with jsonlines.open(fname) as f:
            return flatten_data(f)
    elif 'json' in fname:
        with open(fname, 'r') as f:
            j = json.load(f)
            return flatten_data(j)
    elif 'parquet' in fname:
        df = pd.read_parquet(fname, engine='pyarrow')
        df = df.to_dict()
        df_len = len(df['instruction'].keys())
        return [df['instruction'][i] for i in range(df_len)], [df['output'][i] for i in range(df_len)]
    else:
        raise ValueError(f"{fname} file's file extension not supported.")


def list_len_of_tokenized_sentences(sentences):
    tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    token_length_in_sentences = []
    character_length_in_sentences = []
    for sentence in sentences:
        tokenized_sentence = tokenizer(sentence, return_tensors='np')['input_ids']
        token_length_in_sentences.append(len(tokenized_sentence[0]))
        character_length_in_sentences.append(len(sentence))

    return token_length_in_sentences, character_length_in_sentences

if __name__=="__main__":
    #ifnames = ["share-gpt/1.json", "share-gpt/2.json"]
    #ofname = "share-gpt/status.tsv"
    #ifnames = ["alpaca/1.parquet"]
    #ofname = "alpaca/status.tsv"
    ifnames = ["share-gpt2/1.jsonl"]
    ofname = "share-gpt2-status.tsv"

    input_toks = []
    input_chars = []
    output_toks = []
    output_chars = []

    for fname in ifnames:
        iss, oss = process_file(fname)
        gc.collect()
        toklen, charlen = list_len_of_tokenized_sentences(iss)
        input_toks.extend(toklen)
        input_chars.extend(charlen)
        gc.collect()
        toklen, charlen = list_len_of_tokenized_sentences(oss)
        output_toks.extend(toklen)
        output_chars.extend(charlen)
        gc.collect()

    print(f"number of parsed conversations : input({len(input_chars)}), output({len(output_chars)})")
    print(f"average parsed input characters: {sum(input_chars)/len(input_chars)}")
    print(f"average parsed output characters: {sum(output_chars)/len(output_chars)}")
    print(f"average input toks: {sum(input_toks)/len(input_toks)}")
    print(f"average output toks: {sum(output_toks)/len(output_toks)}")

    with open(ofname, 'w', encoding='utf-8', newline='') as f:
        cw = csv.writer(f, delimiter='\t')
        cw.writerow(['input_toks', 'output_toks'])
        for row in zip(input_toks, output_toks):
            cw.writerow(row)

#        print(input_toks[i], input_chars[i])
#        print(output_toks[i], output_chars[i])

