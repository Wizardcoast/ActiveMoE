# Copyright (c) 2022, HKGAI CORPORATION. All rights reserved.

import json
import os
import sys
from copy import deepcopy

from tqdm import tqdm

# fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from tools.preprocess_data import Encoder
from tools.preprocess_data import get_args as build_args

extra_args = [
            "--tokenizer-model",
            "/workspace/megatron/amber.tokenizer.model",
            "--tokenizer-type",
            "SentencePieceTokenizer",
            "--vocab-file",
            "/workspace/dataset/llama.vocab.json",
            "--append-eod",
            "--workers",
            "1",
            "--log-interval",
            "1",
            "--json-keys",
            "token_ids",
        ]

try:
    input_amber_jsonl = sys.argv[2]
except IndexError:
    input_amber_jsonl = "/workspace/dataset/filtered_amber_full.jsonl"
    print(f"INFO: filtered_amber_full.jsonl contails 614268926 lines")
print(f"INFO: detokenizing {input_amber_jsonl}")
output_jsonl = input_amber_jsonl.replace(".jsonl", "_detok.jsonl")

sys.argv = [sys.argv[0], "--input", input_amber_jsonl, "--output-prefix", None,] + extra_args

encoder = Encoder(build_args())
encoder.initializer()
detok = encoder.tokenizer.detokenize

with open(input_amber_jsonl, "r") as reader:
    with open(output_jsonl, "w") as writer:
        for line in tqdm(reader):
            if not line.startswith("{\"token_ids\":"):
                continue
            line = json.loads(line)
            token_ids = line.pop("token_ids")
            line["text"] = detok(token_ids)
            writer.write(json.dumps(line) + "\n")

print(f"INFO: detokenized jsonl is saved at {output_jsonl}")
print(f"INFO: done")
