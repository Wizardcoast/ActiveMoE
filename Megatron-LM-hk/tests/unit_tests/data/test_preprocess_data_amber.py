# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import tempfile

import nltk
import requests

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.tokenizer.gpt2_tokenization import (
    PRETRAINED_MERGES_ARCHIVE_MAP,
    PRETRAINED_VOCAB_ARCHIVE_MAP,
)
from tools.merge_datasets import main as merge_main
from tools.preprocess_data_amber import Encoder
from tools.preprocess_data_amber import get_args as build_args
from tools.preprocess_data_amber import main as build_main

__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB = (
    "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
)


def dummy_jsonl(odir):
    # amber sample 
    amber_sample_path = "/workspace/dataset/filtered_sampled_amber.jsonl"
    list_amber_sample = []
    sample_cnt = 1000
    print(f"INFO: amber dataset sample (100k lines) is used for testing, located at {amber_sample_path}")
    print(f"INFO: creating dummy dataset with {sample_cnt} lines")
    with open(amber_sample_path) as reader:
        for line in reader:
            list_amber_sample.append(json.dumps({"text": line}) + "\n")
            sample_cnt -= 1
            if sample_cnt == 0:
                break
    with open(os.path.join(odir, "amber_dummy.jsonl"), "w") as writer:
        writer.writelines(list_amber_sample)
    
    
def build_datasets(idir, odir, extra_args=[]):
    for name in os.listdir(idir):
        sys.argv = [
            sys.argv[0],
            "--input",
            os.path.join(idir, name),
            "--output-prefix",
            os.path.join(odir, os.path.splitext(name)[0]),
        ] + extra_args
        build_main()


def merge_datasets(idir):
    sys.argv = [sys.argv[0], "--input", idir, "--output-prefix", os.path.join(idir, "merge")]
    merge_main()


def do_test_preprocess_data(temp_dir, extra_args=[]):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws = os.path.join(temp_dir, "sample_raws")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_raws)
    os.mkdir(path_to_data)

    # create the dummy resources
    dummy_jsonl(path_to_raws)

    # build the datasets
    build_datasets(
        path_to_raws, path_to_data, extra_args=extra_args,
    )

    # merge the datasets
    merge_datasets(path_to_data)

    sys.argv = [sys.argv[0], "--input", None, "--output-prefix", None,] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        # for option in ["decode", "detokenize"]:
        for option in ["detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot `decode` or `detokenize`.")

    merged_index = 0
    merged_dataset = MMapIndexedDataset(os.path.join(path_to_data, "merge"))

    # sorted to ensure ordering matches merged dataset
    basenames = sorted(
        [
            name
            for name in os.listdir(path_to_data)
            if name.endswith(".idx") and not name.startswith("merge")
        ]
    )

    # index into the merged document index
    merged_doc_index_index = 0

    for basename in basenames:
        realpath_raw = f"{os.path.join(path_to_raws, '_'.join(basename.split('_')[:-2]))}.jsonl"
        realpath_doc = os.path.join(path_to_data, basename.split(".")[-2])

        dataset_index = 0
        dataset = MMapIndexedDataset(realpath_doc)

        merged_doc_idx = merged_dataset.document_indices[
            merged_doc_index_index : merged_doc_index_index + len(dataset.document_indices)
        ]
        merged_doc_idx = merged_doc_idx - merged_doc_idx[0]

        assert (
            dataset.document_indices == merged_doc_idx
        ).all(), f"ERROR: {basename.split('_')[:-2]}: merged dataset document indices mismatch"

        merged_doc_index_index += len(dataset.document_indices) - 1

        with open(realpath_raw, "rt") as reader:
            for json_line in reader:
                toks = encoder.encode_no_tokenizer(json_line)[0]["text"]

                # TODO: add skip logic for error lines
                if len(toks) == 0:
                    continue

                raw = tokens_to_string(toks)

                processed_toks = []
                while len(processed_toks) < len(toks):
                    processed_toks.extend(dataset[dataset_index].tolist())
                    dataset_index += 1
                processed = tokens_to_string(processed_toks)

                assert (
                    raw == processed
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents do not match"

                merged_toks = []
                while len(merged_toks) < len(toks):
                    merged_toks.extend(merged_dataset[merged_index].tolist())
                    merged_index += 1
                merged = tokens_to_string(merged_toks)

                assert (
                    raw == merged
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents do not match"

        print(
            f"INFO: {''.join(basename.split('_')[:-2])}: raw, processed, and merged documents match!"
        )

    print("INFO: Success!")


def gpt2_vocab(odir):
    path = os.path.join(odir, "vocab.json")
    with open(path, "wb") as writer:
        writer.write(requests.get(PRETRAINED_VOCAB_ARCHIVE_MAP['gpt2']).content)
    return path


def gpt2_merge(odir):
    path = os.path.join(odir, "merge.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(PRETRAINED_MERGES_ARCHIVE_MAP['gpt2']).content)
    return path


def test_preprocess_data_gpt():
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
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

        do_test_preprocess_data(temp_dir, extra_args=gpt_args)


def bert_vocab(odir):
    path = os.path.join(odir, "vocab.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB).content)
    return path


def test_preprocess_data_bert():
    with tempfile.TemporaryDirectory() as temp_dir:

        # bert specific args
        bert_args = [
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--vocab-file",
            bert_vocab(temp_dir),
            "--split-sentences",
            "--workers",
            "10",
            "--log-interval",
            "1",
            "--partitions",
            "2",
            "--keep-sequential-samples",
        ]

        do_test_preprocess_data(temp_dir, extra_args=bert_args)


if __name__ == "__main__":
    test_preprocess_data_gpt()
