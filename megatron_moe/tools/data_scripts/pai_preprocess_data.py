import os
import subprocess
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", )
    parser.add_argument('--category', nargs='+', type=str,
                        default=[], help="category")

    parser.add_argument("--tokenizer", type=str, default='tokenizer_v2')
    parser.add_argument("--output_path", type=str, default='data')
    parser.add_argument("--v", type=str, default='x',
                        help="use v to format data ")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data_root = args.data_root
    category = args.category
    tokenizer = args.tokenizer
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    world_size = int(os.environ['RC_WORLD_SIZE']
                     ) if 'RC_WORLD_SIZE' in os.environ else 1
    output_path = args.output_path
    version = args.v

    file_list = []
    for c in category:
        print(f'processing {c}')
        c_path = os.path.join(data_root, c)

        for file in os.listdir(c_path):
            if not file.endswith('.jsonl'):
                continue
            file_path = os.path.join(c_path, file)
            file_list.append((file_path, c))

    file_list = sorted(file_list, key=lambda x: x[0])

    import random
    # all ranks share the same seed!
    random.seed(42)
    random.shuffle(file_list)

    print('shuffle list to balance workers. all ranks should be the same', file_list)

    import numpy as np

    arr = range(len(file_list))
    print('using rank to split', np.array_split(arr, world_size))
    i_list = list(np.array_split(arr, world_size)[rank])

    rank_file_list = []
    for i in i_list:
        rank_file_list.append(file_list[i])

    print(f'rank {rank} will process: {rank_file_list}', flush=True)

    cmd = """
    python tools/preprocess_data.py 
       --input {input_file} 
       --output-prefix {output_path}/{category}/{version}_{file_id} 
       --dataset-impl mmap 
       --tokenizer-type PretrainedFromHF 
       --tokenizer-name-or-path {tokenizer} 
       --append-eod 
       --workers 24
    """

    for file, c in rank_file_list:
        if not os.path.exists(f"{output_path}/{c}/"):
            os.makedirs(f"{output_path}/{c}", exist_ok=True)

        file = os.path.basename(file)
        file_id = file[:-6]
        input_file = os.path.join(data_root, c, file)
        cmd_f = cmd.format(input_file=input_file, category=c, tokenizer=tokenizer,
                           file_id=file_id, output_path=output_path, version=version)
        print(cmd_f.split())
        subprocess.run(cmd_f.split())

