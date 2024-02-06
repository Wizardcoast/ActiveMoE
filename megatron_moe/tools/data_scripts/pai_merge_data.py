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
    parser.add_argument("--v", type=str, default='xx',
                        help="use v to format data ")
    parser.add_argument("--workers", type=int, default=5,
                        help="number of workers")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data_root = args.data_root
    category = args.category
    tokenizer = args.tokenizer
    num_workers = args.workers
    output_root = args.output_path
    version = args.v

    cmd = """
    python3 tools/merge_datasets.py \
        --input {entries} \
        --output-prefix {root}/{split}_{date}
    """

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    from multiprocessing import Pool

    def process_category(c):
        c_path = os.path.join(data_root, c)

        cmd_f = cmd.format(entries=c_path, root=output_root,
                           split=c, date=version)

        subprocess.run(cmd_f.split())

    with Pool(num_workers) as p:
        p.map(process_category, category)

