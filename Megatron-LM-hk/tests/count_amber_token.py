amber_path = "/aifs4su/data/rawdata/amber/AmberDatasets"

import json
import os

def get_all_files(path, ext=None):
    files = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            if ext is None or f.endswith(ext):
                files.append(os.path.join(root, f))
    return files

jsonls = get_all_files(amber_path, ".jsonl")

count = 0

selected_jsonl = jsonls[0]

with open(selected_jsonl, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            count += len(data["token_ids"])
            print(count)
        except:
            pass

print("Total number of tokens: ", count)
