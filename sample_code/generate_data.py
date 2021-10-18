import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

TEST_NAMES = []


def process(root1, root2, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("loading pos...")
    pos1 = pd.read_hdf(os.path.join(root1, ""))
    pos2 = pd.read_hdf(os.path.join(root2, ""))
    pos = pd.concat([pos1, pos2], axis=0)
    chunks = []
    for f in os.listdir(root1):
        if "" in f:
            print(f"loading neg {f}...")
            chunk = pd.read_hdf(os.path.join(root1, f))
            chunks.append(chunk)
    for f in os.listdir(root2):
        if "" in f:
            print(f"loading neg {f}...")
            chunk = pd.read_hdf(os.path.join(root2, f))
            chunks.append(chunk)
    neg = pd.concat(chunks, axis=0)
    neg["Target"] = 0
    df = pd.concat([neg, pos]).reset_index(drop=True)
    samples = [{"values": row.Data.T.values, "target": row.Target, "name": row.File.split(".")[0]} for _, row in tqdm(df.iterrows())]
    print(samples[0]["values"].shape)
    with open(os.path.join(save_dir, "all_raw.pkl"), "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    root1 = "data/raw/batch_1"
    root2 = "data/raw/batch_2"
    save_dir = "data/processed"
    process(root1, root2, save_dir)
