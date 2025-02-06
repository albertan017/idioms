"""Select a subset of exebench such that it is the same size as the idioms-format dataset.
"""

import argparse
import random
import shutil
import os
from pathlib import Path

from idioms.dataiter import MatchedFunctionDataset, MatchedBinaryDataset
from prepare import write_shard

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exebench_ds", help="The exebench dataset to subsample; should be in idioms format.")
    parser.add_argument("idioms_ds")
    parser.add_argument("output_ds")
    parser.add_argument("--shard-size", default=10000, help="Number of MatchedBinaries (which here contain exactly one function) per output tar file.")
    parser.add_argument("--random-seed", type=int, default=80)
    return parser.parse_args()

def main():
    args = get_args()
    random.seed(args.random_seed)
    exebench_path = Path(args.exebench_ds)
    idioms_path = Path(args.idioms_ds)
    output_path = Path(args.output_ds)
    assert not output_path.exists() or len(os.listdir(output_path)) == 0
    shard_size: int = args.shard_size

    exebench = MatchedBinaryDataset(exebench_path.glob("train*.tar"))
    idioms_ds = MatchedFunctionDataset(idioms_path.glob("train*.tar"))

    print("Loading exebench...")
    exebench_examples = list(exebench)
    print("Loading idioms dataset...")
    idioms_examples = list(idioms_ds)

    assert all(len(binary.functions) == 1 for binary in exebench_examples)

    print(f"Number of exebench examples: {len(exebench_examples)}")
    print(f"Number of idioms examples: {len(idioms_examples)}")
    print()

    subsample = random.sample(exebench_examples, k=len(idioms_examples))

    output_path.mkdir()
    for i, k in enumerate(range(0, len(subsample), shard_size)):
        shard_contents = subsample[k:(k + shard_size)]
        write_shard(output_path / f"train-{i}.tar", shard_contents)

    # Copy over test and validation sets.
    # Technically not necessary, as we always use the huggingface-format dataset because
    # that dataset format contains the necessary info to run the exebench tests.
    for holdout in ("test", "validation"):
        i = 0
        while (holdout_path := exebench_path / f"{holdout}-{i}.tar").exists():
            shutil.copyfile(holdout_path, output_path / holdout_path.name)
            i += 1
    
    del exebench_examples

    ### Now, double-check it loads back in correctly and pre-populate the length cache.
    reloaded_ds = MatchedFunctionDataset(output_path.glob("train*.tar"), length_cache=output_path / "length_cache.pkl")
    assert len(reloaded_ds) == len(idioms_examples), f"Reloaded dataset has {len(reloaded_ds)} examples, while the idioms dataset has {len(idioms_examples)} examples!"

if __name__ == "__main__":
    main()