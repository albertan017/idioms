"""Store the idioms binaries for a given dataset partition in a tar file.
"""

import argparse
import tarfile
import os
import io
from pathlib import Path
from typing import Iterable

from idioms.dataiter import MatchedBinaryDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("idioms_dataset")
    parser.add_argument("outdir")
    parser.add_argument("binaries_dir")
    parser.add_argument("--partition", default="test")
    return parser.parse_args()

# Taken from compile_exebench to avoid dependency.
def write_tar(tarfile_name: str | os.PathLike, members: Iterable[tuple[str, bytes]]):
    """Write a collection of files, represented as (file name, file contents) tuples to a tar file.
    """
    with tarfile.open(tarfile_name, "w:gz") as tf:
        for filename, content in members:
            # from https://bugs.python.org/issue22208
            info = tarfile.TarInfo(filename[:128])
            info.size = len(content)
            tf.addfile(info, fileobj=io.BytesIO(content))

def main(args: argparse.Namespace):
    dataset_path = Path(args.idioms_dataset)
    outdir = Path(args.outdir)
    binaries_dir = Path(args.binaries_dir)
    partition: str = args.partition

    assert dataset_path.exists(), f"{dataset_path} does not exist!"

    binaries: list[tuple[str, bytes]] = []
    dataset = MatchedBinaryDataset(dataset_path.glob(f"{partition}*.tar"))
    for binary in dataset:
        with open(binaries_dir / binary.repo / binary.binary_hash, "rb") as fp:
            content = fp.read()
        binaries.append((binary.binary_hash, content))
    
    os.makedirs(outdir, exist_ok=True)
    write_tar(outdir / f"{dataset_path.name}_{partition}_binaries.tar.gz", binaries)



if __name__ == "__main__":
    main(get_args())