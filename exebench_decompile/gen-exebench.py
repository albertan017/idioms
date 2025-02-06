"""Decompile all functions in exebench
"""

import argparse
import subprocess
import tempfile
import tarfile
import os
from pathlib import Path

if os.path.exists("/dev/shm"):
    tempfile.tempdir = "/dev/shm"

parser = argparse.ArgumentParser()
parser.add_argument("exebench_dir")
parser.add_argument("optimization", choices=["O0", "O1", "O2", "O3"])
parser.add_argument("--decompiler", type=str)
parser.add_argument("--eval-only", action="store_true")
args = parser.parse_args()

exebench = Path(args.exebench_dir)
compiled = exebench / ("compiled-" + args.optimization)

if Path(args.decompiler).name == "analyzeHeadless":
    decompiler_args = ["--ghidra", args.decompiler]
    decompiled = exebench / ("ghidra-" + args.optimization)
elif Path(args.decompiler).name[:4] == "idat":
    decompiler_args = ["--ida", args.decompiler]
    decompiled = exebench / ("hex-rays-" + args.optimization)
else:
    raise ValueError(f"Unable to detect decompiler from {args.decompiler}")
decompiled.mkdir(exist_ok=True)

for split in compiled.iterdir():
    if not split.is_dir(): # ignore log/statistics files.
        continue
    if args.eval_only and "train" in split.name:
        continue
    with tempfile.TemporaryDirectory() as tempdir:
        for shard in split.iterdir():
            assert ".tar" in shard.name
            with tarfile.open(shard, "r:*") as tf:
                tf.extractall(path=tempdir)

        split_out = decompiled / split.name
        split_out.mkdir(exist_ok=True)
        
        print(f"Decompiling split {split.name}")
        subprocess.run([
            "python",
            "generate.py",
            *decompiler_args,
            "-b", tempdir,
            "-o", str(split_out),
            "-t", "50"
        ])

