"""Compile the code in exebench (https://huggingface.co/datasets/jordiae/exebench). 
"""

import argparse
import subprocess
import os
import tempfile
import tarfile
import re
import io
import multiprocessing
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
from datasets import load_dataset, DatasetDict

TEMP_C_FILE_NAME = "example.c"
TEMP_BIN_FILE_NAME = "example.o"
ERROR_FILE_NAME = "errors.txt"
MEG = 1e6
GIG = 1e9

# Use RAM-backed memory for tmp if available
if os.path.exists("/dev/shm"):
    tempfile.tempdir = "/dev/shm"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument("--shard-size", type=int, default=1, help="Size of each shard, in GB")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--max-num-logged-errors", type=int, default=10000, help="Only log this many errors (to prevent the error file from becoming too large.)")
    return parser.parse_args()


def clean_deps(deps: str, to_remove: Iterable[str]) -> str:
    """Remove any line from deps that contains the dependencies in to_remove.
    """
    lines = deps.split("\n")
    return "\n".join(l for l in lines if all(r not in l for r in to_remove))


CONFLICTING_TYPES_ERROR = re.compile(r"""error: conflicting types for .(\w+).""") # We use . for the quotation marks because it seems they can sometimes be non-ascii.
REDEFINED_SYMBOL_ERROR = re.compile(r"""error: redefinition of .([a-zA-Z_][a-zA-Z0-9 _]*).""")
REDECLARED_SYMBOL_ERROR = re.compile(r"""error: .([a-zA-Z_][a-zA-Z0-9 _]*). redeclared as different kind of symbol""")

def compile_example(example: dict[str, Any], synth_decls_to_remove: set[str] | None = None) -> bytes | subprocess.CompletedProcess:
    """Compile an example and return a unique identifier along with the raw contents of the executable binary, in bytes.
    The unique identifier is based on the example's "path" and "fname" (function name) fields.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        # Write the example t a file.
        with open(os.path.join(tempdir, "example.c"), "w") as fp:
            if example['synth_deps'] is not None:
                synth_deps = example['synth_deps']
                if synth_decls_to_remove is not None:
                    synth_deps = clean_deps(synth_deps, synth_decls_to_remove)
                fp.write(synth_deps)
                fp.write("\n\n\n")
            if example['real_deps'] is not None:
                fp.write(example['real_deps'])
                fp.write("\n\n\n")
            fp.write(example['func_def'])
        
        # Run the compilation command
        command = ["gcc", "-g", "-O0", "-c", TEMP_C_FILE_NAME]
        runresult = subprocess.run(command, capture_output=True, cwd=tempdir)
        object_file = os.path.join(tempdir, TEMP_BIN_FILE_NAME)
        if os.path.exists(object_file):
            with open(object_file, "rb") as fp:
                binary = fp.read()
        else:
            compiler_error = runresult.stderr.decode()
            error_match = CONFLICTING_TYPES_ERROR.search(compiler_error) # Matches 'conflicting types for' errors.
            if error_match is None:
                error_match = REDEFINED_SYMBOL_ERROR.search(compiler_error)
            if error_match is None:
                error_match = REDECLARED_SYMBOL_ERROR.search(compiler_error)
            if error_match is not None:
                # Handle the error by removing the offending synthetic function declaration from the dependencies, and try compiling again.
                if synth_decls_to_remove is None:
                    synth_decls_to_remove = set()
                errorfn = error_match.group(1) # All of the errors we check have the relevant text we want to exclude at group(1)
                if errorfn not in synth_decls_to_remove: # must have increasing set size to prevent infinate recursion.
                    synth_decls_to_remove.add(errorfn)
                    return compile_example(example, synth_decls_to_remove) # Try again after excluding the synthetic function definition causing the type-mismatch error.
            binary = runresult # Which contains the error message

    return binary

def write_tar(tarfile_name: str | os.PathLike, members: Iterable[tuple[str, bytes]]):
    """Write a collection of files, represented as (file name, file contents) tuples to a tar file.
    """
    with tarfile.open(tarfile_name, "w:gz") as tf:
        for filename, content in members:
            # from https://bugs.python.org/issue22208
            info = tarfile.TarInfo(filename[:128]) # tar restricts file names to at most 256 characters.
            info.size = len(content)
            tf.addfile(info, fileobj=io.BytesIO(content))

def main(args: argparse.Namespace):
    dataset = load_dataset('jordiae/exebench')
    assert isinstance(dataset, DatasetDict)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    error_file_path = outdir / ERROR_FILE_NAME
    shard_size: float = args.shard_size
    max_num_logged_errors: int = args.max_num_logged_errors

    # Where the compilation errors for failed compilations are written.
    if error_file_path.exists():
        error_file_path.unlink()

    total_bytes = 0
    total_errors = 0
    shard_no = 0 # Global across all shards.

    progress = tqdm(total=sum(len(shard) for name, shard in dataset.items() if name != "train_not_compilable"), dynamic_ncols=True)
    bytes_bar = tqdm(desc="Total size of binaries collected", unit=" MB")
    error_bar = tqdm(desc="Unfixable compilation errors", unit=" errs")
    for name, split in dataset.items():
        if name == 'train_not_compilable': # Would be rather silly to try and compile this.
            continue
        progress.set_description(f"Processing split '{name}'")
        subdir: Path = outdir / name
        subdir.mkdir(exist_ok=True)
        shard_bytes = 0
        buffer: list[tuple[str, bytes]] = []
        with multiprocessing.Pool(args.workers) as pool:
            for i, example, content in zip(range(len(split)), split, pool.imap(compile_example, split, chunksize=1)):
                filename = f"{name}_{i}_{example['fname']}" # fname key of "example" is "function_name" here.
                if isinstance(content, bytes): # We have successfully compiled an example into a binary
                    buffer.append((filename, content))
                    # Log the number of bytes of binary, in terms of the floor of the number of MB.
                    old_meg = total_bytes // MEG
                    total_bytes += len(content)
                    new_meg = total_bytes // MEG
                    if new_meg > old_meg:
                        bytes_bar.update(new_meg - old_meg)
                    # Write out the buffer if we've recorded enough binaries.
                    shard_bytes += len(content)
                    if shard_bytes / GIG > shard_size:
                        write_tar(subdir / f"shard-{shard_no}.tar.gz", buffer)
                        shard_bytes = 0
                        shard_no += 1
                        del buffer
                        buffer = []
                else: # The example could not be compiled into a binary.
                    assert isinstance(content, subprocess.CompletedProcess)
                    # buffer.append((filename, bytes())) # Don't append this because we'll try to decompile it and cause errors.
                    if total_errors < max_num_logged_errors:
                        with open(error_file_path, "a") as fp:
                            fp.write(f"******** Split {name}, example {i}\n")
                            fp.write(content.stderr.decode())
                            fp.write("\n\n\n\n")
                    error_bar.update(1)
                    total_errors += 1
                progress.update(1)
        # Write out the rest of the binaries in this split.
        write_tar(subdir / f"shard-{shard_no}.tar", buffer)
        shard_bytes = 0
        shard_no += 1
        del buffer
        buffer = []

    progress.close()
    bytes_bar.close()
    error_bar.close()

    with open(outdir / "stats.txt", "w") as fp:
        fp.write(f"Unfixable compilation errors: {total_errors}\n")
        fp.write(f"Total size of all binaries: {round(total_bytes / GIG, 1)} GB\n")

if __name__ == "__main__":
    main(get_args())