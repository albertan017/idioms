"""Add deompiled code to exebench. Data can be saved in either the idioms data format or huggingface's data format.
"""

import argparse
import gzip
import json
import shutil
import os
import re
import io
import hashlib
import tarfile
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset

from idioms.data.dataset import CollectedFunction, DecompiledFunction, MatchedBinary
from prepare import (
    parser,
    canonicalize_function_names,
    build_matched_function,
    FileTypeMapping,
    PreprocessedFunction
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exebenchdir")
    parser.add_argument("output_format", choices=["idioms", "huggingface"])
    parser.add_argument("optimization", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--eval-only", action="store_true", help="Only generate the validation and test sets.")
    parser.add_argument("--idioms-dataset-shard-size", type=int, default=10000, help="The number of examples per idioms dataset shard. Ignored in huggingface mode.")
    parser.add_argument("--decompiler-for-idioms", choices=["hex-rays", "ghidra"], default="hex-rays", help="From which decompiler to source the decompiled code for the Idioms dataset.")
    return parser.parse_args()

class BufferedWriter():
    def __init__(self, outdir: Path, split_name: str, shard_size: int):
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir()
        self.outdir = outdir
        self.split_name = split_name
        self.shard_size = shard_size
        self.buffer: list[MatchedBinary] = []
        self.shard_count = 0
        # used to prevent duplicate items from being entered; some examples are expressly duplicated across train splits.
        self.hashes: set[str] = set() 
    
    def add(self, example: MatchedBinary):
        if example.binary_hash not in self.hashes:
            self.hashes.add(example.binary_hash)
            self.buffer.append(example)
            if len(self.buffer) == self.shard_size:
                self.write()    
    
    def write(self):
        with tarfile.open(self.outdir / f"{self.split_name}-{self.shard_count}.tar", "w") as tf:
            for b in self.buffer:
                content = bytes(json.dumps(b.to_json()), "utf8")
                info = tarfile.TarInfo(f"{b.binary_hash}.json") # tar restricts file names to at most 256 characters.
                info.size = len(content)
                tf.addfile(info, fileobj=io.BytesIO(content))
        self.buffer = []
        self.shard_count += 1


def preprocess_exebench(example: dict[str, str]) -> PreprocessedFunction | None:
    """Convert an exebench entry to a PreprocessedFunction. Return None on error.

    :param example: the dictionary entry at an index of exebench
    :returns: that entry as a preprocessed function, or None if this is not possible.
    """
    types = FileTypeMapping()
    deps: str = example['synth_deps'] if example['synth_deps'] is not None else ""
    deps += (example['real_deps'] if example['real_deps'] is not None else "")
    deps = re.sub(r'(/\*.*?\*/)|(//.*)', '', deps) # remove comments
    root = parser.parse(bytes(deps, 'utf-8')).root_node
    try:
        for node in root.children:
            types.parse_type(node)
    except: # TODO: some of these errors are resolvable; recover from them.
        return None

    tree = parser.parse(bytes(example['func_def'], 'utf-8'))
    root = tree.root_node
    # The 'func_def' field can contain function attributes like __attribute((noinline)),
    # which tree-sitter interprets as 'expression statements'. Filter these out, and get only
    # the actual original code.
    function_definitions = [c for c in root.children if c.type == "function_definition"]
    # assert len(function_definitions) == 1, f"Expected func_def field to contain one function_definition node but found: " + \
    #     ", ".join(c.type for c in root.children) + "\n\n" + example['func_def']
    if len(function_definitions) != 1:
        return None
    try:
        return PreprocessedFunction(function_definitions[0], types)
    except: # (AssertionError, TypeNotFoundError, TypeNotDefinedError, UnsupportedFeatureError)
        return None

def main(args: argparse.Namespace):
    dataset = load_dataset('jordiae/exebench')
    assert isinstance(dataset, DatasetDict)
    is_idioms_format: bool = args.output_format == "idioms"
    optimization: str = args.optimization
    eval_only = args.eval_only
    idioms_decompiler = args.decompiler_for_idioms

    bench_dir = Path(args.exebenchdir)
    assert bench_dir.exists(), f"exebench dir {bench_dir} does not exist!"
    # indir = bench_dir / f"hex-rays-{optimization}"
    # assert indir.exists(), f"hex-rays decompiled information not found at {indir}"
    outdir = bench_dir / f"exebench-idioms-{optimization}-{idioms_decompiler}" if is_idioms_format else bench_dir / f"exebench-hf-{optimization}"
    if eval_only:
        outdir = outdir.with_name(outdir.name + "-eval")
    if outdir.exists():
        shutil.rmtree(outdir)

    split_stats: dict[str, dict[str, dict[str, int]]] = {}

    # These writers are only needed for idioms-format datasets.
    # The datasets package comes with disk serialization built-in
    if is_idioms_format:
        writers = {
            "train": BufferedWriter(outdir, "train", args.idioms_dataset_shard_size),
            "valid": BufferedWriter(outdir, "validation", args.idioms_dataset_shard_size),
            "test": BufferedWriter(outdir, "test", args.idioms_dataset_shard_size)
        }
        if eval_only:
            del writers["train"]
    else:
        updated_splits = DatasetDict()

    for split_name, split in dataset.items():
        assert isinstance(split_name, str) and isinstance(split, Dataset)

        if eval_only and split_name.startswith("train"):
            continue

        missing_data: bool = False
        decompiled_by_decompiler: dict[str, list[DecompiledFunction | None]] = {}
        for decompiler in ((idioms_decompiler,) if is_idioms_format else ("hex-rays", "ghidra")):
            splitdir: Path = bench_dir / f"{decompiler}-{optimization}" / split_name / "bins"
            if not splitdir.exists():
                missing_data = True
                break

            stripped_function_names = 0
            mismatched_names = 0
            missing_debug = 0

            decompiled: list[DecompiledFunction | None] = [None] * len(split)
            for file in tqdm(splitdir.iterdir(), desc=split_name + "/" + decompiler, total=len(os.listdir(splitdir)), dynamic_ncols=True):
                if file.suffix == ".gz":
                    ### Extract the index from the file name.
                    # File name format: hash_splitname_index_fnname
                    # A simple .split("_")[2] won't work here, because "_" is used for different purposes within the filename.
                    # (For instance, it's used to separate the hash from the split name but also the components of the split name.)
                    # Tnus, we add the number of "_"s within the split name itself to get the correct offset of the index in the filename.
                    index = int(file.name.split("_")[split_name.count("_") + 2])
                    with gzip.open(file, "r") as fp:
                        # Technically there's only supposed to be other one function per file, but sometimes there's also other
                        # compiler or decompiler generated stuff in there so there that we have to deal with.
                        content = fp.read()
                    examples: list[DecompiledFunction] = []
                    for line in content.decode().strip().splitlines():
                        raw_example = json.loads(line)
                        # b is the version of the function with debug info.
                        # This has only been observed with the ghidra version of the dataset processor; confirm what's going on
                        # before applying this to the hex-rays dataset processor as well.
                        if raw_example['b'] is None and decompiler == "ghidra":
                            # Use the stripped version of the code as a placeholder because we don't actually care about the debug version anyway.
                            raw_example['b'] = raw_example['c']
                            missing_debug += 1
                        cf = CollectedFunction.from_json(raw_example)
                        examples.append(DecompiledFunction.from_cf(cf, binary="placeholder", max_stack_length=1024, max_type_size=1024))

                    # Note that because the decompiled function name doesn't always exactly match the function name
                    # (occasionally the function name is stripped out or is slightly transformed) we check the 1-function-in-the-list
                    # case separately from the stricter multi-function case, which requires an exact name match.
                    # This maximizes the number of examples we can get decompilation for.
                    if len(examples) == 0:
                        continue # decompilation failed.
                    elif len(examples) == 1:
                        example = examples[0] # usual case. What we'd expect.
                    else:  # select the actual function, removing compiler/decomiler artifacts.
                        exebench_name = split[index]['fname']
                        for example in examples:
                            if example.name == exebench_name:
                                break # the example variable remains set.
                        else: # attached to the for loop
                            example = None # technically unnecessary.
                            continue

                    # Canonicalization is done in-place and sets the field 'canonical_code' on the DecompiledFunction object.
                    example.raw_code = example.raw_code.replace("__fastcall", "") # Confuses tree-sitter, causing it to misidentify the function as a compound statement.
                    canonicalize_function_names([example]) # canonicalize_function_names takes a list to support binaries with multiple functions.
                    
                    # Occasionally the function name can be stripped out.
                    function_name_stripped = example.name == "sub_0" or example.name == "FUN_00100000" # for hex-rays and ghidra, respectively.
                    stripped_function_names += function_name_stripped # now included in mismatched_names

                    assert decompiled[index] is None or decompiled[index].canonical_code == example.canonical_code, f"Function {example.name} at index {index} is already defined!" # type: ignore # mypy can't handle the short-circuting or 'or' along with an array access.
                    if function_name_stripped:
                        example.name = split[index]['fname']
                    elif not (split[index]['fname'].strip("_") == example.name.strip("_")): #, f"Name mismatch at index {index}: {example.name} (decompiled) vs {split[index]['fname']} (exebench).\nFile={file}"
                        mismatched_names += 1
                        continue
                    decompiled[index] = example

            if split_name not in split_stats:
                split_stats[split_name] = {}
            split_stats[split_name][decompiler] = {
                'total_examples': len(decompiled),
                'missing_decompilation': sum(hr is None or hr.canonical_code is None for hr in decompiled),
                'mismatched_function_names': mismatched_names,
                'stripped_function_names (included)': stripped_function_names,
                "missing_debug": missing_debug
            }

            decompiled_by_decompiler[decompiler] = decompiled

        if missing_data:
            continue
        
        # Data-format-specific processing.
        if is_idioms_format:
            writer = writers[split_name.split("_")[0]]
            for bench_info, decomp_info in tqdm(zip(split, decompiled), total=len(decompiled), desc="Converting to idioms format"):
                if decomp_info is None:
                    continue
                assert isinstance(bench_info, dict)
                preprocessed_fn = preprocess_exebench(bench_info)
                if preprocessed_fn is None:
                    continue
                if preprocessed_fn.name != decomp_info.name:
                    # hex-rays can sometimes get rid of leading underscores on function names.
                    # We make sure the difference is only due to leading/trailing underscores
                    # (The names must match to build a PreprocessedFunction)
                    if preprocessed_fn.name.strip("_") != decomp_info.name.strip("_"): #, f"{preprocessed_fn.name} != {decomp_info.name}"
                        continue
                    decomp_info.name = preprocessed_fn.name
                # This isn't a hash of the binary per se, but it's the same idea and is less
                # vulnerable to compilation artifacts that add meaningless difference to the binary itself.
                unique_hash = hashlib.sha256(bytes(bench_info['func_def'], 'utf-8')).hexdigest()
                decomp_info.binary = unique_hash
                path = Path(bench_info['path'])
                repo = path.parts[0] + "/" + path.parts[1]
                fn = build_matched_function(decomp_info, preprocessed_fn, repo)
                if fn is None:
                    continue
                # Wrap the function in a MatchedBinary because that's what the downstream code expects,
                # though it really isn't necessary for exebench otherwise.
                writer.add(MatchedBinary([fn], unique_hash, repo, {fn.name: []}, {}))
        else:
            # d.canonical_code can be None as well but that's fine.
            hex_rays_decompiled_code: list[str | None] = [(None if d is None else d.canonical_code) for d in decompiled_by_decompiler['hex-rays']]
            ghidra_decompiled_code: list[str | None] = [(None if d is None else d.canonical_code) for d in decompiled_by_decompiler['ghidra']]
            split = split.add_column("hex-rays", hex_rays_decompiled_code) # type: ignore # there's a required argument added by a decorator that mypy isn't aware of.
            split = split.add_column("ghidra", ghidra_decompiled_code)
            updated_splits[split_name] = split
            # split.save_to_disk(bench_dir / split_name)

    if is_idioms_format:
        # Flush the buffers
        for writer in writers.values():
            writer.write()
            print(f"{writer.split_name} set: {len(writer.hashes)} examples.")
    else:
        updated_splits.save_to_disk(outdir)

    print("---- Statistics ----")
    for split_name, decompiler_stats in split_stats.items():
        print(split_name)
        for decompiler, stats in decompiler_stats.items():
            print(f"-- {decompiler}")
            for stat, value in stats.items():
                print(f"   {stat}: {value}")
    
    with open(bench_dir / f'{optimization}-stats.json', "w") as fp:
        json.dump(split_stats, fp, indent=2)


if __name__ == "__main__":
    main(get_args())