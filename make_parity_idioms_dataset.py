"""Make a new idioms-format dataset that is the same size as the old idioms-format dataset.
Ensure that wherever possible the same binaries (by repo name/binaries in the function.)
"""

import argparse
import itertools
import functools
import shutil
import sys
import os
from pathlib import Path
from typing import Any
from collections import Counter

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from idioms.data.dataset import MatchedBinary, MatchedFunction
from idioms.dataiter import MatchedBinaryDataset, MatchedFunctionDataset
from prepare import write_shard

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_datasets", nargs='+', help="The two input datasets.")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Only perform the process for the test sets.")
    parser.add_argument("--output-suffix", default="opt_parity", help="This suffix will be attached to the name of each repository to form the names of the output repositories.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write out any files.")
    parser.add_argument("--skip-first-write", action="store_true", help="Do not write out only the first dataset.")
    return parser.parse_args()

def print_binary(b: MatchedBinary):
    print(f"-- Repo: {b.repo}")
    print(f"-- Hash: {b.binary_hash}")
    print(f"-- Functions:")
    for fn in b.functions:
        print(fn.name)
    print(f"-- Call Graph:")
    for node, edges in b.call_graph.items():
        print(node, edges)
    print(f"-- Unmatched:")
    for name, code in b.unmatched.items():
        print(name)
    print()

def dataset_by_repo(dataset: MatchedBinaryDataset) -> dict[str, list[MatchedBinary]]:
    byrepo: dict[str, list[MatchedBinary]] = {}
    for binary in dataset:
        if binary.repo not in byrepo:
            byrepo[binary.repo] = []
        byrepo[binary.repo].append(binary)
    return byrepo

def name_set(b: MatchedBinary) -> set[str]:
    return {fn.name for fn in b.functions}

def calculate_generalized_intersection(indices: tuple[int,...], names: list[list[set[str]]]) -> int:
    return len(set.intersection(*(names[i][j] for i, j in enumerate(indices))))

def make_parity_partition(dataset_paths: list[Path], partition: str, verbose: bool = False) -> list[list[MatchedBinary]]:
    """Subsamples the datasets at dataset_paths for the provided partition, ensuring only functions that occur
    in all three datasets make it into the final datasets.
    """
    assert len(dataset_paths) >= 1
    print(f"---- Processing {partition} set ----")
    print("Loading datasets...")
    datasets = [
        dataset_by_repo(MatchedBinaryDataset(dsp.glob(f"{partition}*.tar"), shuffle=False))
        for dsp in dataset_paths
    ]
    
    shared_repos = set(datasets[0])
    for dataset in datasets[1:]:
        shared_repos.intersection_update(dataset)
    shared_repos = sorted(shared_repos) # Sorted alphabetically just for reproducability

    for path, dataset in zip(dataset_paths, datasets):
        shared_bins = sum(len(dataset[repo]) for repo in shared_repos)
        shared_functions = sum(sum(len(b.functions) for b in dataset[repo]) for repo in shared_repos)
        print(f"Shared part of {path.name} dataset: {shared_bins} binaries, {shared_functions} functions.")
    
    combinations = 0
    for repo in shared_repos:
        term = 1
        for i in range(len(datasets)):
            term *= len(datasets[i][repo])
        combinations += term
    print(f"Total number of combinations: {combinations}\n")

    mapped_binaries: list[tuple[MatchedBinary,...]] = []
    progress = tqdm(desc=f"Finding matched repos", total=combinations)

    slices = [slice(None) for _ in range(len(datasets))]

    # Match binaries with each other
    for repo in shared_repos:
        # Outer list represents a dataset, inner corresponds to the binaries in a given dataset for a specific repository
        binaries: list[list[MatchedBinary]] = [ds[repo] for ds in datasets]
        mapped: set[str] = set()
        name_occurrences: list[dict[str, list[MatchedBinary]]] = []
        original_term_size = functools.reduce(lambda x, y: x * len(y), binaries, 1)
        
        ## Attempt to match binaries together by unique functions.
        # Determine which binaries contain which names
        for i, ds_subset in enumerate(binaries):
            occurrences: dict[str, list[MatchedBinary]] = {}
            for binary in ds_subset:
                for fn in binary.functions:
                    if fn.name in occurrences:
                        occurrences[fn.name].append(binary)
                    else:
                        occurrences[fn.name] = [binary]
            name_occurrences.append(occurrences)

        all_function_names: set[str] = set.intersection(*(set(occ) for occ in name_occurrences))
        for fnname in all_function_names:
            if all(fnname in occs and len(occs[fnname]) == 1 for occs in name_occurrences):
                unique = tuple(occs[fnname][0] for occs in name_occurrences)
                if all(u.binary_hash not in mapped for u in unique):
                    mapped.update(u.binary_hash for u in unique)
                    mapped_binaries.append(unique)

        # For remaining binaries, use a brute-force computation of all generalized-intersections 
        # between all binaries in each dataset's representation of the repository.
        binaries = [[b for b in bins if b.binary_hash not in mapped] for bins in binaries]
        reduced_term_size = functools.reduce(lambda x, y: x * len(y), binaries, 1)
        progress.update(original_term_size - reduced_term_size)
        if reduced_term_size == 0:
            continue # there's nothing left to match.
        names: list[list[set[str]]] = [[name_set(b) for b in bins] for bins in binaries]
        shape = tuple(len(bins) for bins in binaries)
        shared: NDArray = np.zeros(shape)
        for indices in itertools.product(*(range(bound) for bound in shape)):
            shared[*indices] = len(set.intersection(*(names[i][j] for i, j in enumerate(indices))))
            progress.update(1)
        
        # Greedily match binaries to each other by taking those with the largest intersection size.
        # Ensure that binaries are not repeated.
        for _ in range(min(shape)):
            indices = np.unravel_index(shared.argmax(), shared.shape)
            if shared[*indices] > 0:
                mapped_binaries.append(tuple(binaries[i][j] for i, j in enumerate(indices)))
                for i in range(len(indices)):
                    to_eliminate: list[Any] = slices.copy()
                    to_eliminate[i] = indices[i]
                    shared[*to_eliminate] = -1
            else:
                break # max value is 0 so there's no function that corresponds with any other function in the matrix.

    progress.close()

    # Perform checks that the binaries matched to each other in the previous stage
    # actually should match to each other
    parity: list[list[MatchedBinary]] = [list() for _ in range(len(datasets))]
    for bins in mapped_binaries:
        bins: tuple[MatchedBinary,...] # indexed by dataset.
        common_function_names = set.intersection(*(name_set(b) for b in bins))

        if len(common_function_names) == 0:
            continue # There's no useful data here; we need at least one matched_function to do much of anything.

        parity_fns: list[list[MatchedFunction]] = []
        for i in range(len(datasets)):
            parity_fns.append([
                f for f in bins[i].functions if f.name in common_function_names
            ])

        # Ensure that there are no functions with duplicate names in the binary.
        if any(len(b) != len(set(fn.name for fn in b)) for b in parity_fns):
            continue

        # Sanity check: ensure that all functions with the same original name have the same body. If not, filter these out.
        # Next, ensure that the binaries are exactly the same in terms of the functions they contain, as an additional control.
        mismatched = False
        first = sorted(parity_fns[0], key=lambda x: x.name)
        for other in parity_fns[1:]:
            other = sorted(other, key=lambda x: x.name)
            for fn1, fn2 in zip(first, other):
                assert fn1.name == fn2.name, f"{fn1.repo}: {fn1.name} != {fn2.name} ({fn1.binary_hash}, {fn2.binary_hash})"
            if not all(fn1.canonical_original_code == fn2.canonical_original_code for fn1, fn2 in zip(first, other)):
                mismatched=True
                break
        if mismatched:
            continue
        
        def make_unmatched(binary: MatchedBinary) -> dict[str, str]:
            unmatched: dict[str, str] = binary.unmatched.copy()
            for fn in binary.functions:
                if fn.name not in common_function_names:
                    unmatched[fn.name] = fn.canonical_decompiled_code
            return unmatched
    
        for i in range(len(datasets)):
            parity[i].append(MatchedBinary(
                functions=parity_fns[i],
                binary_hash=bins[i].binary_hash,
                repo=bins[i].repo,
                call_graph=bins[i].call_graph,
                unmatched=make_unmatched(bins[i])
            ))

        if verbose:
            print("*" * 20 + " Before " + "*" * 20)
            for i in range(len(datasets)):
                print(f"-------- {dataset_paths[i].name}")
                print_binary(bins[i])
                print()

            print("*" * 20 + " After " + "*" * 20)
            for i in range(len(datasets)):
                print(f"-------- {dataset_paths[i].name}")
                print_binary(parity[i][-1])
                print()
            print(("*" * 40 + "\n") * 3)

    for i in range(len(datasets)):
        total = calc_call_graph_sizes(parity[i])
        print(f"Call graph sizes for {dataset_paths[i].name}: Total: {total} edges, Mean: {total / len(parity[i])} edges.")
    print()
    
    for i in range(len(datasets)):
        original_size = len(MatchedFunctionDataset(dataset_paths[i].glob(f"{partition}*.tar"), shuffle=False))
        print(f"{dataset_paths[i].name} size: {original_size} functions, {sum(len(bins) for bins in datasets[i])} binaries; reduced to {sum(len(b.functions) for b in parity[i])} functions, {len(parity[i])} binaries.")
        assert len(parity[i]) == len(set(f"{b.binary_hash}_{b.repo}" for b in parity[i])), list(filter(lambda x: x[1] > 1, Counter(f"{b.binary_hash}_{b.repo}" for b in parity[i]).items() ))
    
    return parity

def calc_call_graph_sizes(mbs: list[MatchedBinary]):
    total = 0
    for mb in mbs:
        for edges in mb.call_graph.values():
            total += len(edges)
    return total

def main(args: argparse.Namespace):
    dataset_paths: list[Path] = []
    for path in args.in_datasets:
        dataset_paths.append(Path(path))
        assert dataset_paths[-1].exists()
    shard_size: int = args.shard_size
    suffix: str = args.output_suffix
    verbose: bool = args.verbose
    test_only: bool = args.test_only
    dry_run: bool = args.dry_run
    skip_first_write: bool = args.skip_first_write

    output_paths: list[Path] = []
    for path in dataset_paths:
        outpath = path.parent / (path.name + "_" + suffix)
        assert not outpath.exists(), f"Output path {outpath} already exists!"
        output_paths.append(outpath)

    test_sets = make_parity_partition(dataset_paths, "test", verbose)
    partition_names = ("test",)
    partition_contents = (test_sets,)
    if not test_only:
        validation_sets = make_parity_partition(dataset_paths, "validation", verbose)
        train_sets = make_parity_partition(dataset_paths, "train", verbose)
        partition_names += ("validation", "train")
        partition_contents += (validation_sets, train_sets)

    if not dry_run:
        for i, path in enumerate(dataset_paths):
            if skip_first_write and i == 0:
                continue
            outbase = output_paths[i]
            print(f"Making {outbase}")
            os.makedirs(outbase, exist_ok=True)
            for partition_name, partitions in zip(partition_names, partition_contents):
                partition = partitions[i]
                shard_no = 0
                for j in range(0, len(partition), shard_size):
                    write_shard(outbase / f"{partition_name}-{shard_no}.tar", partition[j:j+shard_size])
                    shard_no += 1
            for file in path.iterdir():
                if file.is_file() and file.suffix != ".tar" and file.suffix != ".pkl":
                    print(f"Copying {file} -> {outbase / file.name}.")
                    shutil.copyfile(file, outbase / file.name)
            with open(outbase / "command.txt", "a") as fp:
                fp.write(" ".join(sys.argv))
            print()

if __name__ == "__main__":
    main(get_args())
