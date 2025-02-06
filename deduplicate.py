"""Determine which repositories are duplicates of each other.
"""

import argparse
import tarfile
import functools
import itertools
import multiprocessing
import json
import hashlib
import struct
import tempfile
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TypeVar, Iterator, Iterable, NamedTuple

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from pygments.lexers.c_cpp import CLexer
from pygments.token import Whitespace

# From https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication.py
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)
T = TypeVar('T')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_archives", help="The path to the directory containing the repo archives. Expected to be in the format owner/repo.tar.gz")
    parser.add_argument("output", help="The name of the json file to which the output should be written.")
    parser.add_argument("--shingle-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0, help=f"Number of multiprocessing workers. 0 for single-process operation.")
    parser.add_argument("--random-seed", type=int, default=80)
    parser.add_argument("--num-hash-fns", type=int, default=128, help="The number of minhash hashing functions to use.")
    parser.add_argument("--num-tables", type=int, default=64, help="Number of hash tables used in LSH. Band size is --num-hash-fns / --num-tables")
    parser.add_argument("--lexical", action="store_true", help="Use subsequences of lexical tokens for computing shingles rather than raw substrings.")
    parser.add_argument("--limit", type=int, help="Only process this many repos (for debugging).")
    return parser.parse_args()

class MinHashedRepo(NamedTuple):
    repo: Path
    minhash: NDArray

    def __hash__(self):
        """This hash function for normal python Dict/Set hashing, not MinHashing or LSH.
        MinHashing and LSH should use the minhash field of this NamedTuple.
        """
        return hash(self.repo)

    def __eq__(self, other):
        return isinstance(other, MinHashedRepo) and self.repo == other.repo
    
    @property
    def slug(self):
        return self.repo.parent.name + "/" + self.repo.with_suffix("").with_suffix("").name

class MinHashFailed(NamedTuple):
    repo: Path
    reason: str

    @property
    def slug(self):
        return self.repo.parent.name + "/" + self.repo.with_suffix("").with_suffix("").name


def sha1_hash32(string: str) -> int:
    """Heavily based on
    https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication.py#L73
    
    In that project, this was taken from the datasketch package to avoid dependency.
    """
    return struct.unpack("<I", hashlib.sha1(bytes(string, "utf8")).digest()[:4])[0]

def extract_c_code_from_archive(archive: Path) -> Iterable[str]:
    assert archive.name.endswith(".tar.gz"), f"Repo archive {archive} is not a tar.gz file!"

    with tempfile.TemporaryDirectory() as td:
        tempdir = Path(td)
        with tarfile.open(archive, mode="r:gz") as tf:
            repo_archive_dir_name = tf.getnames()[0] # the name of the root directory of the archive.
            tf.extractall(tempdir, filter="data")
        
        raw_code_dir = tempdir / (repo_archive_dir_name) # Where the contents of the repository are stored
        
        for c_file in itertools.chain(raw_code_dir.glob("**/*.c"), raw_code_dir.glob("**/*.C")):
            if c_file.is_file() and not c_file.is_symlink():
                try:
                    with open(c_file, "r") as fp:
                        yield fp.read()
                except UnicodeDecodeError:
                    pass

def get_textual_shingle_set(repo_archive: Path, shingle_size: int) -> set[int]:
    shingles = set()
    for c_file in extract_c_code_from_archive(repo_archive):
        shingles.update(
            sha1_hash32(c_file[i:(i + shingle_size)]) % MAX_HASH # % max_hash should no longer be necessary with sha1_hash32 function. TODO: check this.
            for i in range(len(c_file) - shingle_size)
        )
    return shingles

LEXER = CLexer()
def get_lexical_shingle_set(repo_archive: Path, shingle_size: int) -> set[int]:
    shingles = set()
    for c_file in extract_c_code_from_archive(repo_archive):
        tokens: list[str] = [t[1] for t in LEXER.get_tokens(c_file) if t[0] is not Whitespace]
        shingles.update(
            sha1_hash32(" ".join(tokens[i:(i + shingle_size)])) % MAX_HASH # % max_hash should no longer be necessary with sha1_hash32 function. TODO: check this.
            for i in range(len(tokens) - shingle_size)
        )
    return shingles

def minhash(repo_archive: Path, shingle_size: int, a: NDArray, b: NDArray, c: np.uint64, lexical: bool) -> MinHashedRepo | MinHashFailed:
    """Minhashes the repo in the archive using hash functions of the form (ax + b) % c, where
    c is a prime integer larger than the maximum value of x.

    The shape of the returned is equal to the number of hash functions used.
    """
    try:
        if lexical:
            shingle_set = get_lexical_shingle_set(repo_archive, shingle_size)
        else:
            shingle_set = get_textual_shingle_set(repo_archive, shingle_size)
    except Exception as e:
        return MinHashFailed(repo_archive, repr(e))

    # Happens when there is no text to hash or the amount of text is smaller than the shingle size.
    if len(shingle_set) == 0:
        return MinHashFailed(repo_archive, "Empty shingle set.")
    x = np.fromiter(shingle_set, dtype=np.uint64)
    x = x.reshape((1, x.shape[0]))

    # Compute the hash functions.
    minimums = np.min((a @ x + b) % c, axis=1)
    return MinHashedRepo(repo_archive, minimums)

class UnionFind:
    def __init__(self, items: Iterable[T]):
        self.index: dict = {item: item for item in items}
        self.rank: dict = {item: 0 for item in self.index}
    
    def find(self, item: T) -> T:
        if self.index[item] != item:
            self.index[item] = self.find(self.index[item])
        return self.index[item]
    
    def union(self, a: T, b: T):
        x = self.find(a)
        y = self.find(b)
        if x == y:
            return
        
        if self.rank[x] > self.rank[y]:
            self.index[y] = x
        elif self.rank[x] < self.rank[y]:
            self.index[x] = y
        else:
            self.index[x] = y
            self.rank[y] += 1

    def export_sets(self) -> list[set[T]]: # type: ignore
        sets: dict[T, set[T]] = {}
        for item in self.index:
            representative = self.find(item)
            if representative in sets:
                sets[representative].add(item)
            else:
                sets[representative] = {item}
        return [s for _, s in sets.items()]

class LSH:
    """Find possible similar candidate pairs of repositories using locality sensitive hashing on 
    MinHash hash tables.

    Loosely based on 
      https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication.py#L238
    which is source code for the preprint paper
      "The Stack: 3 TB of permissively licensed source code" by Kocetkov et al.
    """
    def __init__(self, dataset: list[MinHashedRepo], num_tables: int, show_progress: bool = False):
        self.dataset = dataset
        # self.show_progress = show_progress
        self.num_hash_fns = dataset[0].minhash.shape[0]
        assert self.num_hash_fns % num_tables == 0, "Number of LSH hash tables should evenly divide the number of minhash functions."
        self.band_size = int(self.num_hash_fns / num_tables) # Number of rows (entries in each signature matrix) in each band.
        hash_tables: list[defaultdict[tuple[int,...], set[MinHashedRepo]]] = [defaultdict(set) for _ in range(num_tables)]

        # Once we've finished sorting elements into individual hash buckets, we'll then merge those buckets.
        # This can be done efficiently with a union-find data structure.
        union_find = UnionFind(self.dataset)

        if show_progress:
            dataset_iter = tqdm(dataset, dynamic_ncols=True, desc="Performing LSH...")
        else:
            dataset_iter = dataset
        for hashed_repo in dataset_iter:
            bands = self.split_into_bands(hashed_repo.minhash)
            for b, hash_table in zip(bands, hash_tables):
                hash_table[b].add(hashed_repo)

        if show_progress:
            hash_table_iter = tqdm(hash_tables, dynamic_ncols=True, desc="Clustering...")
        else:
            hash_table_iter = hash_tables

        # Functions that ended up in the same hash bucket belong in the same cluster. However, functions
        # will end up in num_tables different buckets (one for each table). Using a union-find structure
        # allows us to group together similarites that occur based on different tables.
        for hash_table in hash_table_iter:
            for cluster in hash_table.values():
                if len(cluster) <= 1:
                    continue
                an_element = min(cluster)
                for element in cluster:
                    # MinHashedRepos hash and compare based on their integer function IDs. This 
                    # makes doing such comparisons very efficient.
                    union_find.union(an_element, element)

        if show_progress:
            export_iter = tqdm(union_find.index, dynamic_ncols=True, desc="Exporting clusters...")
        else:
            export_iter = union_find.index
        clusterer: defaultdict[MinHashedRepo, list[MinHashedRepo]] = defaultdict(list)
        for fn in export_iter:
            clusterer[union_find.find(fn)].append(fn)

        self.clusters: list[list[MinHashedRepo]] = list(clusterer.values())
        print(f"Number of clusters: {len(self.clusters)}")
    
    
    def split_into_bands(self, signature: NDArray) -> list[tuple[int,...]]:
        """Precondition: signature is of shape (self.num_hash_fns,)
        """
        assert signature.shape == (self.num_hash_fns,), f"MinHash signatures should be of shape ({self.num_hash_fns},) but found {signature.shape} instead."
        return [
            tuple(signature[i:i+self.band_size])
            for i in range(0, self.num_hash_fns, self.band_size)
        ]


def repoiter(archive_location: Path) -> Iterable[Path]:
    assert archive_location.is_dir(), f"Archive location {archive_location} must be a directory."
    for owner in archive_location.iterdir():
        if owner.is_dir():
            for reponame in owner.iterdir():
                if reponame.suffixes == [".tar", ".gz"]:
                    yield reponame

def deduplicate(
        repo_paths: list[Path], 
        shingle_size: int,
        num_hash_fns: int,  
        num_tables: int,
        lexical: bool,
        workers: int,
        random_seed: int,
        ram_backed_tempfiles: bool = True
    ) -> tuple[list[set[str]], list[str]]:
    """Builds equivalence classes of repositories using MinHashing and locality-sensitve hashing (LSH).

    :param repo_paths: the location of each repository archive. Expected to be in tar.gz format.
    :param shingle_size: the size, in characters, of a shingle (contiguous chunk of a document) used in minhashing
    :param num_hash_fns: the number of permutations to use in minhashing
    :param num_tables: the number of tables used in locality sensitive hashing. Band size is num_hash_fns / num_tables; num_tables must evenly divide num_hash_fns
    :param lexical: whether to use segments of raw text (False) or subsequences of lexical tokens (True) for computing shingles.
    :param workers: number of multiprocessing workers to use. Provide 0 for single-process mode.
    :param random_seed: seed for numpy random state for generating permutations
    
    :returns: a list of the equivalence classes of repositories, as well as the repositories that could not be minhashed.
    """
    assert num_hash_fns % num_tables == 0, f"num_tables must evenly divide num_hash_fns!"
    np_random_state = np.random.RandomState(random_seed)
    a = np_random_state.randint(1, MERSENNE_PRIME, size=(num_hash_fns,1), dtype=np.uint64)
    b = np_random_state.randint(0, MERSENNE_PRIME, size=(num_hash_fns,1), dtype=np.uint64)

    if ram_backed_tempfiles:
        assert Path("/dev/shm").exists(), f"RAM-backed tempfiles are unavailable."
        tempfile.tempdir = "/dev/shm"

    if workers == 0:
        dataset = []
        try:
            for repo in tqdm(repo_paths, dynamic_ncols=True):
                dataset.append(minhash(repo, shingle_size, a, b, MERSENNE_PRIME, lexical))
        except KeyboardInterrupt:
            print(f"Currently processing {repo}")
            print(f"Moving to LSH with partial results...")
    else:
        with multiprocessing.Pool(workers) as pool:
            partial_minhash = functools.partial(minhash, shingle_size=shingle_size, a=a, b=b, c=MERSENNE_PRIME, lexical=lexical)
            # dataset = list(tqdm(pool.imap_unordered(partial_minhash, repo_paths), total=len(repo_paths)))
            try:
                dataset = []
                for item in tqdm(pool.imap_unordered(partial_minhash, repo_paths), total=len(repo_paths), dynamic_ncols=True):
                    dataset.append(item)
            except KeyboardInterrupt:
                print(f"Keyboard interrupt detected.")
                print(f"Computing unprocessed items...")
                incomplete = set(repo_paths) - set(d.repo for d in dataset)
                with open("deduplicate_debug.incomplete", "w") as fp:
                    for inc in incomplete:
                        fp.write(f"{inc}\n")

    dataset = [d for d in dataset if isinstance(d, MinHashedRepo)]
    uncomputable = [d.slug for d in dataset if isinstance(d, MinHashFailed)]

    lsh = LSH(dataset, num_tables, show_progress=True)
    equivalence_classes = [{repo.slug for repo in cluster} for cluster in lsh.clusters]
    
    return equivalence_classes, uncomputable

def main(args: argparse.Namespace):
    shingle_size: int = args.shingle_size
    workers: int = args.workers
    num_tables: int = args.num_tables
    output = Path(args.output)

    repo_paths = list(repoiter(Path(args.repo_archives)))
    if args.limit:
        repo_paths = repo_paths[:args.limit]
    
    print("Using lexical shingles." if args.lexical else "Using textual shingles.")
    clusters, uncomputible = deduplicate(repo_paths, shingle_size, args.num_hash_fns, num_tables, args.lexical, workers, args.random_seed)
    print(f"Unable to find code for {len(uncomputible)} repos.")

    if output.suffix != ".json":
        output = output.with_suffix(".json")
    with open(output, "w") as fp:
        json.dump({"clusters": [list(c) for c in clusters], "uncomputible": uncomputible}, fp)

if __name__ == "__main__":
    main(get_args())
