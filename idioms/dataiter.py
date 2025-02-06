"""Load data and prepare it for input during training.
"""

import tarfile
import random
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Iterable, Optional, Callable
if TYPE_CHECKING:
    from _typeshed import StrPath

from torch.utils.data import IterableDataset

from idioms.data.dataset import MatchedFunction, MatchedBinary

class IdiomsDataset(IterableDataset):
    def __init__(self, shards: Iterable["StrPath"], length_cache: Optional["StrPath"] = None, shuffle: bool=True):
        self.shards = list(shards)
        assert len(self.shards) > 0, f"Dataset has no shards."
        self.sizes = None
        self.length_cache = None if length_cache is None else Path(length_cache)
        self.shuffle = shuffle
    
    def _iter_tar(self, tar: "StrPath") -> Iterator[MatchedBinary]:
        with tarfile.open(tar, "r") as tf:
            for member in tf:
                f = tf.extractfile(member)
                binary = MatchedBinary.from_json(json.load(f)) # type: ignore
                yield binary

    def compute_sizes(self) -> list[int]:
        """Returns a list recording the number of functions in each binary in the dataset.
        This is expensive, as it involves loading the entire dataset to count each example.
        """
        sizes = []
        for shard in self.shards:
            for binary in self._iter_tar(shard):
                sizes.append(len(binary.functions))
        return sizes

    def _len_cache(self) -> list[int]:
        if self.sizes is None:
            if self.length_cache is not None:
                if self.length_cache.exists():
                    with open(self.length_cache, "rb") as fp:
                        self.sizes = pickle.load(fp)
                else:
                    self.sizes = self.compute_sizes()
                    with open(self.length_cache, "wb") as fp:
                        pickle.dump(self.sizes, fp)
            else:
                self.sizes = self.compute_sizes()
        return self.sizes

class MatchedFunctionDataset(IdiomsDataset):
    def __iter__(self) -> Iterator[MatchedFunction]:
        for shard in self.shards:
            binaries = list(self._iter_tar(shard))
            if self.shuffle:
                random.shuffle(binaries)
            for binary in binaries:
                for fn in binary.functions:
                    yield fn
        # Shuffle the shards at the end of each epoch
        if self.shuffle:
            random.shuffle(self.shards)

    def __len__(self) -> int:
        if not hasattr(self, "length"):
            self.length = sum(self._len_cache())
        return self.length

class MatchedBinaryDataset(IdiomsDataset):
    def __iter__(self) -> Iterator[MatchedBinary]:
        for shard in self.shards:
            binaries = list(self._iter_tar(shard))
            if self.shuffle:
                random.shuffle(binaries)
            yield from binaries
            
    def __len__(self) -> int:
        return len(self._len_cache())
    
class MatchedBinaryFunctionWrapper(IterableDataset):
    def __init__(self, 
                 binary_dataset: MatchedBinaryDataset, 
                 max_fns_per_binary: int | None = None, 
                 function_filter: Callable[[MatchedFunction], bool] | None = None,
                 binary_filter: Callable[[MatchedBinary], bool] | None = None):
        self.dataset = binary_dataset
        self.max_fns_per_binary = max_fns_per_binary
        self.function_filter = function_filter
        self.binary_filter = binary_filter

    def __iter__(self) -> Iterator[tuple[MatchedBinary, int]]:
        for binary in filter(self.binary_filter, self.dataset):
            if self.function_filter is None:
                fnids = range(len(binary.functions))
            else:
                fnids = [i for i in range(len(binary.functions)) if self.function_filter(binary.functions[i])]

            if self.max_fns_per_binary is not None and self.max_fns_per_binary < len(fnids):
                fnids = random.sample(fnids, self.max_fns_per_binary)

            for i in fnids:
                yield binary, i
    
    def __len__(self) -> int:
        if not hasattr(self, "length"):
            if self.max_fns_per_binary is None:
                self.length = sum(self.dataset._len_cache())
            else:
                self.length = sum((self.max_fns_per_binary if l > self.max_fns_per_binary else l) for l in self.dataset._len_cache())
        return self.length