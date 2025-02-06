"""Utils for working with huggingface transformers models.
"""

import itertools
from collections import deque
from typing import Iterable

from transformers import PreTrainedTokenizerBase

from idioms.data.dataset import MatchedFunction, MatchedBinary

DECOMPILED_ORIG_SEP = "@@"
TARGET_FN_NAME_SEP = "##"

def nhop_neighbors(start: str, graph: dict[str, list[str]], hops: int) -> Iterable[str]:
    """Return the neighbors in the graph that can be reached from the start point by traversing at most `hops` edges.
    The return value is ordered in terms of least to greatest distance, including the start (at distance 0.)

    :param start: the start node of the search, represented by a string label.
    :param graph: an adjacency list representing the graph
    :param hops: the maximum number of edges to traverse.
    :returns: an iterable over the neighbors, in the order from closest to the start to farthest away.
    """
    # Use insertion-order property of dictionaries to make sure the neighbors are properly ordered.
    neighbors: dict[str, int] = {start: 0}
    frontier: deque[str] = deque()
    depths: deque[int] = deque()
    frontier.append(start)
    depths.append(0)
    while len(frontier) > 0:
        node = frontier.popleft()
        depth = depths.popleft()
        next_depth = depth + 1
        if next_depth <= hops:
            for neighbor in graph[node]:
                if neighbor not in neighbors:
                    neighbors[neighbor] = next_depth
                    frontier.append(neighbor)
                    depths.append(next_depth)

    return neighbors

### Stringify functions. Used in both training and evaluation.
def stringify_function_target(fn: MatchedFunction) -> str:
    return fn.canonical_original_code + "\n" + "\n".join(udt.declaration("") + ";" for udt in fn.user_defined_types)

def causal_stringify_function_prompt(fn: MatchedFunction) -> str:
    return fn.canonical_decompiled_code + DECOMPILED_ORIG_SEP

def _causal_stringify_functions_prompt(functions: Iterable[str]):
    return "\n\n".join(functions)

def causal_stringify_neighbors_prompt(binary: MatchedBinary, fn: MatchedFunction, nhops: int, tokenizer: PreTrainedTokenizerBase | None = None, max_context: int | None = None) -> str:
    neighbors = nhop_neighbors(fn.name, binary.call_graph, nhops)
    lookup = binary.canonical_decompiled_code_lookup
    context = _causal_stringify_functions_prompt(lookup[name] for name in neighbors)
    remainder = TARGET_FN_NAME_SEP + fn.canonical_name + DECOMPILED_ORIG_SEP
    if max_context is not None:
        encoded = tokenizer.encode(context)
        reserved = len(tokenizer.encode(remainder))
        if len(encoded) > max_context - reserved:
            context = tokenizer.decode(encoded[:(max_context - reserved)])
    return context + remainder


def causal_stringify_binary_prompt(binary: MatchedBinary, fn: MatchedFunction) -> str:
    return _causal_stringify_functions_prompt(itertools.chain(
        (f.canonical_decompiled_code for f in binary.functions),
        binary.unmatched.values() # values are the canonical code for the unmatched functions.
    )) + TARGET_FN_NAME_SEP + fn.canonical_name + DECOMPILED_ORIG_SEP




