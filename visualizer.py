"""Visualize results by putting them into tables.
"""

import argparse
import json
import itertools
import pickle
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from evaluator import FunctionEvaluator
from idioms.data.dataset import MatchedFunction

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("x_axis", choices=["model", "run", "metric"], help="x_axis in the output graphs")
    parser.add_argument("y_axis", choices=["model", "run", "metric"], help="y_axis in the output graph.")
    parser.add_argument("--cache", type=str, help="Stores the result of the state of the results directory.")
    # parser.add_argument("--no-recompute-results", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--eval-partition", choices=["validation", "test"], default="test")
    parser.add_argument("--exebench-subpartition", choices=["both", "real", "synth"], default="real")
    parser.add_argument("--checkpoint-type", choices=["best", "last"], default="best")
    parser.add_argument("--table-type", choices=["ascii", "latex"], default="ascii")
    return parser.parse_args()

MAX_COLUMN_WIDTH = None
DECIMAL_PLACES = 2
BEST_CHECKPOINT_LOCATION = "results/best_checkpoints.json"

MODEL_NAMES = [
    "qwen-0.5b",
    "llm4decompile-1.3b",
    "codegemma-2b",
    "codegemma-7b",
    "codellama-7b"
]

RUN_TYPES = [
    "exebench-O0",
    "parity-exebench-O0", 
    "functions-idioms", 
    "neighbors-idioms"
]

EXEBENCH_RUN_TYPES = [
    "exebench-O0",
    "parity-exebench-O0"
]

METRICS = [
    "perfectly_aligned",
    "perfectly_aligned_and_typechecks",
    "variable_name_accuracy",
    "variable_type_accuracy",
    "variable_udt_exact_matches",
    "variable_udt_composition_matches",
    "bleu",
]

EXEBENCH_METRICS = [
    "exebench_correct",
    "exebench_partially_correct",
    "exebench_total_errors",
    "exebench_compilation_errors"
]
EXEBENCH_METRICS.extend([metric + "_permissive" for metric in EXEBENCH_METRICS])

AXIS_INDICES = {
    "model": MODEL_NAMES,
    "run": RUN_TYPES,
    "metric": METRICS
}

EXEBENCH_AXIS_INDICES = {
    "model": MODEL_NAMES,
    "run": EXEBENCH_RUN_TYPES,
    "metric": EXEBENCH_METRICS
}

class VisualizationTensor:
    def __init__(self, data: NDArray, axes: list[str], labels: dict[str, list[str]]):
        assert len(data.shape) == len(axes), f"Not enough axis labels for data of shape {data.shape}: {len(axes)}"
        self.data: NDArray = data
        self._init_axes(axes)
        self.labels = labels

    def _init_axes(self, axes: list[str]):
        self.axes: dict[str, int] = {axis: i for i, axis in enumerate(axes)}

    def permute(self, new_axes: list[str]):
        """Permute axes of the tensor such that the axes of this tensor
        match the order of new_axes.
        """
        permutation = tuple(self.axes[axis] for axis in new_axes)
        self.data = self.data.transpose(permutation)
        self._init_axes(new_axes)

    def make_tables(self, table_callback):
        if len(self.axes) == 3:
            table_context, y_axis, x_axis = self.axes
            context_axis_labels = self.labels[table_context]
            for i, data_slice in enumerate(self.data):
                yield table_callback(data_slice, x_axis, y_axis, self.labels, title=context_axis_labels[i])
        else:
            raise ValueError(f"Making tables for {len(self.axes)}-dimension data is not supported.")

def format_table_number(n) -> str:
    """Format a number in an easy-to-read way for a table, or return a placeholder string is the value is
    missing (represented by nan.)
    """
    n = float(n)
    if math.isnan(n):
        return "--"
    return f"{round(n * 100, DECIMAL_PLACES):.{DECIMAL_PLACES}f}"

def make_terminal_table(data: NDArray, x_axis: str, y_axis: str, labels: dict[str, list[str]], title: str | None = None):
    """Make an ascii table designed to be displayed in a terminal window or plain text file.

    data: NDArray, shape (# rows/y-size, # cols/x-size). Unintuitively, y/rows is first because we generate
          the table as a sequence of lines of text representing rows, and then join them on newline.
    x_axis: the name of the x axis.
    y_axis: the name of the y axis.
    title: a title for the plot.
    """
    
    x_labels = labels[x_axis]
    y_labels = labels[y_axis]

    first_col_width = max(len(lab) for lab in y_labels) # Contains the label names, which we don't want to cut off.
    max_width = max(len(lab) for lab in x_labels)
    col_width = max_width if MAX_COLUMN_WIDTH is None else min(max_width, MAX_COLUMN_WIDTH)
    entry_width = col_width - 1

    total_width = col_width * len(x_labels) + first_col_width + 1
    if title is not None:
        if len(title) + 8 < total_width:
            title = f"### {title} ###"
        title_centerizing_space = " " * (total_width // 2 - len(title) // 2)
        output = [title_centerizing_space + "#" * len(title), title_centerizing_space + title, title_centerizing_space + "#" * len(title)]
    else:
        output = []
    output.append(" " * first_col_width + "| " + " ".join(lab[:entry_width].rjust(entry_width) for lab in x_labels))
    output.append("-" * total_width)
    for i, row in enumerate(data):
        outrow = y_labels[i].ljust(first_col_width) + "| " + " ".join(f"{round(float(d) * 100, DECIMAL_PLACES):.{DECIMAL_PLACES}f}".rjust(entry_width) for d in row)
        output.append(outrow)
    return "\n".join(output)

def make_latex_table(data: NDArray, x_axis: str, y_axis: str, labels: dict[str, list[str]], title: str | None = None):
    x_labels = labels[x_axis]
    y_labels = labels[y_axis]

    if title is not None:
        output = ["\\caption{" + title.replace("_", " ") + "\\label{tab:" + title.replace("-", "_") +  "}}"]
    else:
        output = []

    output.append("\\begin{tabular}{" + "l" + "r" * len(x_labels) + "}")
    output.append("\\toprule")
    output.append("&" + "&".join(x_labels) + "\\\\")
    output.append("\\midrule")
    for i, row in enumerate(data):
        output.append(y_labels[i].replace("_", " ") + " & " + " & ".join(format_table_number(d) for d in row) + "\\\\")
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    return "\n".join(output)


def read_results(
        run_results_dir: Path,
        evaluator: FunctionEvaluator,
        eval_partition: str, 
        exebench_subpartition: str | None,
    ):
    if exebench_subpartition is not None:
        if eval_partition == "validation":
            eval_partition = "valid"
        raw_results = []
        for subpartition in ("real", "synth") if exebench_subpartition == "both" else (exebench_subpartition,):
            with open(run_results_dir / f"{eval_partition}_{subpartition}_results.json", "r") as fp:
                raw_results.extend(
                    (MatchedFunction.from_json(fn), prediction)
                    for fn, prediction in json.load(fp)
                )
    else:
        with open(run_results_dir / f"{eval_partition}_results.json", "r") as fp:
            existing = set()
            raw_results = []
            for fn, prediction in json.load(fp):
                if fn["canonical_original_code"] not in existing:
                    raw_results.append((MatchedFunction.from_json(fn), prediction))
                    existing.add(fn["canonical_original_code"])

    scores = evaluator(raw_results)
    return {metric: scores[metric] for metric in METRICS}

def read_exebench_scores(run_results_dir: Path, eval_partition: str, exebench_subpartition: str):
    """Load exebench metrics from disk and return, computing a weighted average if both exebench
    subpartitions are specified.
    
    Don't recalculate these. That would take forever and there's no need because there aren't 
    exebench metrics that have the same base-rate errors that codealign metrics would have.
    """
    if eval_partition == "validation":
        eval_partition = "valid"
    if exebench_subpartition == "real" or exebench_subpartition == "both":
        with open(run_results_dir / f"{eval_partition}_real_exebench_scores.json", "r") as fp:
            real = json.load(fp)
        if exebench_subpartition == "real":
            return {metric: real[metric] for metric in EXEBENCH_METRICS}
    if exebench_subpartition == "synth" or exebench_subpartition == "both":
        with open(run_results_dir / f"{eval_partition}_synth_exebench_scores.json"):
            synth = json.load(fp)
        if exebench_subpartition == "synth":
            return {metric: synth[metric] for metric in EXEBENCH_METRICS}
    assert eval_partition == "both"
    # Do a weighted average of the metrics
    return {
        metric: (real[metric] * len(real) + synth[metric] * len(synth)) / (len(real) + len(synth))
        for metric in EXEBENCH_METRICS 
    }
        

def main(args: argparse.Namespace):
    eval_partition: str = args.eval_partition
    exebench_subpartition: str = args.exebench_subpartition
    results_dir = Path(args.results_dir)
    use_best = args.checkpoint_type == "best"
    if args.cache is None:
        cache = None
    else:
        cache = Path(args.cache)
        cache = cache.with_suffix(".pkl")
    assert results_dir.exists(), f"Results dir {results_dir} does not exist!"
    assert results_dir.is_dir(), f"Results dir {results_dir} is not a directory!"
    # run_type_indices: dict[str, int] = {t: i for i, t in enumerate(RUN_TYPES)}
    # recompute: bool = not args.no_recompute_results
    x_axis = args.x_axis
    y_axis = args.y_axis

    if use_best:
        with open(BEST_CHECKPOINT_LOCATION, "r") as fp:
            use_checkpoint: dict[str, str] = json.load(fp)

    if cache is None or not cache.exists():
        # Default axes:
        # 1. model name
        # 2. run type (dataset/context type)
        # 3. metrics
        evaluator = FunctionEvaluator()

        results: list[list[None | dict]] = [[None] * len(RUN_TYPES) for _ in range(len(MODEL_NAMES))]
        exebench_results: list[list[None | dict]] = [[None] * 2 for _ in range(len(MODEL_NAMES))]

        for (i, model_name), (j, run_type) in tqdm(itertools.product(enumerate(MODEL_NAMES), enumerate(RUN_TYPES)), total=len(MODEL_NAMES) * len(RUN_TYPES)):
            run_name = f"{model_name}-{run_type}"
            run_results_dir = results_dir / run_name
            if not run_results_dir.exists():
                continue

            if use_best and (run_type != "exebench-O0" or "qwen" in model_name):
                if run_name in use_checkpoint:
                    run_results_dir = run_results_dir / use_checkpoint[run_name]
                else:
                    continue
            if "exebench-O0" in run_type:
                run_results_dir = run_results_dir / "exebench-hf-O0-eval"
            
            try:
                print(f"Using {run_results_dir}")
                results[i][j] = read_results(run_results_dir, evaluator, eval_partition, exebench_subpartition if "exebench" in run_type else None)
            except FileNotFoundError:
                pass

            if "exebench" in run_type:
                try:
                    exebench_results[i][j] = read_exebench_scores(run_results_dir, eval_partition, exebench_subpartition)
                except (FileNotFoundError, KeyError):
                    pass

        metrics = []
        axes = ["model", "run", "metric"]
        for (_results, n_run_types, _metrics) in ((results, len(RUN_TYPES), METRICS), (exebench_results, 2, EXEBENCH_METRICS)):
            array = np.full((len(MODEL_NAMES), n_run_types, len(_metrics)), float("nan"))
            for i, run in enumerate(_results):
                for j, scores in enumerate(run):
                    if scores is not None:
                        for k, metric in enumerate(_metrics):
                            array[i][j][k] = scores[metric]
            metrics.append(array)

        if cache is not None:
            with open(cache, "wb") as fp:
                pickle.dump((metrics, axes), fp)
        
    else:
        with open(cache, "rb") as fp:
            metrics, axes = pickle.load(fp)

    make_table_fn = make_terminal_table if args.table_type == "ascii" else make_latex_table
    
    print_sep = False
    for m, labels in zip(metrics, (AXIS_INDICES, EXEBENCH_AXIS_INDICES)):
        regular_results = VisualizationTensor(m, axes, labels)
        new_axes = [axis for axis in axes if axis != x_axis and axis != y_axis]
        new_axes.append(y_axis)
        new_axes.append(x_axis)
        regular_results.permute(new_axes)
        
        if print_sep:
            print("\n\n")
            print("*" * 40)
            print("*" * 40)
            print("*" * 40)
        
        for table in regular_results.make_tables(make_table_fn):
            print()
            print()
            print(table)
        
        print_sep = True


if __name__ == "__main__":
    main(get_args())


    

    