"""Put all checkpoints in a table
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


MAX_COLUMN_WIDTH = 20
DECIMAL_PLACES = 2

METRICS = [
    "perfectly_aligned",
    "perfectly_aligned_and_typechecks",
    "variable_name_accuracy",
    "variable_type_accuracy",
    "variable_udt_exact_matches",
    "variable_udt_composition_matches",
    "bleu",
]

def select_best_checkpoint(scores: NDArray, checkpoint_names: list[str], selection_metrics: set[str]) -> str:
    """Select the best checkpoint based on the normalized relevant scores. Values are normalized to the range observed in the checkpoint.
    """
    # Account for the fact that different scores vary in different ways.
    min_scores = scores.min(axis=1, keepdims=True)
    scores_range = (scores.max(axis=1, keepdims=True) - min_scores)
    # If the range is 0 (all values of this metric are the same) we get a division by zero error.
    # Setting the divisor to anything other than zero works, since the score contributed by this metric will always be the same.
    scores_range[scores_range == 0.0] = 1
    standardized = (scores - min_scores) / scores_range
    
    # Get only scores for the desired metrics
    desired_metrics = np.array([[int(m in selection_metrics) for m in METRICS]]).T
    checkpoint_summary_scores: NDArray = (standardized * desired_metrics).sum(axis=0)
    return checkpoint_names[checkpoint_summary_scores.argmax(axis=0)]


def make_terminal_table(data: NDArray, x_labels: list[str], y_labels: list[str], title: str | None = None):
    """Make an ascii table designed to be displayed in a terminal window or plain text file.

    data: NDArray, shape (# rows/y-size, # cols/x-size). Unintuitively, y/rows is first because we generate
          the table as a sequence of lines of text representing rows, and then join them on newline.
    x_axis: the name of the x axis.
    y_axis: the name of the y axis.
    title: a title for the plot.
    """
    first_col_width = max(len(lab) for lab in y_labels) # Contains the label names, which we don't want to cut off.
    max_width = max(len(lab) for lab in x_labels)
    col_width = max_width if MAX_COLUMN_WIDTH is None else min(max_width + 1, MAX_COLUMN_WIDTH)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path_or_name", nargs="?")
    args = parser.parse_args()
    if args.run_path_or_name is None:
        runs = [d.name for d in Path("results").iterdir() if d.is_dir()]
        runs.sort(key=lambda n: n.split("-", maxsplit=2)[-1])
    else:
        runs = [Path(args.run_path_or_name).name]

    best_checkpoints: dict[str, str] = {}

    for run_name in runs:
        is_exebench = "exebench" in run_name
        scores_file_name = "valid_real_scores.json" if is_exebench else "validation_scores.json"

        run_dir = Path("results") / run_name
        checkpoint_dirs = list(r for r in run_dir.iterdir() if r.name.startswith("checkpoint"))
        checkpoint_dirs.sort(key=lambda d: int(d.name.split("-")[1]))
        # names = ["checkpoint-" + d.name.split("-")[1] for d in checkpoint_dirs]
        names = [d.name for d in checkpoint_dirs]
        # if (run_dir / scores_file_name).exists() or (run_dir / "exebench-hf-O0-eval" / scores_file_name).exists():
        #     checkpoint_dirs.append(run_dir)
        #     names.append("ckpt-last")

        if len(names) == 0:
            continue

        values = []
        for checkpoint in checkpoint_dirs:
            if is_exebench:
                checkpoint = checkpoint / "exebench-hf-O0-eval"
            try:
                with open(checkpoint / scores_file_name, "r") as fp:
                    scores = json.load(fp)
                values.append([
                    scores[metric] for metric in METRICS
                ])
            except FileNotFoundError:
                values.append([float("nan")] * len(METRICS))

        values = np.array(values).T

        print(make_terminal_table(values, names, METRICS, run_name))
        print()

        # Metrics we use to decide which checkpoint to use for testing.
        selection_metrics = {"perfectly_aligned", "perfectly_aligned_and_typechecks"}
        if not is_exebench:
            # There are so few structs in the "real" exebench partition that these scores will be very
            # high variance; only consider this for the idioms dataset, which has many more structs.
            selection_metrics.add("variable_udt_composition_matches")

        if np.isnan(values).sum() == 0:
            if len(names) >= 8:
                best_checkpoint = select_best_checkpoint(values, names, selection_metrics)
                print("Best checkpoint:", best_checkpoint)
                best_checkpoints[run_name] = best_checkpoint
                if len(names) > 8:
                    print(f"WARNING: Run {run_name} has more than 8 checkpoints! Best checkpoint is {best_checkpoint}")

    best_checkpoint_save_path = "results/best_checkpoints.json"
    with open(best_checkpoint_save_path, "w") as fp:
        json.dump(best_checkpoints, fp, indent=2)

    print(f"Wrote {len(best_checkpoints)} best checkpoints to {best_checkpoint_save_path}")
    
    

if __name__ == "__main__":
    main()
