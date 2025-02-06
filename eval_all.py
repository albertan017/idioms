"""Run validation on checkpoints for all epochs. Ignores full exebench except for qwen-0.5b, which was trained for 8 epochs.
"""

import argparse
import subprocess
import re
import json
from pathlib import Path
from typing import Iterable

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default="")
    parser.add_argument("--exclude", type=str, help="Exclude runs containing this string.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--exebench-subpartition", default="real", choices=["real", "synth", "both"])
    parser.add_argument("--dry-run", action="store_true", help="Just generate the commands, don't actually run them.")
    parser.add_argument("--rerun", action="store_true", help="Instead, rerun the commands, using --evaluate-existing-predictions. Selects only runs that do have predictions.")
    parser.add_argument("--missing-predictions", action="store_true", help="Run with the --missing-tests-only flag. Selects only runs that do have predictions.")
    parser.add_argument("--test-best-checkpoints", action="store_true", help="Instead of running validation, run the best checkpoint on the test partition.")
    parser.add_argument("--best-checkpoint-save-path", default="results/best_checkpoints.json", help="Where to find the best checkpoints.")
    return parser.parse_args()

EXEBENCH_EVAL_DATASET = "exebench-hf-O0-eval"
NUM_EPOCHS = 8 # all experiments used 8 epochs except full exebench, which we're ignoring in this script.
MISSING_CHECKPOINTS = Path("missing_checkpoints.txt")

def select_checkpoints(run_dir: Path) -> Iterable[Path]:
    """Determine which checkpoints to validate.
    """
    checkpoints = list(c for c in run_dir.iterdir() if c.is_dir() and c.name.startswith("checkpoint"))
    if len(checkpoints) < NUM_EPOCHS:
        with open(MISSING_CHECKPOINTS, "a") as fp:
            fp.write(run_dir.name + "\n")
        yield from checkpoints
    elif len(checkpoints) == NUM_EPOCHS:
        yield from checkpoints
    else:
        by_step = {int(c.name.split("-")[1]): c for c in checkpoints}
        total_steps = max(by_step) # transformers Trainer saves a checkpoint at the maximum number of steps when done training.
        epoch_size = total_steps // NUM_EPOCHS # in number of steps
        for epoch in range(NUM_EPOCHS):
            tgt_steps = epoch * epoch_size + epoch_size
            closest_ckpt = min(by_step, key=lambda c: abs(c - tgt_steps))
            yield by_step[closest_ckpt]

def get_batch_size(run_name: str) -> int:
    """Extract the batch size from the run name.
    """
    match = re.search(r"""(\d+(\.\d+)?)b""", run_name)
    assert match is not None, run_name
    size = float(match.group(1)) # in billions of parameters
    if "neighbors" in run_name and size > 5:
        return 4
    else:
        if size > 5:
            return 16
        else:
            return 32
    
def make_base_command(checkpoint: Path, run_name, eval_partition: str):
    return ["python", "evaluator.py", str(checkpoint), "--eval-partition", eval_partition, "--batch-size", str(get_batch_size(run_name))]

def main(args: argparse.Namespace):
    if MISSING_CHECKPOINTS.exists():
        MISSING_CHECKPOINTS.unlink()
    results_dir = Path(args.results_dir)
    runs_dir = Path(args.runs_dir)
    pattern = "" if args.pattern is None else args.pattern
    exclude = args.exclude
    rerun: bool = args.rerun
    missing_predictions: bool = args.missing_predictions
    use_existing = rerun or missing_predictions
    assert not (rerun and missing_predictions), f"--rerun and --missing-predictions options are contradictory."
    do_test: bool = args.test_best_checkpoints
    eval_partition = "test" if do_test else "validation"
    subpartitions = []
    if args.exebench_subpartition == "both" or args.exebench_subpartition == "real":
        subpartitions.append("real")
    if args.exebench_subpartition == "both" or args.exebench_subpartition == "synth":
        subpartitions.append("synth")

    if do_test:
        with open(args.best_checkpoint_save_path, "r") as fp:
            best_checkpoints = json.load(fp)

    # checkpoints_to_eval: list[tuple[Path, str, bool]] = []
    commands: list[list[str]] = []
    for run in runs_dir.iterdir():
        if pattern in run.name and (exclude is None or exclude not in run.name): #and not run.name.split("-", maxsplit=2)[-1] == "exebench-O0":
            try:
                get_batch_size(run.name) # all valid (non-development) runs have a size in their run names.
            except:
                continue

            # Select checkpoints depending on the run mode. Only want to evaluate the best one for test.
            if do_test:
                if run.name in best_checkpoints:
                    ckpt: Path = runs_dir / run.name / best_checkpoints[run.name]
                    assert ckpt.exists(), f"Checkpoint {ckpt} does not exist!"
                    checkpoints = [ckpt]
                else:
                    continue
            else:
                if run.name.split("-", maxsplit=2)[-1] == "exebench-O0" and "qwen" not in run.name:
                    continue
                checkpoints = select_checkpoints(run)

            # Generate the commands.
            for checkpoint in checkpoints:
                if "exebench" in run.name:
                    for subpartition in subpartitions:
                        results_file_exists = (results_dir / run.name / checkpoint.name / EXEBENCH_EVAL_DATASET / f"{eval_partition[:5]}_real_results.json").exists()
                        if (use_existing and results_file_exists) or (not use_existing and not results_file_exists):
                            commands.append(make_base_command(checkpoint, run.name, eval_partition) + [
                                "--dataset", EXEBENCH_EVAL_DATASET, "--exebench-subpartition", subpartition, "--no-exebench-tests"
                            ])
                else:
                    results_file_exists = (results_dir / run.name / checkpoint.name / f"{eval_partition}_results.json").exists()
                    if (use_existing and results_file_exists) or (not use_existing and not results_file_exists):
                        commands.append(make_base_command(checkpoint, run.name, eval_partition))
    
    if rerun:
        for command in commands:
            command.append("--evaluate-existing-predictions")
    if missing_predictions:
        for command in commands:
            command.append("--missing-predictions-only")


    print("#### Command preview:")
    for command in commands:
        print(" ".join(command))
    print(len(commands), "total commands")

    assert len(commands) == len({" ".join(c) for c in commands}), "There are duplicate commands!"
    
    if not args.dry_run:
        print("\n\n\n\n" + "#" * 8, "Running commands", "#" * 8)
        for command in commands:
            print(" ".join(command))
            subprocess.run(command)
            print("\n\n")

if __name__ == "__main__":
    main(get_args())
            