"""Run all exebench tests
"""

import argparse
import json
import sys
import subprocess
import itertools
from pathlib import Path

RESULTS_DIR = Path("results")
OVERWRITE = True

PARTITION = ("test",)
SUBPARTITION = ("real",)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="If exebench results already exist, run the tests again and overwrite the existing scores.")
    parser.add_argument("--dry-run", action="store_true", help="Only show the commands, don't actually run them.")
    parser.add_argument("--exclude", type=str, help="Exclude commands with this string in their name.")
    parser.add_argument("--require", type=str, help="Require commands to have this string in their name.")
    parser.add_argument("--batches", type=int, help="Generate bash scripts that correspond to one batch each.")
    return parser.parse_args()

# python evaluator.py runs/codegemma-2b-exebench-O0/ --dataset exebench-hf-O0-eval/ --eval-partition test --exebench-subpartition synth --batch-size 8 --no-exebench-tests
def make_command(run_name: str, checkpoint: str | None = None, dataset: str | None = None, eval_partition: str | None = None, exebench_subpartition: str | None = None) -> list[str]:
    command = ["python", "evaluator.py"]
    if checkpoint is None:
        command.append(f"runs/{run_name}")
    else:
        command.append(f"runs/{run_name}/{checkpoint}")
    command.append("--evaluate-existing-predictions")
    if dataset is not None:
        command.append("--dataset")
        command.append(dataset)
    if eval_partition is not None:
        command.append("--eval-partition")
        command.append("validation" if eval_partition == "valid" else eval_partition)
    if exebench_subpartition is not None:
        command.append("--exebench-subpartition")
        command.append(exebench_subpartition)
    return command

def eligible_for_exebench_tests(directory: Path, prefix: str, overwrite: bool) -> bool:
    if not (directory / f"{prefix}_results.json").exists():
        return False # Can only run the tests if there are predictions to run them on.

    if overwrite:
        return True
    
    try:
        with open(directory / f"{prefix}_scores.json", "r") as fp:
            scores = json.load(fp)
    except FileNotFoundError:
        print(f"WARNING: results file exists for {directory} but scores file does not.", file=sys.stderr)
        return True
    
    return not any("exebench" in metric for metric in scores)

def main():
    args = get_args()
    overwrite: bool = args.overwrite
    dry_run: bool = args.dry_run
    exclude: str | None = args.exclude
    require: str | None = args.require
    batches: int | None = args.batches

    with open("results/best_checkpoints.json", "r") as fp:
        best_checkpoints: dict[str, str] = json.load(fp)

    commands = []
    for rundir in RESULTS_DIR.iterdir():
        if (exclude is not None and exclude in rundir.name) or \
           (require is not None and require not in rundir.name):
            continue

        scores_dir = rundir
        if "exebench-O0" in rundir.name:
            if "parity" in rundir.name or rundir.name in best_checkpoints:
                if rundir.name in best_checkpoints:
                    checkpoint = best_checkpoints[rundir.name]
                    scores_dir = scores_dir / checkpoint / "exebench-hf-O0-eval"
                else:
                    continue
            else:
                checkpoint = None
                scores_dir = scores_dir / "exebench-hf-O0-eval"
            for partition, subpartition in itertools.product(PARTITION, SUBPARTITION):
                if eligible_for_exebench_tests(scores_dir, f"{partition}_{subpartition}", overwrite):
                    commands.append(make_command(rundir.name, checkpoint, "exebench-hf-O0-eval", partition, subpartition))

    commands.sort(key=lambda x: x[2])

    print(f"Commands to be run:")
    for command in commands:
        print(" ".join(command))
    print(f"{len(commands)} total commands.")
    print()

    if batches is not None:
        batch_size = int(len(commands) / batches)
        for i in range(batches):
            with open(f"batch{i}_exebench.sh", "w") as fp:
                for command in commands[(i * batch_size):((i + 1) * batch_size)]:
                    fp.write(" ".join(command) + "\n")
    elif not dry_run:
        for i, command in enumerate(commands):
            print(f"Running command {i}/{len(commands)}")
            print(" ".join(command))
            subprocess.run(command)
            print("\n\n")

if __name__ == "__main__":
    main()