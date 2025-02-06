"""Extract preprocessed C code for all repositories we have binaries for.
"""

import argparse
import json
import sys
import tempfile
import functools
import tarfile
import subprocess
import os
import random
import traceback
import multiprocessing
from pathlib import Path
from typing import NamedTuple, Any, Iterator

from tqdm import tqdm

import ghcc

COMPILER = "gcc" # used for running the preprocessor, not actually compiling.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("binaries_dir")
    parser.add_argument("archive_location")
    parser.add_argument("output_location")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-example timeout")
    parser.add_argument("--max-repos", default=None, type=int, help="Stop processing repos after this many.")
    parser.add_argument("--workers", default=0, type=int, help="Number of worker processes")
    return parser.parse_args()


class PreprocessResult:
    def __init__(self, repo: str):
        self.repo = repo
    
    def to_json(self):
        raise NotImplementedError()

class Failure(PreprocessResult):
    def __init__(self, repo: str, stage: str, error: str):
        super().__init__(repo)
        self.stage = stage
        self.error = error
    
    def to_json(self):
        return {
            "repo": self.repo,
            "stage": self.stage,
            "error": self.error
        }

class Success(PreprocessResult):
    class Makefile:
        def __init__(self, mkinfo: dict[str, Any]):
            self.directory = mkinfo['directory'] # The relative path of the makefile in the repository
            self.success = mkinfo['success'] # Whether the makefile completed without errors
            self.preprocessed_files = mkinfo['binaries'] # the preprocessed files (in normal usage without -E, this would be binaries, hence the dictionary key name)
            self.hashes = mkinfo['sha256']
            self.json_repr = mkinfo

    def __init__(self, repo: str, makefiles: list[dict[str, Any]]):
        super().__init__(repo)
        self.makefiles = [Success.Makefile(m) for m in makefiles]

    def count_preprocessed_files(self) -> int:
        return sum(len(m.preprocessed_files) for m in self.makefiles)
    
    def to_json(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "makefiles": [m.json_repr for m in self.makefiles]
        }
        

def exception_handler(e: Exception, repo: str):
    # TODO: make this better
    print(f"Exception {e} occurred when processing {repo}")


def extract_original_code(archive: Path, repo: str, output_loc: Path, timeout: int):
    assert archive.name.endswith(".tar.gz"), f"Repo archive {archive} is not a tar.gz file!"

    with tempfile.TemporaryDirectory() as td:
        tempdir = Path(td)
        try:
            with tarfile.open(archive, mode="r:gz") as tf:
                repo_archive_dir_name = tf.getnames()[0] # the name of the root directory of the archive.
                tf.extractall(tempdir, filter="data")
        except Exception as e:
            return Failure(repo, "tarfile", repr(e))
        
        raw_code_dir = tempdir / (repo_archive_dir_name) # Where the contents of the repository are stored

        # preprocessed_dir = tempdir / "bin"
        # preprocessed_dir.mkdir()
        output_loc.mkdir(exist_ok=True, parents=True)

        directory_mapping = None # Use this to map additional directories on the host machine to directories in Docker.
        gcc_flags = "-E -P" # -E: preprocess only, -P: no linemarkers

        # We're technically not actually compiling here, just running the preprocessor.
        makefiles = ghcc.docker_batch_compile(
            str(output_loc), str(raw_code_dir), COMPILER, compile_timeout=timeout,
            gcc_override_flags=gcc_flags, extract_preprocessed_files=True, directory_mapping=directory_mapping,
            user_id=30101,  # user IDs 30000 ~ 39999
            exception_log_fn=functools.partial(exception_handler, repo=repo))


        # For the time being, assume that these are all things that we want.
        # TODO: filter out only the things that we want (e.g. no products of ./configure)
        # if len(diff_files) > 0:
        #     output_loc.mkdir(parents=True, exist_ok=True)
        #     for f in (Path(f) for f in diff_files):
        #         relpath = f.relative_to(raw_code_dir)
        #         outpath = output_loc / relpath
        #         outpath.parent.mkdir(parents=True, exist_ok=True)
        #         shutil.copyfile(f, outpath)
    
    return Success(repo, makefiles)

# For multiprocessing, if necessary.
def extract_original_code_wrapper(args: tuple[Path, str, Path, int]) -> PreprocessResult:
    try:
        return extract_original_code(*args)
    except Exception as e:
        return Failure(args[1], "unknown", repr(e))

def main(args: argparse.Namespace):
    # if not ghcc.utils.verify_docker_image(verbose=True):
    #     sys.exit(1)

    random.seed(80)

    # Use RAM-backed memory for tmp if available
    if os.path.exists("/dev/shm"):
        print("Ram-backed tempfiles are available!")
        tempfile.tempdir = "/dev/shm"

    repos = []
    for owner in Path(args.binaries_dir).iterdir():
        if owner.is_dir():
            for reponame in owner.iterdir():
                if reponame.is_dir() and len(os.listdir(reponame)) > 0:
                    repos.append(f"{owner.name}/{reponame.name}")
    
    print(f"Found {len(repos)} repositories. Sample:")
    print("\n".join(repos[:5]))
    print()

    archive_base_path = Path(args.archive_location)
    output_base_path = Path(args.output_location)
    output_base_path.mkdir(parents=True, exist_ok=True)
    repo_path = output_base_path / "repos"
    repo_path.mkdir(exist_ok=True)
    timeout = args.timeout
    assert timeout > 0

    workers = args.workers
    if args.max_repos is not None and workers > 0:
        print(f"WARNING: setting '--max-repos' in multiprocessing mode (--workers > 0) will lead to unreproducible results.")

    total_repos = 0
    completely_failed_repos = 0
    total_found = 0

    successes: list[Success] = []
    failures: list[Failure] = []

    def result_logger(result: PreprocessResult):
        repo = result.repo
        nonlocal total_found, total_repos, completely_failed_repos
        if isinstance(result, Success):
            successes.append(result)
            found = result.count_preprocessed_files()
            total_found += found
            total_repos += 1
            if found == 0:
                completely_failed_repos += 1
        else:
            assert isinstance(result, Failure)
            print(f"{repo} failed.")
            failures.append(result)

    archives_not_found: list[str] = []
    # repos.sort()
    random.shuffle(repos)
    if args.max_repos is not None:
        repos = repos[:args.max_repos]
    # count = 0
    try:
        if workers == 0:
            for repo in tqdm(repos):
                archive = archive_base_path / (repo + ".tar.gz")
                if not archive.exists():
                    # with tempfile.TemporaryDirectory() as td:
                    #     owner, name = repo.strip().split("/")
                    #     print(owner, name)
                    #     clone_result = ghcc.clone(
                    #         repo_owner=owner, repo_name=name, clone_folder="repo_temp",
                    #         timeout=timeout, recursive=True
                    #     )
                    #     print(clone_result)
                    #     print()
                    #     if not clone_result.success:
                    #         print(f"Skipping {archive}: not found.")
                    #         archives_not_found.append(repo)
                    #     elif owner != name:
                    #         if count == 5:
                    #             sys.exit(0)
                    #         else:
                    #             count += 1
                        
                    continue
                try:
                    result = extract_original_code(archive, repo, repo_path / repo, timeout)
                except Exception as e:
                    result = Failure(repo, "unknown", repr(e))
                result_logger(result)
        else:
            progress = tqdm(total=len(repos))
            def preproc_arg_generator(repos: list[str]) -> Iterator[tuple[Path, str, Path, int]]:
                """Prepare the arguments for extract_original_code as a tuple. Ignore and log repositories
                without archives.
                """
                for repo in repos:
                    archive = archive_base_path / (repo + ".tar.gz")
                    if not archive.exists():
                        archives_not_found.append(repo)
                        progress.update(1)
                        continue
                    yield (archive, repo, repo_path / repo, timeout)

            with multiprocessing.Pool(workers) as pool:
                iterator = pool.imap_unordered(extract_original_code_wrapper, preproc_arg_generator(repos))
                for result in iterator:
                    progress.update(1)
                    result_logger(result)
            progress.close()
    except KeyboardInterrupt:
        print("Keyboard Interrupt detected; exiting.")
    except Exception as e:
        print(f"WARNING: Terminating excution early due to exception: {repr(e)}")
        with open(output_base_path / "EARLY_TERMINATION_WARNING.txt", "w") as fp:
            fp.write(traceback.format_exc())
            fp.write("\n")

    outs = "---- Summary Results ----\n"
    outs += f"Total repos examined: {total_repos}\n"
    outs += f"Repos with no preprocessed files produced: {completely_failed_repos}\n"
    outs += f"Total new files found: {total_found}\n"
    outs += f"Total failures: {len(failures)}\n"
    outs += f"Total archives not found: {len(archives_not_found)}\n"

    print("\n")
    print(outs)

    with open(output_base_path / "summary.json", "w") as fp:
        json.dump({
            "successes": [s.to_json() for s in successes],
            "failures": [f.to_json() for f in failures]
        }, fp)
    
    with open(output_base_path / "command.txt", "w") as fp:
        fp.write(" ".join(sys.argv))
    
    with open(output_base_path / "info.txt", "w") as fp:
        fp.write(outs)

    if len(archives_not_found) > 0:
        with open(output_base_path / "archives_not_found.txt", "w") as fp:
            fp.write("\n".join(archives_not_found))
            fp.write("\n")
        

if __name__ ==  "__main__":
    main(get_args())