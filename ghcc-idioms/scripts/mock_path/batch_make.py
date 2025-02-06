#!/usr/bin/env python3
#print("IN batch_make.py **********")

import functools
import multiprocessing as mp
import os
import pickle
import queue
import time
import subprocess
from typing import Dict, List, Optional

import argtyped
import flutes
from argtyped import Switch

import ghcc


class Arguments(argtyped.Arguments):
    compile_timeout: int = 900  # wait up to 15 minutes
    record_libraries: Switch = False
    gcc_override_flags: Optional[str] = None
    extract_preprocessed_files: Switch = False
    single_process: Switch = False  # useful for debugging
    verbose: Switch = False
    compiler: str # type of compiler to use, "gcc" or "g++"


args = Arguments()

TIMEOUT_TOLERANCE = 5  # allow worker process to run for maximum 5 seconds beyond timeout
REPO_PATH = "/usr/src/repo"
BINARY_PATH = "/usr/src/bin"


def compile_makefiles():
    if args.extract_preprocessed_files:
        def check_file_fn(directory: str, file: str) -> bool:
            path = os.path.join(directory, file)
            output = subprocess.check_output(["file", path], timeout=10).decode("utf8")
            print(output)
            return "C source" in output

        compile_fn = functools.partial(
            ghcc.compile._make_skeleton, make_fn=ghcc.compile._unsafe_make, check_file_fn=check_file_fn)
        makefile_dirs = ghcc.find_makefiles(REPO_PATH) # list(makefile_info.keys())
        kwargs = {"compile_fn": compile_fn}
    else:
        makefile_dirs = ghcc.find_makefiles(REPO_PATH)
        kwargs = {"compile_fn": ghcc.unsafe_make}

    for makefile in ghcc.compile_and_move(
            BINARY_PATH, REPO_PATH, makefile_dirs, compiler=args.compiler,
            compile_timeout=args.compile_timeout, record_libraries=args.record_libraries,
            gcc_override_flags=args.gcc_override_flags, **kwargs):
        makefile['directory'] = os.path.relpath(makefile['directory'], REPO_PATH)
        yield makefile


def worker(q: mp.Queue):
    for makefile in compile_makefiles():
        q.put(makefile)


def read_queue(makefiles: List, q: 'mp.Queue'):
    try:
        while True:
            makefiles.append(q.get_nowait())
    except queue.Empty:
        pass  # queue empty, wait until next round
    except (OSError, ValueError):
        pass  # data in queue could be corrupt, e.g. if worker process is terminated while enqueueing


def main():
    import sys
    print(f"\n********\n\nCall to batch_make.py:\n\n{' '.join(sys.argv)}\n\n*********")
    if args.single_process:
        makefiles = list(compile_makefiles())
    else:
        q = mp.Queue()
        process = mp.Process(target=worker, args=(q,))
        process.start()
        start_time = time.time()

        makefiles: List = []
        while process.is_alive():
            time.sleep(2)  # no rush
            cur_time = time.time()
            # Get stuff out of the queue before possible termination -- otherwise it might deadlock.
            # See https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming,
            # the "Joining processes that use queues" section.
            read_queue(makefiles, q)
            # Note that it's still possible to have deadlocks if the child process pushed new elements into the queue
            # after we read and before we terminate. A better solution would be to send a message to the child and ask
            # it to quit, and only terminate when it doesn't respond. However, this current implementation is probably
            # good enough for most cases.
            if cur_time - start_time > args.compile_timeout + TIMEOUT_TOLERANCE:
                process.terminate()
                print(f"Timeout ({args.compile_timeout}s), killed", flush=True)
                ghcc.clean(REPO_PATH)  # clean up after the worker process
                break
        read_queue(makefiles, q)

    flutes.kill_proc_tree(os.getpid(), including_parent=False)  # make sure all subprocesses are dead
    with open(os.path.join(BINARY_PATH, "log.pkl"), "wb") as f:
        pickle.dump(makefiles, f)
    flutes.run_command(["chmod", "-R", "g+w", BINARY_PATH])
    flutes.run_command(["chmod", "-R", "g+w", REPO_PATH])


if __name__ == '__main__':
    main()
