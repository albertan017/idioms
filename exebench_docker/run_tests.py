"""Run the tests for each example in exebench
"""

import argparse
import os
import sys
import json
import subprocess
import shutil

TEST_HARNESS_NAME = "prediction_harness.cpp"
IO_PAIRS_JSON = "io_pairs.json"

def load_test_case(raw):
    """Convert a single example into a format that the test harness can use.
    For some reason, a slightly different format is used in exebench itself; it must be converted to be read by the test harness.
    """
    return {var: json.loads(value) for var, value in zip(raw["var"], raw["value"])}

def get_trial_result(error = None, tests = None):
    return {"error": error, "tests": tests}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", help="Where to get the input/write the output")
    parser.add_argument("test_harness", help="C++ file for running the tests")
    parser.add_argument("prediction", help="The C file which contains the predicted code. Expected to be compilable from the same directory as the harness.")
    parser.add_argument("io_pairs", help="The json with the I/O examples for each test.")
    parser.add_argument("output", help="The name of the output json file.")
    parser.add_argument("--include-fpermissive", action="store_true", help="Get -fpermissive results as well.")
    args = parser.parse_args()

    volume = args.volume
    fpermissive: bool = args.include_fpermissive

    # Transfer over the files that we need to run the tests.
    # We don't define a name for the prediction/solution C file because it is referenced in the test harness.
    # It is the responsibility of the test harness to name the file correctly; we don't want to mess that up here. 
    shutil.move(os.path.join(volume, args.test_harness), TEST_HARNESS_NAME)
    shutil.move(os.path.join(volume, args.prediction), ".")
    shutil.move(os.path.join(volume, args.io_pairs), IO_PAIRS_JSON)

    # Read test cases 
    with open(IO_PAIRS_JSON, "r") as fp:
        io_pairs = json.load(fp)
    inputs = io_pairs['input']
    outputs = io_pairs['output']
    assert len(inputs) == len(outputs), f"Different numbers of inputs and outputs: {len(inputs)} and {len(outputs)}"

    results = {}
    def run_trial(name, harness, flags = []):
        """Run a trial. First compile, and error out if the program did not compile.
        Otherwise, run each test and report the result: whether the program passed or failed
        as a list of boolean.
        """
        if os.path.exists(name):
            os.unlink(name)
        print(f"---- {name}", file=sys.stderr)
        # First, compile the test harness
        cmd = ["g++", harness, "-o", name] + flags
        print(" ".join(cmd), file=sys.stderr)
        subprocess.run(cmd)
        if not os.path.exists(name):
            results[name] = get_trial_result(error="compilation")
            return

        passed = [] # list of boolean for whether or not the test passed.
        # Each example represents a test case.
        for input, output in zip(inputs, outputs):
            with open("in.json", "w") as fp:
                json.dump(load_test_case(input), fp)

            subprocess.run([f"./{name}", "in.json", "out.json"])
            
            if not os.path.exists("out.json"):
                passed.append(False) # indicates runtime error
                continue
            
            try:
                with open("out.json", "r") as fp:
                    predicted_output = json.load(fp)
            except json.decoder.JSONDecodeError:
                # There was a runtime error when the test harness was generating the JSON,
                # leading to an improperly formatted JSON.
                passed.append(False)
                continue

            correct_output = load_test_case(output)
            passed.append(predicted_output == correct_output)
        
        results[name] = get_trial_result(tests=passed)
    
    run_trial("standard", TEST_HARNESS_NAME)
    if fpermissive:
        run_trial("permissive", TEST_HARNESS_NAME, ["-fpermissive"])

    outfile = os.path.join(volume, args.output)   
    with open(outfile, "w") as fp:
        json.dump(results, fp)

if __name__ == "__main__":
    main()