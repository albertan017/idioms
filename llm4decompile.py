"""Evaluate plain LLM4Decompile on exebench, which has not been finetuned or modified in any way. 
Thus, we use prompting techniques specified in the models' documentation, not the tokens
that are used elsewhere in evaluator. Should be used with:
https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.5
https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v1.5
https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v2
https://huggingface.co/LLM4Binary/llm4decompile-6.7b-v2
"""

import argparse
import re
import os
import json
import random
import subprocess
import tempfile
import shutil
import tarfile
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from idioms.data.dataset import MatchedFunction
from idioms.hf import causal_stringify_function_prompt
from evaluator import (
    FunctionEvaluator,
    ORIGINAL_EXAMPLE_ATTR,
    exebench_to_matched_function,
    calculate_executable_metrics,
    write_output_to_files
)

CLANG_FORMAT = "clang-format" # "./format/clang-format"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="The path to/huggingface ID of the model")
    parser.add_argument("eval_dataset", help="The exebench dataset to use.")
    parser.add_argument("--compiled", help="Location of the compiled binaries corresponding to the exebench dataset.")
    parser.add_argument("--dataset-split", default="valid_real", help="The partition of the validation set on which to evaluate.")
    parser.add_argument("--outdir", default="baselines", help="Where to write the results summarizing the output.")
    # parser.add_argument("--max-context-length", type=int, default=None, help="The maximum length of the pre-prediction context (the decompiled information)")
    parser.add_argument("--max-prediction-length", type=int, default=1024, help="The maximum number of new tokens to be predicted for the original function code and UDT definitions.")
    parser.add_argument("--max-decompiled-function-size", type=int, default=1024, help=f"Filter out functions with decompiled code size greater than this amount.")
    parser.add_argument("--random-seed", type=int, default=80, help=f"Used to seed python's standard random module.")
    parser.add_argument("--limit", type=int, help="Only predict this many examples.")
    return parser.parse_args()

ORIGINAL_INDEX_KEY = "original_index"

def build_assembly_prompt(meta: dict, binaries_dir: str, split_name: str) -> str | None:
    """Process an exebench example in the way the LLM4Decompile authors describe.

    Based on https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.5

    :param meta: an exebench entry
    :returns: the fully-formed prompt, or None if an error occurred.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        binary_name = f"{split_name}_{meta['index']}_{meta['fname']}"
        binary = os.path.join(binaries_dir, binary_name)
        if not os.path.exists(binary):
            return None
        
        shutil.copyfile(binary, os.path.join(tempdir, binary_name))

        # input_file = fileName+'.c'
        # A -c is required here to compile an arbitrary function without a main function, though it's not in the original script
        # compile_command = f'gcc -c -o {output_file}.o {input_file} -{opt_state} -lm'#compile the code with GCC on Linux
        # subprocess.run(compile_command, shell=True, check=True, cwd=tempdir)
        
        # Custom addition to see if the name of the function is reverted to <func0> when stripped
        # subprocess.run(["strip", f"{output_file}.o", "--strip-debug"])
        # dump_command = f'objdump -d {output_file}.o > {output_file}.s'#disassemble the binary file into assembly instructions
        
        dump_command = f'objdump -d {binary_name} > {binary_name}.s'
        subprocess.run(dump_command, shell=True, check=True, cwd=tempdir)
        
        input_asm = ''
        with open(os.path.join(tempdir, binary_name+'.s'), "r") as f:#asm file
            asm= f.read()
    function_name = meta['fname']
    if '<'+function_name+'>:' not in asm: #IMPORTANT replace func0 with the function name
        return None
    asm = '<func0>:' + asm.split('<'+function_name+'>:')[-1].split('\n\n')[0] #IMPORTANT replace func0 with the function name
    asm_clean = ""
    asm_sp = asm.split("\n")
    for tmp in asm_sp:
        if len(tmp.split("\t"))<3 and '00' in tmp:
            continue
        idx = min(
            len(tmp.split("\t")) - 1, 2
        )
        tmp_asm = "\t".join(tmp.split("\t")[idx:])  # remove the binary code
        tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
        asm_clean += tmp_asm + "\n"
    input_asm = asm_clean.strip()
    before = f"# This is the assembly code:\n"#prompt
    after = "\n# What is the source code?\n"#prompt
    input_asm_prompt = before+input_asm.strip()+after
    return input_asm_prompt

def build_decompilation_prompt(meta: dict) -> str | None:
    """Make a prompt for decompilation-input (v2) LLM4Decompile models.
    """
    # This is directly taken from the official documentation:
    # https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v2.
    # Even for ghidra decompilation, the prompt says "assembly code."
    before = f"# This is the assembly code:\n"#prompt
    after = "\n# What is the source code?\n"#prompt
    if meta['ghidra'] is None:
        return None
    with tempfile.TemporaryDirectory() as tempdir:
        c_file = os.path.join(tempdir, "temp.c")
        with open(c_file, "w") as fp:
            fp.write(meta['ghidra'].strip())
        run = subprocess.run([CLANG_FORMAT, c_file], capture_output=True)
         # clang-format writes the formatted code stdout.
        formatted = run.stdout.decode()
       
    input_asm_prompt = before + formatted + after
    return input_asm_prompt

def main(args: argparse.Namespace):
    random.seed(args.random_seed)
    model_type: str = args.model_type
    eval_dataset = Path(args.eval_dataset)
    split_name: str = args.dataset_split
    assert eval_dataset.exists(), f"Eval dataset {eval_dataset} does not exit."
    # max_context_length = args.max_context_length
    max_prediction_length = args.max_prediction_length
    max_decompiled_function_size = args.max_decompiled_function_size
    limit = args.limit
    outdir = Path(args.outdir) / model_type.split("/")[1] / eval_dataset.name
    os.makedirs(outdir, exist_ok=True)

    assert "llm4decompile" in model_type.lower(), f"This script only suppots LLM4Decompile models."

    if "v1.5" in model_type:
        use_assembly = True
    else:
        assert "v2" in model_type, model_type
        use_assembly = False

    if use_assembly:
        assert args.compiled is not None, f"Must supply location of compiled binaries with --compiled argument for v1.5 (assembly-based) models."
        compiled = Path(args.compiled) / split_name
        assert compiled.exists(), f"Compiled binaries directory {compiled} not found."
    
    omatch = re.search("O[0123]", eval_dataset.name)
    assert omatch is not None
    optimization = omatch.group(0)
    print(f"Optimization: {optimization}")

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.bfloat16).cuda()

    def function_size_filter(fn: MatchedFunction) -> bool:
        """Return True if the decompiled code fits within the allowed context size.
        """
        # Note: this actually uses the hex-rays size to be consistent with evaluator.py, (that way, the same functions are filtered out)
        # though llm4decompile in its unmodified form uses ghidra.
        return len(tokenizer.encode(causal_stringify_function_prompt(fn))) <= max_decompiled_function_size

    raw_dataset = load_from_disk(eval_dataset)
    if isinstance(raw_dataset, DatasetDict):
        raw_dataset = raw_dataset[split_name]
    
    raw_dataset = raw_dataset.add_column("index", list(range(len(raw_dataset)))) # type: ignore
    holdout_set = list(filter(function_size_filter, filter(None, map(exebench_to_matched_function, raw_dataset)))) # type: ignore
    if limit is not None:
        random.shuffle(holdout_set)
        holdout_set = holdout_set[:limit]

    # Gets rid of the following warning: 
    #   Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    unable_to_generate_prompt = 0
    ooms = 0

    results = []
    with tempfile.TemporaryDirectory() as binaries_dir:
        if use_assembly:
            for shard in compiled.iterdir():
                assert ".tar" in shard.name
                with tarfile.open(shard, "r:*") as tf:
                    tf.extractall(path=binaries_dir)

        for fn in tqdm(holdout_set, dynamic_ncols=True):
            # exebench assembly is slightly different than the assembly 
            # generated by the authors' process so we'll use the authors' process here.
            meta = getattr(fn, ORIGINAL_EXAMPLE_ATTR)
            if use_assembly:
                prompt = build_assembly_prompt(meta, binaries_dir, split_name)
            else:
                prompt = build_decompilation_prompt(meta)

            if prompt is None:
                unable_to_generate_prompt += 1
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_prediction_length)
            except torch.cuda.OutOfMemoryError:
                ooms += 1
            prediction = tokenizer.decode(outputs[0][len(inputs[0]):-1]) # type: ignore
            results.append((fn, prediction))
    
    evaluator = FunctionEvaluator()
    scores = evaluator(results)

    exebench_info = [getattr(r[0], ORIGINAL_EXAMPLE_ATTR) for r in results]
    write_output_to_files(results, outdir / f"{args.dataset_split}_results", exebench_info)
    scores |= calculate_executable_metrics(results, exebench_info)
    scores["unable_to_generate_prompt"] = unable_to_generate_prompt
    scores["out_of_memory_errors"] = ooms

    for score, value in scores.items():
        print(score, value)
    print()

    with open(outdir / f"{args.dataset_split}_scores.json", "w") as fp:
        json.dump(scores, fp, indent=2)

    
if __name__ == "__main__":
    main(get_args())
