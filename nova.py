"""Evaluate plain NOVA on exebench, which has not been finetuned or modified in any way. 
Thus, we use prompting techniques specified in the models' documentation, not the method
that are used elsewhere in evaluator. Should be used with:
 - https://huggingface.co/lt-asset/nova-1.3b-bcr
 - https://huggingface.co/lt-asset/nova-6.7b-bcr

The modeling_nova script is obtainable from the "files and versions" tab at the huggingface links above.
"""

import argparse
import re
import os
import json
import random
import tempfile
import tarfile
import shutil
import itertools
import subprocess
from pathlib import Path

import torch
from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from modeling_nova import NovaTokenizer, NovaForCausalLM

from idioms.dataiter import MatchedFunctionDataset, MatchedBinaryDataset
from idioms.data.dataset import MatchedFunction
from evaluator import (
    FunctionEvaluator,
    ORIGINAL_EXAMPLE_ATTR,
    exebench_to_matched_function,
    calculate_executable_metrics,
    write_output_to_files,
    read_predictions,
    read_exebench_info
)

# Nova actually just parses call instructions as if they're in an object file,
# so no relocation information (to simulate linking) is required. (This is how
# examples in the provided version of the humaneval-decompile dataset are 
# processed, for instance.)
RELOCATION = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="The path to/huggingface ID of the model")
    parser.add_argument("eval_dataset", help="The evaluation dataset to use.")
    parser.add_argument("compiled", help="Location of the compiled binaries corresponding to the exebench dataset.")
    parser.add_argument("--dataset-split", default=None, help="The partition of the validation set on which to evaluate. Defaults to 'test_real' for exebench-format datasets and 'test' for idioms-format datasets.")
    parser.add_argument("--outdir", default="baselines", help="Where to write the results summarizing the output.")
    parser.add_argument("--max-prediction-length", type=int, default=1024, help="The maximum number of new tokens to be predicted for the original function code and UDT definitions.")
    parser.add_argument("--random-seed", type=int, default=80, help=f"Used to seed python's standard random module.")
    parser.add_argument("--limit", type=int, help="Only predict this many examples.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    parser.add_argument("--use-existing-predictions", action="store_true", help="Instead of running prediction itself, use an existing set of predictions.")
    parser.add_argument("--no-exebench-tests", action="store_true", help="Don't run exebench tests.")
    return parser.parse_args()

def stringify_function_target(fn: MatchedFunction):
    return fn.canonical_original_code

def normalize_instruction(instruction: str) -> str:
    """Convert an individual instruction into the form expected by nova.
    Precondition: the argument includes only the instruction (e.g. "movb   $0x72,(%rax)"), 
    not any line numbers/offsets or instruction hex values
    """
    for hexval in re.finditer(r"0x[\da-f]+", instruction):
        instruction = instruction.replace(hexval.group(0), str(int(hexval.group(0), base=16)))
    instruction = instruction.replace(",", " , ").replace("(", " ( ").replace(")", " ) ")
    instruction = re.sub(r" +", " ", instruction) # collapse multiple spaces into one.
    instruction = instruction.replace("%", "")
    return instruction

def get_raw_objdump_output(binary_name: str, binaries_dir: str, tempdir: str, relocation: bool) -> str:
    """Run objdump's disassembly function and return the output. Include reloaction information.
    """
    binary = os.path.join(binaries_dir, binary_name)
    assert os.path.exists(binary)
    
    shutil.copyfile(binary, os.path.join(tempdir, binary_name))

    flags = "-d"
    if relocation:
        flags += "r"

    # Get the assembly with the debug info.
    run_result = subprocess.run(['objdump', flags, binary_name], check=True, cwd=tempdir, capture_output=True)
    return run_result.stdout.decode('utf-8').strip()

# TEXT_SECTIONS = {".text", ".text.hot", ".text.unlikely"} # .text.hot and .text.unlikely are from -freorder-functions
def get_text_sections(raw_debug_asm: str, binary_name: str):
    """Takes in the output of objdump -dr and returns a list of the lines
    in the text sections, as well as the relocation information for functions
    in the text sections.
    """
    # Sanity-check the start of of the objdump output
    lines = raw_debug_asm.splitlines()
    assert lines[0].startswith(binary_name)
    assert len(lines[1]) == 0
    assert len(lines[2]) == 0
    del lines # throw away old lines, we don't need them anymore. From here, we're just interested in the .text section(s)

    # Extract the text sections
    sections = raw_debug_asm.split("Disassembly of section ")
    text_sections = [sec for sec in sections if sec.startswith(".text")]
    text_lines: list[str] = [] 
    for section in text_sections:
        section_lines = section.splitlines()
        # The -ffunction-sections flag on some compilers makes a separate section for each function named .text.function_name
        # assert (secname := section_lines[0][:-1]) in TEXT_SECTIONS, f"{secname} is not a recognized text section."
        assert len(section_lines[1]) == 0
        text_lines.extend(section_lines[2:])
    assert len(text_lines) > 0, f"Could not find text section for binary {binary_name}."
    return text_lines

def extract_relocation_info(lines: list[str]) -> tuple[list[str], dict[int, str]]:
    """If objdump was run with the -r argument, relocation information will be interspersed with the assembly.
    Remove that information and return the relocation information for functions, which is indexed by the address
    at which that relocation information is used.

    :param lines: the lines of the objdump output from the .text section(s)
    :returns: a tuple of two values:
     -  The lines of the .text sections, excluding lines listing relocation information
     -  The relocation information for functions, indexed by offset.
    """
    outlines = []
    relocation: dict[int, str] = {}
    for line in lines:
        if line.startswith("\t\t\t"): # is a relocation entry
            loc_type, value = line[3:].split("\t")
            offset, symbol_type = loc_type.split(": ")
            assert offset is not None, f"Failed to find relocation offset in line {line}"
            if "PLT" in symbol_type.upper():
                # This is very platform-dependent so we defensively do the if-assert to cause a failure if platform-specific assumptions
                # are not met. Not perfect, but okay.
                assert symbol_type == "R_X86_64_PLT32", f"Unknown PLT-associated symbol type: {symbol_type}"
                function_name = value.split("-")[0]
                assert re.match(r"^\w+$", function_name) is not None, f""
                relocation[int(offset, base=16)] = function_name
            # Ignore other symbol types for now (e.g. string literals) because there's little that we can do about them. 
        else:
            outlines.append(line)

    return outlines, relocation

def normalize_asm(fn: MatchedFunction, binaries_dir: str, split_name: str, dataset_is_exebench: bool, fnnames_in_binary: list[str]) -> str | None:
    """Process an exebench example in the way the Nova authors described.

    :param fn: a MatchedFunction. Expected to have exebench metadata attached if dataset_is_exebench==True
    :binaries_dir: where the compiled binaries for this dataset are stored.
    :split_name: the partition of the training set
    :returns: the normalized assembly.
    """

    fn_name = fn.name
    with tempfile.TemporaryDirectory() as tempdir:
        if dataset_is_exebench:
            meta = getattr(fn, ORIGINAL_EXAMPLE_ATTR)
            assert fn_name == meta['fname'], f"Inconsistent function names: {fn_name} (MatchedFunction) vs {meta['fname']} (exebench metadata)"

            # Exebench binaries exist in isolation, so we build them each with their own object file, with one function per file.
            # They use the naming convention "split-name_dataset-index_function-name", which is different from how idioms datasets
            # handle things: by hashing each binary and using that as its name.
            binary_name = f"{split_name}_{meta['index']}_{fn_name}"
            raw_debug_asm = get_raw_objdump_output(binary_name, binaries_dir, tempdir, RELOCATION)
            
            # With the way we built the exebench dataset's binaries, there's exactly one function per binary so we can 
            # assume that the one function present is the target function. Confirm that this is the case.
            function_names = re.findall(r"<(\w+)>", raw_debug_asm)
            # The call to set() here reflacts the fact that the function might be recursive, referencing
            # itself in a callq instruction
            assert len(set(function_names)) == 1 and function_names[0] == fn_name, function_names

            lines = get_text_sections(raw_debug_asm, binary_name)
            if RELOCATION:
                lines, _ = extract_relocation_info(lines)
            assert lines[0].strip() == f"0000000000000000 <{fn_name}>:"
            lines = lines[1:]
        else:
            raw_debug_asm = get_raw_objdump_output(fn.binary_hash, binaries_dir, tempdir, RELOCATION)

            lines = get_text_sections(raw_debug_asm, fn.binary_hash)
            if RELOCATION:
                lines, _ = extract_relocation_info(lines)

            # Map start line to function name
            start_line: int = -1
            end_line: int = -1
            for i, line in enumerate(lines):
                m = re.match(fr"""[0-9a-f]+ <(\w+)>:""", line)
                if m is not None and m.group(1) == fn_name:
                    start_line = end_line = i
                    # search for the end of the function
                    while end_line < len(lines) and len(lines[end_line]) > 0:
                        end_line += 1
                    break

            if start_line == -1:
                return None
            lines = lines[start_line + 1:end_line] # use state_line + 1 to get ride of the '00000000000001e4 <fn_name>:' header

            # check that the lines we found are all instruction lines that the downstream code expects.
            for line in lines:
                assert re.match(r"""\s*[0-9a-f]+:""", line) is not None, line

    instructions: list[tuple[str, str]] = [] # tuple of instruction, label
    labels: dict[str, str] = {} # maps offset labels to nova's normalized labels, eg 3a: <label-24>
    # Break the raw objdump output into more useful pieces.
    for i, line in enumerate(lines, start=1):
        columns = line.strip().split("\t")
        offset_text = columns[0]
        assert offset_text[-1] == ":", columns
        offset = offset_text[:-1]

        if len(columns) == 2:
            instruction_text = columns[1]
        elif len(columns) == 3:
            instruction_text = columns[2]
        else:
            raise ValueError(f"{line} has the wrong number of columns: {len(line)}")

        if "#" in instruction_text:
            instruction_text = re.sub("#.*", "", instruction_text).strip()

        label = f"<label-{i}>"
        labels[offset] = label
        instructions.append((instruction_text.strip(), label))
    
    function_canonical_names: dict[str, str] = {fn_name: "<func0>"}
    for binary_function_name in fnnames_in_binary:
        if binary_function_name not in function_canonical_names: # don't want to re-name fn_name, which should always be func0 by nova's convention.
            function_canonical_names[binary_function_name] = f"<func{len(function_canonical_names)}>"

    # Make the normalized assembly
    normalized: list[str] = ["<func0>:"] # all functions are called func0
    for instruction, label in instructions:
        # Group 0: all call/jump arguments/comments
        # Group 1: concrete target address
        # Group 2: the name of the function being called or for which the subsequent offset is relative to
        # Group 3: compiler modifications to the function name, currently reflecting the -fipa-sra and -fipa-cp flag.
        # Group 4: the differentiating number the compiler uses to rename modified (optimized) versions of the same function.
        # Group 5: additional information about the location of the call/jump target
        # Group 6: the relative jump offset with prefix
        # Group 7: the relative jump offset
        # Group 8: whether or not the target function is a PLT stub.
        destinations = list(re.finditer(fr"""\s*([\da-f]+) <(\.?\w+)(\.\w+(\.\d+)?)*((\+0x([\da-f]+))|(@plt))?>""", instruction))
        assert instruction.count("<") == len(destinations), f"Function {fn_name}: There may be an offset not handled in instruction:\n{instruction}"
        for destination in destinations:
            if destination.group(6) is not None: # if there is an (implicilty-nonzero) relative offset to the function
                if destination.group(2) != fn_name and destination.group(2)[:3] != '.LC': # if that relative offset is NOT within the current function, which represents a jump into the middle of another function
                    print(f"WARNING: Call to an offset within a different function: {instruction} called from function {fn_name}.")
                target_offset = destination.group(1) # The concrete target offset of this jump
                assert not dataset_is_exebench or target_offset == destination.group(7), f"Nonequivalent offsets in {instruction}"
                if target_offset in labels: # destination.group(2) is the name of the function in the offset.
                    assert destination.group(2) == fn_name, f"Based on debug information, the specified function is outside of the current function {fn_name}: {instruction}"
                    target = labels[target_offset]
                else:
                    target = "<unk>"
            else:
                # Recursive call, or call to external function.
                # Doesn't occur in humaneval-decompile, so it's not clear how this is normalized. This is a reasonable guess.
                tgt_fn_name = destination.group(2)
                if tgt_fn_name in function_canonical_names:
                    target = function_canonical_names[tgt_fn_name]
                elif tgt_fn_name[0] == '.':
                    assert tgt_fn_name[1:3] == "LC", f"Unexpected target function symbol: {tgt_fn_name}"
                    target = "<unk>" # This comes up in the provided examples and this is how it's handled.
                else: # to be consistent with idioms, only canonicalize the names of functions that occur in the binary.
                    target = f"<{tgt_fn_name}>"
            # For some reason there's a tab in the label so we account for that.
            instruction = instruction.replace(destination.group(0), f"\t{target}")
        # NOTE: nova normalization leaves bytes not associated with an instruction unchanged and intersperses them with instructions.
        instruction = normalize_instruction(instruction)
        normalized.append(instruction + "\t" + label)

    return "\n".join(normalized)

def main(args: argparse.Namespace):
    random.seed(args.random_seed)
    model_type: str = args.model_type
    eval_dataset = Path(args.eval_dataset)
    assert eval_dataset.exists(), f"Eval dataset {eval_dataset} does not exit."
    # max_context_length = args.max_context_length
    max_prediction_length = args.max_prediction_length
    limit = args.limit
    greedy: bool = args.greedy
    do_prediction: bool = not args.use_existing_predictions
    do_exebench_tests: bool = not args.no_exebench_tests

    outdir = Path(args.outdir) / model_type.split("/")[1] / eval_dataset.name / ("greedy" if greedy else "sampling")
    if do_prediction:
        os.makedirs(outdir, exist_ok=True)
    else:
        assert outdir.exists(), f"Prediction dir {outdir} does not exist."

    assert "nova" in model_type.lower(), f"This script only suppots NOVA models."
    
    omatch = re.search("O[0123]", eval_dataset.name)
    assert omatch is not None
    optimization = omatch.group(0)
    print(f"Optimization: {optimization}")

    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('Vocabulary:', len(tokenizer.get_vocab()))    # 32280
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    nova_tokenizer = NovaTokenizer(tokenizer)

    def function_size_filter(fn: MatchedFunction) -> bool:
        """Return True if the decompiled code and original code (together with UDT definitions) fit
        fit within the maximum context length.
        """
        return len(tokenizer.encode(stringify_function_target(fn))) <= max_prediction_length

    namesByBinary: dict[str, list[str]] = {} # used to determine what names to canonicalize in linking
    compiled = Path(args.compiled)
    dataset_is_exebench = (eval_dataset / "dataset_info.json").exists() or (eval_dataset / "dataset_dict.json").exists()
    if args.dataset_split is None:
        split_name = "test_real" if dataset_is_exebench else "test"
    else:
        split_name = args.dataset_split

    if do_prediction:
        model = NovaForCausalLM.from_pretrained(model_type, torch_dtype=torch.bfloat16).eval()
        model.to('cuda') # type: ignore

        ### Load the dataset
        if dataset_is_exebench:
            raw_dataset = load_from_disk(eval_dataset)
            if isinstance(raw_dataset, DatasetDict):
                raw_dataset = raw_dataset[split_name]
            
            raw_dataset = raw_dataset.add_column("index", list(range(len(raw_dataset)))) # type: ignore
            holdout_set = list(filter(function_size_filter, filter(None, map(exebench_to_matched_function, raw_dataset)))) # type: ignore

            compiled: Path = compiled / split_name
            assert compiled.exists(), f"Compiled binaries directory {compiled} not found."

            # Exebench stores a single function independent of any given binary so by construction there's
            # just one function in each "binary." 
            for fn in holdout_set:
                namesByBinary[fn.binary_hash] = [fn.name]
        else:
            holdout_set = list(filter(function_size_filter, MatchedFunctionDataset(eval_dataset.glob(f"{split_name}*.tar"), shuffle=False)))
            if compiled.is_dir():
                compiled = compiled / f"{eval_dataset.name}_{split_name}_binaries.tar.gz"
            assert compiled.exists(), f"Compiled binaries archive {compiled} not found."

            for binary in MatchedBinaryDataset(eval_dataset.glob(f"{split_name}*.tar")):
                namesByBinary[binary.binary_hash] = list(itertools.chain((fn.name for fn in binary.functions), binary.unmatched))
        if limit is not None:
            random.shuffle(holdout_set)
            holdout_set = holdout_set[:limit]
    
        ### Run prediction
        ooms = 0
        failed_to_extract_assembly = 0
        results = []
        with tempfile.TemporaryDirectory() as binaries_dir:
            # Extract the binaries from which the real assembly will be sourced.
            if compiled.is_dir():
                shards = compiled.iterdir()
            else:
                shards = [compiled]
            for shard in shards:
                assert ".tar" in shard.name
                with tarfile.open(shard, "r:*") as tf:
                    tf.extractall(path=binaries_dir)
            for fn in tqdm(holdout_set, dynamic_ncols=True):
                # Prediction code closely follows the authors' example at https://huggingface.co/lt-asset/nova-6.7b-bcr
                prompt_before = f'# This is the assembly code with {optimization} optimization:\n<func0>:'
                asm = normalize_asm(fn, binaries_dir, split_name, dataset_is_exebench, namesByBinary[fn.binary_hash])
                prompt_after = '\nWhat is the source code?\n'

                if asm is None:
                    failed_to_extract_assembly += 1
                    continue

                inputs = prompt_before + asm + prompt_after
                char_types = '0' * len(prompt_before) + '1' * len(asm) + '0' * len(prompt_after)
                
                tokenizer_output = nova_tokenizer.encode(inputs, '', char_types)
                input_ids = torch.LongTensor(tokenizer_output['input_ids'].tolist()).unsqueeze(0)
                nova_attention_mask = torch.LongTensor(tokenizer_output['nova_attention_mask']).unsqueeze(0)

                try:
                    if greedy: # greedy defaults to false; the authors use sampling in their documentation.
                        outputs = model.generate(
                            inputs=input_ids.cuda(), max_new_tokens=max_prediction_length, do_sample=False, nova_attention_mask=nova_attention_mask.cuda(),
                            no_mask_idx=torch.LongTensor([tokenizer_output['no_mask_idx']]).cuda(), 
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
                        )
                    else:
                        outputs = model.generate(
                            inputs=input_ids.cuda(), max_new_tokens=max_prediction_length, temperature=0.2, top_p=0.95,
                            num_return_sequences=1, do_sample=True, nova_attention_mask=nova_attention_mask.cuda(),
                            no_mask_idx=torch.LongTensor([tokenizer_output['no_mask_idx']]).cuda(), 
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
                        )

                    for output in outputs: # Could be done with a squeeze but this way is more extensible if we want to handle top-k predictions in the future.
                        prediction = tokenizer.decode(output[input_ids.size(1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        break # only get the top-1 prediction

                    results.append((fn, prediction))
                except torch.cuda.OutOfMemoryError:
                    ooms += 1
                    results.append((fn, ""))
        # Save the predictions as soon as we finish with them.
        if dataset_is_exebench:
            exebench_info = [getattr(r[0], ORIGINAL_EXAMPLE_ATTR) for r in results]
        else:
            exebench_info = None
        write_output_to_files(results, outdir / f"{split_name}_results", exebench_info)
    else:
        results = read_predictions(outdir / f"{split_name}_results.json")
        if dataset_is_exebench:
            exebench_info = read_exebench_info(outdir / f"{split_name}_results_exebench_info.json")
        else:
            exebench_info = None
        with open(outdir / f"{split_name}_scores.json", "r") as fp:
            scores = json.load(fp)
        ooms = scores["out_of_memory_errors"]
        failed_to_extract_assembly = scores["failed_to_extract_assembly"]
        del scores
    
    evaluator = FunctionEvaluator()
    scores = evaluator(results)

    if dataset_is_exebench and do_exebench_tests:
        scores |= calculate_executable_metrics(results, exebench_info) # type: ignore
    scores["out_of_memory_errors"] = ooms
    scores["failed_to_extract_assembly"] = failed_to_extract_assembly

    for score, value in scores.items():
        print(score, value)
    print()

    with open(outdir / f"{split_name}_scores.json", "w") as fp:
        json.dump(scores, fp, indent=2)

    
if __name__ == "__main__":
    main(get_args())
