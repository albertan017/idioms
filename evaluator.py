"""Perform a detailed evaluation of idioms.
"""

import warnings
import argparse
import functools
import random
import json
import sys
import re
import tempfile
import subprocess
import itertools
from collections import Counter
from pathlib import Path
from typing import Iterable, Callable, TypeVar, Any, Container

import torch
from torch.utils.data import DataLoader
import tree_sitter_c
from tree_sitter import Node, Parser, Language
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from datasets import load_from_disk, DatasetDict
from numpy.typing import NDArray
from tqdm import tqdm
from peft import PeftModel # type: ignore # mypy thinks that PeftModel is a private class.
from pygments.lexers.c_cpp import CLexer
from pygments.token import Whitespace
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

from prepare import (
    Scope, 
    FileTypeMapping,
    PreprocessedFunction,
    get_all_user_defined_types,
    has_error, 
    get_child,
    TypeNotDefinedError, 
    TypeNotFoundError, 
    UnsupportedFeatureError,
    TypeNotFoundError
)
from idioms.data.dataset import MatchedFunction, MatchedBinary
from idioms.data.types import *
from idioms.dataiter import MatchedBinaryDataset, MatchedBinaryFunctionWrapper
from idioms.hf import (
    causal_stringify_binary_prompt,
    causal_stringify_neighbors_prompt,
    causal_stringify_function_prompt,
    DECOMPILED_ORIG_SEP
)
from codealign import align, Alignment
from codealign.ir import Variable, Parameter, GlobalVariable
from codealign.lang.c import ParsingError, SemanticError

ADAPTER_NAME="decomp_fn_rewrite"
ORIGINAL_EXAMPLE_ATTR = "raw_exebench_example"

C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)

T = TypeVar("T")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="The directory produced by training a model, or a huggingface model ID")
    parser.add_argument("--dataset", type=str, help="The path to the evaluation dataset. If not specified, will use the dataset specified for training.")
    parser.add_argument("--evaluate-existing-predictions", action="store_true", help="load predictions from the predictions JSON file instead of recalculating them.")
    parser.add_argument("--eval-partition", choices=["test", "validation"], default="validation", help="The dataset partition to use.")
    parser.add_argument("--exebench-subpartition", choices=["real", "synth"], default="real", help="Which partition of the exebench evaluation sets to use: synth or real.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size during prediction.")
    parser.add_argument("--max-context-length", type=int, default=None, help="The maximum length of the pre-prediction context (the decompiled information, including neighboring functions in neighbors mode.)")
    parser.add_argument("--max-decompiled-function-size", type=int, default=1024, help="Filter out any any functions with more than this many decompiled tokens.")
    parser.add_argument("--max-prediction-length", type=int, default=1024, help="The maximum number of new tokens to be predicted for the original function code and UDT definitions.")
    parser.add_argument("--random-seed", type=int, default=80, help=f"Used to seed python's standard random module.")
    parser.add_argument("--limit", type=int, help="Only predict this many examples instead of all of them.")
    parser.add_argument("--no-exebench-tests", action="store_true", help="Don't execute exebench tests")
    parser.add_argument("--missing-predictions-only", action="store_true", help="Only generate predictions for examples that aren't in the original predictions file.")
    return parser.parse_args()


class FunctionEvaluator:
    def __init__(self, write_output_to: Path | None = None):
        self.metric_names: list[str] = [
            "bleu",
            "syntatically_valid",
            "semantically_valid",
            "perfectly_aligned",
            "perfectly_aligned_and_typechecks",
            "variable_name_accuracy",
            "variable_type_accuracy",
            "variable_udt_exact_matches",
            "variable_udt_composition_matches",
            "variables_inherently_alignable",
            "oracle_has_nonexistent_field",
            "codealign_failures"
        ]
        self.write_output_to = write_output_to
        self.lexer = CLexer()

    def bleu(self, f: MatchedFunction, prediction: str) -> float:
        original_tokens: list[str] = [t[1] for t in self.lexer.get_tokens(f.canonical_original_code) if t[0] is not Whitespace]
        predicted_tokens: list[str] = [t[1] for t in self.lexer.get_tokens(prediction) if t[0] is not Whitespace]
        with warnings.catch_warnings(action="ignore"): # there's a warning when there's no overlaps of a certain n-gram type, which is normal when predictions are poor early in training.
            return sentence_bleu([original_tokens], predicted_tokens) # type: ignore (sentence_bleu's type hints are wrong.)

    def __iter__(self) -> Iterable[str]:
        return iter(self.metric_names)

    def __call__(self, predictions: Iterable[tuple[MatchedFunction, str]]) -> dict[str, float]:
        if self.write_output_to is not None and not isinstance(predictions, list):
            self.predictions = list(predictions)
        errors_during_evaluation = 0
        metric_values = {m: list() for m in self.metric_names}
        for ground_truth, prediction in predictions:
            metric_values["bleu"].append(self.bleu(ground_truth, prediction))
     
            try:
                for metric, value in self.get_analysis_metrics(ground_truth, prediction).items():
                    metric_values[metric].append(value)
            except:
                errors_during_evaluation += 1
                
        metrics: dict[str, float] = {}
        for metric, values in metric_values.items():
            if metric.startswith("variable"):
                successful = total = 0
                for subsuccessful, subtotal in values:
                    successful += subsuccessful
                    total += subtotal
                metrics[metric] = float(successful / total) if total > 0 else 0.0
            else:
                metrics[metric] = float(sum(values) / len(values)) if len(values) > 0 else 0.0

        if self.write_output_to is not None:
            try:
                write_output_to_files(predictions, self.write_output_to) # type: ignore (doesn't handle the conversion to a list in the if above well.)
            except FileNotFoundError:
                print(f"Evaluation log file {self.write_output_to} not found!", file=sys.stderr)
        
        metrics["errors_during_evaluation"] = float(errors_during_evaluation)
        return metrics

    def __contains__(self, metric: str) -> bool:
        return metric in self.metric_names
    
    def get_analysis_metrics(self, fn: MatchedFunction, prediction: str) -> dict[str, Any]:
        """Compute metrics that require program analysis.
        """
        ### Compute baselines for adjustment.
        try:
            original_code = canonicalize_udt_field_names(fn.canonical_original_code, fn.variable_types, fn.user_defined_types)
        except NonexistentFieldError:
            return {"oracle_has_nonexistent_field": 1}

        try:
            self_alignment: Alignment = align(fn.canonical_original_code, fn.canonical_original_code, 'c')
        except (ParsingError, SemanticError, AssertionError, KeyError, NotImplementedError):
            # This is a failure of codealign; we don't want to penalize or reward the model for it so we exclude it from the evaluation.
            return {}
        alignable = get_aligned_variables(self_alignment)
        assert all(k in v for k, v in alignable.items()) # Sanity check, can delete later.

        alignable_variables = len(alignable)
        alignable_udt_variables = sum(name in alignable and has_udt(typ) for name, typ in fn.variable_types.items())
        variables_inherently_alignable = (len(alignable), len(fn.variable_types))

        ### Set up default metric results. These will be overriden as values are computed.
        metrics = {
            "syntatically_valid": 0,
            "semantically_valid": 0,
            "perfectly_aligned": 0,
            "perfectly_aligned_and_typechecks": 0,
            "variable_name_accuracy": (0, alignable_variables),
            "variable_type_accuracy": (0, alignable_variables),
            "variable_udt_exact_matches": (0, alignable_udt_variables),
            "variable_udt_composition_matches": (0, alignable_udt_variables),
            "variables_inherently_alignable": variables_inherently_alignable,
            "oracle_has_nonexistent_field": 0,
            "codealign_failures": 0
        }

        ### Parse and sort nodes
        root = parser.parse(bytes(prediction, "utf8")).root_node
        if root.type == "ERROR":
            return metrics
        assert root.type == "translation_unit"
        fn_nodes: list[Node] = []
        udt_nodes: list[Node] = []
        other_nodes: list[Node] = []
        for node in root.children:
            if node.type == "function_definition":
                fn_nodes.append(node)
            elif node.type in {"struct_specifier", "union_specifier", "enum_specifier"}:
                udt_nodes.append(node)
            elif node.type == ";":
                pass
            else:
                other_nodes.append(node)

        # Process all UDTs
        types = FileTypeMapping()
        for type_node in udt_nodes:
            if not has_error(type_node):
                try:
                    types.parse_type(type_node)
                except:
                    pass # Ignore the misgenerated type. If it's necessary, we'll get another exception later.

        if len(fn_nodes) == 0:
            return metrics  
        fn_node = fn_nodes[0]
        assert fn_node.text is not None # To make mypy happy

        # We don't track or measure the accuracy of types not associated with variables (though this is taken into account when computing an alignment).
        # However PreprocessedFunction will throw an exception if it encounters an unrecognized type in a typecast.
        # To prevent this, we add generic placholder types.
        predicted_body = fn_node.child_by_field_name("body")
        assert predicted_body is not None
        try: # Add placholders in a try-except in case a non-variable type is unparsable.
            add_placeholders_for_nonvariable_types(predicted_body, types)
            predicted_fn = PreprocessedFunction(fn_node, types)
        except (TypeNotFoundError, TypeNotDefinedError, UnsupportedFeatureError, AssertionError):
            # We need the variable-type mapping that PreprocessedFunction provides to standardize the field names
            # for the alignment. We don't return here, however, because PreprocessedFunction is very strict about
            # types; a function may be syntatically, or arguably, semantically valid but fail here. Thus, we
            # assign predicted_fn to None here and then do the early return later.
            predicted_fn = None

        predicted_code = fn_node.text.decode()
        if predicted_fn is not None:
            predicted_udts = get_all_user_defined_types(predicted_fn)
            try:
                predicted_code = canonicalize_udt_field_names(predicted_code, predicted_fn.variable_types, predicted_udts)
            except NonexistentFieldError:
                pass # Do nothing; will fail to perfectly align and then the relevant types will be counted as incorrect later.
    
        try:
            alignment: Alignment = align(predicted_code, original_code, 'c')
        except ParsingError:
            return metrics
        except SemanticError:
            metrics["syntatically_valid"] = 1
            return metrics
        except (AssertionError, AttributeError, KeyError, NotImplementedError):
            # This is a failure of codealign; we don't want to penalize or reward the model for it so we exclude it from the evaluation.
            return {"codealign_failures": 1, "variables_inherently_alignable": variables_inherently_alignable}
        
        is_perfectly_aligned = perfectly_aligned(alignment)
        metrics["perfectly_aligned"] = is_perfectly_aligned
        metrics["syntatically_valid"] = 1
        metrics["semantically_valid"] = 1

        if predicted_fn is None:    
            return metrics
        
        # Standardize types
        ground_truth_types = FileTypeMapping()
        for udt in fn.user_defined_types:
            ground_truth_types.add_type(str(udt.stub), udt)
        fn.variable_types = expand_all(fn.variable_types, ground_truth_types)
        predicted_fn.variable_types = expand_all(predicted_fn.variable_types, types)
        
        var_map = get_aligned_variables(alignment)

        def get_predicted_var_info(ground_truth_name: str) -> list[tuple[str, TypeInfo]] | None:
            if ground_truth_name not in var_map:
                return None
            predicted_names = var_map[ground_truth_name]
            if any(name not in predicted_fn.variable_types for name in predicted_names):
                # This shouldn't happen, but it isn't serious enough that we'll want to crash everything with an assertion.
                warnings.warn(f"Property violation: {predicted_names} is in the alignment but not the preprocessed function.")
                return None
            return [(predicted_name, predicted_fn.variable_types[predicted_name]) for predicted_name in predicted_names]

        variable_name_matches = 0
        variable_type_exact_matches = 0
        variable_udt_exact_matches = 0
        variable_udt_composition_matches = 0
        typechecks: bool = True # a function with no variables trivially typechecks. (We ignore return type here.)

        for ground_truth_name, ground_truth_type in fn.variable_types.items():
            if ground_truth_name not in alignable:
                continue
            predicted_var_info = get_predicted_var_info(ground_truth_name)
            if predicted_var_info is None:
                continue # add zero to the totals of each variable-name level metrics.
            ground_truth_has_udt = has_udt(ground_truth_type)

            # shape: (number of applicable metrics, number of predictions)
            variable_results = np.zeros((2 + 2 * ground_truth_has_udt, len(predicted_var_info)))
            for i, (predicted_name, predicted_type) in enumerate(predicted_var_info):
                variable_results[0][i] = ground_truth_name == predicted_name
                variable_results[1][i] = ground_truth_type == predicted_type
                if ground_truth_has_udt:
                    variable_results[2][i] = ground_truth_type == predicted_type
                    variable_results[3][i] = type(ground_truth_type) == type(predicted_type) and has_identical_composition(predicted_type, ground_truth_type) # the type(...) == type(...) is to improve efficiency and is not strictly necessary.

            # When len(predicted_var_info) > 1, this ground-truth variable aligns with multiple variables in prediction.
            # This just mean the variables have equivalent values, which can happen even in two identical functions.
            # We choose the one that maximizes the number of scoring metrics (as must be done even when two functions are 
            # identical to get a perfect score.)

            scores: NDArray = variable_results.sum(axis=0)
            assert scores.dtype == variable_results.dtype # Sanity check; can delete later.
            argmax_idx = scores.argmax(axis=0) # axis=0 because sum() reduces the axis.
            max_score = scores[argmax_idx]
            if (max_score == scores).sum() > 1: # There is a tie in the scores. We break it by prioritizing names over types.
                # First, zero-out all rows that aren't involved in the tie.
                variable_results[np.repeat((max_score!=scores)[:,np.newaxis], variable_results.shape[0], axis=1).transpose()] = 0
                variable_results[0] *= 2 # Upweight variable name scores, the first row.
                argmax_idx = variable_results.sum(axis=0).argmax(axis=0) # recalculate the (arg)maximum
                variable_results = (variable_results > 0).astype(scores.dtype) # convert everything back to 0s and 1s.

            best_results = variable_results[:,argmax_idx]
            variable_name_matches += best_results[0]
            variable_type_exact_matches += best_results[1]
            if ground_truth_has_udt:
                variable_udt_exact_matches += best_results[2]
                variable_udt_composition_matches += best_results[3]
            
            typechecks = typechecks and bool(best_results[3 if ground_truth_has_udt else 1])

        if alignable_variables > 0:
            metrics |= {
                "variable_name_accuracy": (variable_name_matches, alignable_variables),
                "variable_type_accuracy": (variable_type_exact_matches, alignable_variables),
                "perfectly_aligned_and_typechecks": is_perfectly_aligned and typechecks
            }
        
        if alignable_udt_variables > 0:
            metrics |= {
                "variable_udt_exact_matches": (variable_udt_exact_matches, alignable_udt_variables),
                "variable_udt_composition_matches": (variable_udt_composition_matches, alignable_udt_variables)
            }
        return metrics

### Utility functions for working with TypeInfo objects
def expand_all(variable_types: dict[str, TypeInfo], types: FileTypeMapping) -> dict[str, TypeInfo]:
    scope = Scope(mapping=types)
    return {
        name: scope.expand_type(typ)
        for name, typ in variable_types.items()
    }
    
def has_identical_composition(candidate: TypeInfo, reference: TypeInfo, _seen_udts: dict[TypeStub, TypeStub] | None = None) -> bool:
    """Return true if both types have fields of the same types in the same order, and false otherwise. This is defined recursively for nested structs.

    The _seen_udts argument is used internally and should not be supplied by the caller.
    """
    if _seen_udts is None:
        _seen_udts = {}
    if type(candidate) == type(reference):
        if isinstance(candidate, (Struct, Union)):
            layouts = ((candidate.layout, reference.layout) if isinstance(candidate, Struct) else (candidate.members, reference.members)) # type: ignore (mypy doesn't handle the equivalent types condition on the if)
            if len(layouts[0]) != len(layouts[1]):
                return False
            _seen_udts[candidate.stub] = reference.stub # type: ignore
            for cand_member, ref_member in zip(*layouts): 
                if type(cand_member) != type(ref_member):
                    return False
                if isinstance(cand_member, (Struct, Union)):
                    if not has_identical_composition(cand_member, ref_member, _seen_udts): # type: ignore
                        return False
                else:
                    assert isinstance(cand_member, UDT.Field)
                    if not has_identical_composition(cand_member.type_name, ref_member.type_name, _seen_udts): # type: ignore
                        return False
        elif isinstance(candidate, Pointer):
            return has_identical_composition(candidate.target_type_name, reference.target_type_name, _seen_udts) # type: ignore
        elif isinstance(candidate, Array):
            return candidate.nelements == reference.nelements and has_identical_composition(candidate.element_type, reference.element_type, _seen_udts) # type: ignore
        elif isinstance(candidate, FunctionType):
            return len(candidate.parameters) == len(reference.parameters) and has_identical_composition(candidate.return_type, reference.return_type, _seen_udts) and all(has_identical_composition(t1, t2, _seen_udts) for (t1, _), (t2, _) in zip(candidate.parameters, reference.parameters)) # type: ignore
        elif isinstance(candidate, TypeStub):
            return _seen_udts[candidate] == reference
        else:
            return candidate == reference
    else:
        return False
    return True

def has_udt(typ: TypeInfo) -> bool:
    """Return true if there is a struct or union type, or a stub of such, in this type
    """
    if type(typ) is TypeInfo:
        return False
    # When types are expanded/standardized, the StructStub and UnionStub should become redundant, 
    # because they will only exist for recursive types, which means the struct/union is already defined.
    # However, this is handy pre-expansion/standardization for checking which types will contain structs/unions.
    if isinstance(typ, (Struct, Union, StructStub, UnionStub)):
        return True
    if isinstance(typ, Pointer):
        assert isinstance(typ.target_type_name, TypeInfo)
        return has_udt(typ.target_type_name)
    if isinstance(typ, Array):
        assert isinstance(typ.element_type, TypeInfo)
        return has_udt(typ.element_type)
    if isinstance(typ, FunctionType):
        return has_udt(typ.return_type) or any(has_udt(t) for t, _ in typ.parameters) 
    return False

def add_placeholders_for_nonvariable_types(node: Node, types: FileTypeMapping):
    """Functions may have references to types outside of their variables' types and return types, including
    in typecasts and sizeof expressions. This function identifies such situations and adds a generic
    placeholder to the FileTypeMapping so that they can be interpreted by PreprocessedFunction.
    """
    if node.type == "type_descriptor":
        base_type_node = node.child_by_field_name("type")
        assert base_type_node is not None and base_type_node.text is not None
        base_type_text = base_type_node.text.decode()
        if base_type_text not in types.types:
            typ = types.parse_type(base_type_node)
            assert typ is not None, f"Failed to parse type {base_type_text}."
            # Add generic placeholder UDT types. Normally we wouldn't want to do this, but
            # we'll never actually need the full definitions of the types during evaluation, 
            # so it's fine here.
            if isinstance(typ, StructStub):
                typ = Struct(name=typ.name, layout=[])
            elif isinstance(typ, UnionStub):
                typ = Union(name=typ.name, members=[])
            elif isinstance(typ, EnumStub):
                typ = Enum(name=typ.name, members=[])
            types.add_type(base_type_text, typ)
    elif node.type != "declaration":
        for child in node.children:
            add_placeholders_for_nonvariable_types(child, types)


### Utility functions for working with Alignment objects
def perfectly_aligned(alignment: Alignment) -> bool:
    return all(bool(alignment[op]) for bb in alignment.reference_ir for op in bb) and \
           all(bool(alignment[op]) for bb in alignment.candidate_ir for op in bb)

def get_aligned_variables(alignment: Alignment) -> dict[str, set[str]]:
    """Return the variables from the reference that align with those in the candidate.

    :alignment: an Alignment object for which to compute variable alignment.
    :returns: a dictionary mapping reference variable to target variable.
    """
    var_map: dict[str, set[str]] = {} # candidate (ground-truth) to reference (prediction)
    for candidate_op, reference_op in alignment.alignment_list:
        if isinstance(candidate_op, Parameter) or isinstance(reference_op, Parameter):
            assert type(candidate_op) == type(reference_op), f"Only paramters should be aligned with parameters."
            if reference_op.name not in var_map: # type: ignore
                var_map[reference_op.name] = set() # type: ignore
            var_map[reference_op.name].add(candidate_op.name) # type: ignore
        elif candidate_op is not None and candidate_op.var_operator is not None and isinstance(candidate_op.var_operator.result, Variable) and not candidate_op.var_operator.result.is_temporary and not isinstance(candidate_op.var_operator.result, GlobalVariable) and \
             reference_op is not None and reference_op.var_operator is not None and isinstance(reference_op.var_operator.result, Variable) and not reference_op.var_operator.result.is_temporary and not isinstance(reference_op.var_operator.result, GlobalVariable):
            if (ref_name := reference_op.var_operator.result.name) not in var_map:
                var_map[ref_name] = set()
            var_map[ref_name].add(candidate_op.var_operator.result.name)
    return var_map

### Functions for editing code to standard form ###
def try_dereference(t: TypeInfo) -> TypeInfo | None:
    """Returns the target of t if it's a pointer or array type. Otherwise, return None.
    """
    if isinstance(t, Pointer):
        assert isinstance(t.target_type_name, TypeInfo)
        return t.target_type_name
    elif isinstance(t, Array):
        assert isinstance(t.element_type, TypeInfo)
        return t.element_type
    else: # Can't dereference something that is not a pointer
        return None

def field_index(t: TypeInfo, field_expression: Node, udts: dict[TypeStub, UDT]) -> tuple[int, TypeInfo] | None:
    """Return the index in the struct or union that field_name occurs at or None if t is not a 
    struct or union or if the field does not exist.
    """
    assert field_expression.type == "field_expression"
    if get_child(field_expression, "operator").text.decode() == "->": # type: ignore
        t = try_dereference(t) # type: ignore # re-defining t as a variable that could be None.

    if isinstance(t, TypeStub) and t in udts:
        t = udts[t]

    if isinstance(t, (Struct, Union)): # t == None is filtered out here.
        field_name: str = get_child(field_expression, "field").text.decode() # type: ignore
        for i, f in enumerate(t.layout if isinstance(t, Struct) else t.members):
            if isinstance(f, UDT.Field) and f.name == field_name:
                assert isinstance(f.type_name, TypeInfo)
                return (i, f.type_name)

    return None

def get_type_of_field_expression_argument(expression: Node, variable_types: dict[str, TypeInfo], udts: dict[TypeStub, UDT]) -> TypeInfo | None:
    """Returns the type of the provided expression.
    """
    if expression.type == "identifier":
        variable_name: str = expression.text.decode() # type: ignore
        return variable_types.get(variable_name, None)
    if expression.type == "field_expression":
        t = get_type_of_field_expression_argument(get_child(expression, "argument"), variable_types, udts)
        if t is None:
            return None
        field_info = field_index(t, expression, udts)
        if field_info is None:
            return None
        else:
            return field_info[1]
    if expression.type == "parenthesized_expression":
        assert len(expression.children) == 3 and expression.children[0].type =="(" and expression.children[2].type == ")"
        return get_type_of_field_expression_argument(expression.children[1], variable_types, udts)
    if expression.type == "pointer_expression":
        operator: str = get_child(expression, "operator").text.decode() # type: ignore
        argument = get_child(expression, "argument")
        assert operator == "&" or operator == "*"
        t = get_type_of_field_expression_argument(argument, variable_types, udts)
        if t is None:
            return None
        if operator == "*":
            return try_dereference(t)
        else:
            return Pointer(t)
    if expression.type == "subscript_expression":
        argument = get_child(expression, "argument")
        t = get_type_of_field_expression_argument(argument, variable_types, udts)
        if isinstance(t, Array):
            assert isinstance(t.element_type, TypeInfo)
            return t.element_type
        if isinstance(t, Pointer):
            assert isinstance(t.target_type_name, TypeInfo)
            return t.target_type_name
        return None
    if expression.type == "cast_expression":
        # non () children are "type" and "value".
        # The value actually doesn't matter here because we only care about it for its type, but
        # the type is being changed to the type specifed in the cast. So we just parse and return that.
        descriptor = get_child(expression, "type")
        assert descriptor.type == "type_descriptor", f"Expected a type descriptor in a cast expression but found {descriptor.type}"
        base_type_node = descriptor.child_by_field_name("type")
        declarator = get_child(descriptor, "declarator")
        assert base_type_node is not None and base_type_node.text is not None
        base_type_text = base_type_node.text.decode()
        type_mapping = FileTypeMapping()
        typ = type_mapping.parse_type(base_type_node)
        assert typ is not None, f"Failed to parse type {base_type_text}."
        if isinstance(typ, (TypeStub)) and typ in udts:
            full_type, _ = type_mapping.parse_abstract_declarators(declarator, udts[typ])
            return full_type
        else:
            return None
    if expression.type == "binary_expression":
        return None # Could handle this, but is extremely rare and requires parsing both operands; exactly one must be a normal expression.
    if expression.type == "call_expression":
        return None # we can't do anything with this unless we know the called function's return type.
    
    raise NotImplementedError(f"Not supported: field name canonicalization: {expression.type}")

class NonexistentFieldError(Exception):
    pass

def canonicalize_udt_field_names(code: str, variable_types: dict[str, TypeInfo], user_defined_types: list[UDT]) -> str:
    """Replace all field names used in field expressions (e.g point.x or point->x) with the
    same standard name "field".
    """
    # For field_expressions:
    # expression.children[0]: (argument) - an expression that resolves to the struct
    # expression.children[1]: (operator) ->
    # expression.children[2]: (field) - the field being accessed.

    # Contains the changes we want to make to the text.
    # Tuples of (node to be deleted, text replacement).
    edits: list[tuple[Node, str]] = []

    udts = {t.stub: t for t in user_defined_types}

    def find_field_expression(node: Node):
        if node.type == "field_expression":
            t = get_type_of_field_expression_argument(get_child(node, "argument"), variable_types, udts)
            if t is not None:
                field_info = field_index(t, node, udts)
                if field_info is not None:
                    canonical_field_name = f"field{field_info[0]}"
                    edits.append((get_child(node, "field"), canonical_field_name))
                else:
                    raise NonexistentFieldError()
        for child in node.children:
            find_field_expression(child)

    root = parser.parse(bytes(code, 'utf8')).root_node
    find_field_expression(root) # populate the list 'edits'

    # Sorting edits in reverse order reduces the offset bookkeeping we have to do.
    edits.sort(key=lambda x: x[0].start_byte, reverse=True)
    assert all(a[0].start_byte > b[0].end_byte for a, b in zip(edits, itertools.islice(edits, 1, None)))

    start = root.start_byte # should always be 0 in this context
    text = root.text
    assert text is not None
    components = []
    for subnode, replacement in edits:
        components.append(text[(subnode.end_byte - start):])
        components.append(bytes(replacement, 'utf8'))
        text = text[:(subnode.start_byte - start)]
    components.append(text[(root.start_byte - start):])
    components.reverse() # We've been adding components backwards, reverse them for the correct output.
    return b"".join(components).decode("utf8")

### Running exebench tests ###
def get_function_name(definition: Node) -> str:
    """Get the name of a function.
    """
    assert definition.type == "function_definition", f"{definition.type} is not a function_definition"
    declarator = get_child(definition, "declarator")
    while declarator.type == "pointer_declarator":
        declarator = get_child(declarator, "declarator")
    
    assert declarator.type == "function_declarator"
    name = get_child(declarator, "declarator")
    assert name.type == "identifier"
    return name.text.decode("utf8") # type: ignore

def run_command_in_docker(
        command: list[str],
        cwd: str | None, # inside the docker container
        directory_mapping: dict[str | Path, str | Path] = {},
        timeout: float | None = None,
        image: str = 'exebench-test'
    ) -> subprocess.CompletedProcess[bytes]:
    """Run the command 'command' inside a docker container
    """
    full_command: list[str] = ["docker", "run", "--rm"]
    for host, container in directory_mapping.items():
        full_command.extend(["-v", f"{os.path.abspath(host)}:{container}"])
    if cwd is not None: # cwd of the command inside the docker container.
        full_command.extend(["-w", cwd])

    full_command.append(image)
    full_command.extend(command)

    return subprocess.run(full_command, timeout=timeout, capture_output=True)

DOCKER_VOLUME = "/fileio"
TEST_HARNESS_NAME = "prediction_harness.cpp"
PREDICTION_FILE_NAME = "prediction.c"
IO_PAIRS_JSON = "io_pairs.json"
EXEBENCH_TEST_OUT = "results.json"
def setup_and_run_docker_call(prediction_wrapper: str, prediction: str, io_pairs, fpermissive: bool = False) -> str | dict[str, dict[str, str | list[bool]]]:
    """Test the prediction on the io_pairs using the prediction wrapper in docker.

    Return a failure message if the command fails or the results of the test script in docker if available.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        # Prediction and test harness
        with open(os.path.join(tempdir, TEST_HARNESS_NAME), "w") as fp:
            fp.write(prediction_wrapper)
        with open(os.path.join(tempdir, PREDICTION_FILE_NAME), "w") as fp:
            fp.write(prediction)
        with open(os.path.join(tempdir, IO_PAIRS_JSON), "w") as fp:
            json.dump(io_pairs, fp)

        command = ["python", "run_tests.py", DOCKER_VOLUME, TEST_HARNESS_NAME, PREDICTION_FILE_NAME, IO_PAIRS_JSON, EXEBENCH_TEST_OUT]
        if fpermissive:
            command.append("--include-fpermissive")
        try:
            run = run_command_in_docker(command, "/exebench/exebench", {tempdir: DOCKER_VOLUME}, timeout=500)
        except subprocess.TimeoutExpired:
            return "timeout"

        results_file = os.path.join(tempdir, EXEBENCH_TEST_OUT)
        if not os.path.exists(results_file):
            return "unknown"

        with open(results_file, "r") as fp:
            results = json.load(fp)
    return results


def run_exebench_test(meta: dict[str, str | None], prediction: str) -> str | dict[str, dict[str, str | list[bool]]]:
    """Run an exebench test suite on a function (i.e. a model's prediction). Returns a string describing 
    the error if the test suite could not be run and the results of the trials
    (with standard and permissive compilation) as a dictionary upon success. 

    meta: an entry corresponding to a single function in exebench.
    predictions: the model's prediction
    :returns: an error type or the results of each test.
    """
    if meta['real_exe_wrapper'] is not None:
        exe_wrapper = meta['real_exe_wrapper']
    elif meta['synth_exe_wrapper'] is not None:
        exe_wrapper = meta['synth_exe_wrapper']
    else:
        raise ValueError(f"Missing exe wrapper for example {meta['fname']}")
    
    if meta['real_io_pairs'] is not None:
        io_pairs = meta['real_io_pairs']
    elif meta['synth_io_pairs'] is not None:
        io_pairs = meta['synth_io_pairs']
    else:
        raise ValueError(f"No IO pairs for testing {meta['fname']}")
    
    ### Modify the exe wrapper to work for this prediction and machine
    def edit_exe_wrapper(_predicted_name: str | None):
        # Include the prediction C file instead of the temporary C file hardcoded in the test harness which doesn't exist.
        # The real and synth partitions use different file paths at different locations, tmp and run, respectively. Synth also has extra path components.
        _exe_wrapper = re.sub(r"""\#include \"/(tmp|run)/(\w+/)*\w+\.c\"""", f"#include \"{PREDICTION_FILE_NAME}\"", exe_wrapper)
        # Use the predicted function name in the call instead of the function name
        if _predicted_name is not None:
            _exe_wrapper = re.sub(fr"""{meta['fname']}(\(.*\))""", _predicted_name + r"\1", _exe_wrapper)
        # Use relative header locations provided in the exebench repo instead of standard ones.
        _exe_wrapper = _exe_wrapper.replace("<nlohmann/json.hpp>", '"nlohmann/json.hpp"')
        _exe_wrapper = _exe_wrapper.replace("<clib/synthesizer.h>", '"clib/synthesizer.h"')
        return _exe_wrapper
    
    oracle_wrapper = edit_exe_wrapper(None)

    deps: str = meta['synth_deps'] if meta['synth_deps'] is not None else ""
    deps += (meta['real_deps'] if meta['real_deps'] is not None else "")
    assert meta['func_def'] is not None
    oracle_solution = deps + "\n\n" + meta['func_def']

    oracle_result = setup_and_run_docker_call(oracle_wrapper, oracle_solution, io_pairs, False)
    if isinstance(oracle_result, str) or oracle_result["standard"]["error"] is not None or not all(r for r in oracle_result["standard"]["tests"]):
        return "oracle_failure"

    # We need to parse the predicted solution for two reasons:
    # 1. The test provided in exebench has the original name of the function hardcoded. 
    #    The prediction may be functionally correct, but have a different name. Therefore,
    #    we have to identify what that name is so we can edit the test and use it instead.
    # 2. Problems with data structures: If a data structure prediction was cut off due to the 
    #    token limit, that prediction will be syntatically invalid and cause g++ to crash. 
    #    Additionally, sometimes there may be problems with a predicted data structure that
    #    is unused by the function itself. In this case, the solution can still be salvaged.
    #    We include only the types that are parsable and relevant, and hope it works. If not
    #    then it'll be counted as incorrect anyway.
    root = parser.parse(bytes(prediction, "utf8")).root_node
    predicted_fn: Node | None = None
    other_nodes: list[Node] = []
    types = FileTypeMapping()
    for node in root.children:
        # tree-sitter can sometimes get confused and identify things that are definitely not function definitions
        # (e.g. struct definitions) as function definitions. However, when it does so, it flags them as functions with errors.
        # We do need to find the predicted function so we can use its name in the test harness below, but otherwise we'll let
        # g++ determine whether or not something is syntatically/semantically incorrect. (The actual compiler is the gold
        # standard for syntatic/semantic correctness for our purposes; conformity with the C standard would be another but
        # that's harder to do automatically). Thus, we'll just ignore this erroneous node here and see if it causes problems
        # down the line.
        if predicted_fn is None and node.type == "function_definition" and not node.has_error:
            predicted_fn = node
        else:
            other_nodes.append(node)
            try:
                types.parse_type(node)
            except:
                # Do nothing. If this type was necessary, then PreprocessedFunction will fail, and
                # we'll just feed the raw output in other_nodes to g++, which is the final arbiter of 
                # whether or not the code will compile.
                pass
    
    if predicted_fn is None:
        return "no_functions" # There was no recognizable function found.
    try:
        predicted_name = get_function_name(predicted_fn)
    except AssertionError:
        return "function_name"

    if len(other_nodes) > 0 and other_nodes[-1].has_error and not other_nodes[-1].is_missing:
        other_nodes = other_nodes[:-1]

    # Get only the relevant/used UDTs in the function. If there's a problem, just use all possible
    # udt nodes and let g++ sort it out, if possible.
    try:
        used_udts: list[str] = [
            udt.declaration("") + ";" for udt in 
            get_all_user_defined_types(PreprocessedFunction(predicted_fn, types))
        ]
    except (AssertionError, TypeNotFoundError, TypeNotDefinedError, UnsupportedFeatureError):
        used_udts: list[str] = [node.text.decode() for node in other_nodes if node.text is not None]
    
    prediction_wrapper = edit_exe_wrapper(predicted_name)

    ### Write the tests out to files and run the test in docker
    prediction = "\n\n".join(used_udts)
    prediction += predicted_fn.text.decode() # type: ignore
    results = setup_and_run_docker_call(prediction_wrapper, prediction, io_pairs, True)
    if isinstance(results, dict):
        for run_name, run_result in results.items():
            # != is xor
            assert (run_result["error"] is None) != (run_result["tests"] is None), run_name + str(run_result)
    return results


### Do exebench tests.
def calculate_executable_metrics(predictions: list[tuple[MatchedFunction, str]], exebench_entries: list[dict[str, str | None]]) -> dict[str, float]:
    """Run each prediction's exebench tests and report related metrics.

    :param predictions: the model's predictions. The MatchedFunction is unused.
    :param exebench_entries: the corresponding dataset entries from exebench. Expected to be parallel to `predictions`.
    :returns: metrics in a dictionary mapping metric name to value.
    """
    precompilation_errors: list[str] = []

    standard_results = []
    permissive_results = []
    oracle_failures: int = 0

    for (_, prediction), meta, in tqdm(zip(predictions, exebench_entries), desc="Running exebench tests", total=len(predictions), dynamic_ncols=True):
        result = run_exebench_test(meta, prediction)
        if result == "oracle_failure":
            oracle_failures += 1
        elif isinstance(result, str):
            precompilation_errors.append(result)
        else:
            standard_results.append(result["standard"])
            permissive_results.append(result["permissive"])

    def compute_trial_level_metrics(runnable_results: list[dict[str, str | list[bool]]], metric_postfix: str = ""):
        """Compute metrics that occur under each trial (marked by different compilation settings.)
        """
        correct = 0
        partially_correct = 0
        compilation_errors = 0
        for result in runnable_results:
            assert (result["error"] is None) != (result["tests"] is None), str(result)
            if result["error"] is not None:
                assert result["error"] == "compilation" # currently this is the only type of error that we record from exebench_docker/run_tests.py
                compilation_errors += 1
            else:
                test_results = result["tests"]
                assert isinstance(test_results, list) and all(isinstance(r, bool) for r in test_results)
                correct += all(test_results)
                partially_correct += any(test_results)
        base_num = len(predictions) - oracle_failures
        return {
            f"exebench_correct{metric_postfix}": correct / base_num,
            f"exebench_partially_correct{metric_postfix}": partially_correct / base_num,
            f"exebench_total_errors{metric_postfix}": (len(precompilation_errors) + compilation_errors) / base_num,
            f"exebench_compilation_errors{metric_postfix}": compilation_errors / base_num
        }
    
    metrics = compute_trial_level_metrics(standard_results)
    metrics |= compute_trial_level_metrics(permissive_results, "_permissive")
    
    for err, count in Counter(precompilation_errors).items():
        metrics[f"exebench_{err}_errors"] = count / len(predictions)
    
    metrics["oracle_failures"] = oracle_failures
    
    return metrics


### File IO
def write_output_to_files(results: list[tuple[MatchedFunction, str]], stem: Path, exebench_info: list[dict[str, str]] | None = None):
    """Write raw model output along with the ground-truth solution in machine-readable and human-readable formats.

    :param results: model predictions
    :param stem: the file name, minus the extension. Will create .json for the machine-readable version and .c for the human-readable version.
    :param write_exebench_info: writes the exebench entries for each function in `stem`.json a parallel json file named `stem`_exebench_info.json. Requires that the exebench info is attached to each function with the attribute name in ORIGINAL_EXAMPLE_ATTR.
    """
    serialized = [
        (fn.to_json(), prediction) for fn, prediction in results
    ]

    with open(stem.with_suffix(".json"), "w") as fp:
        json.dump(serialized, fp)

    if exebench_info is not None:
        with open(stem.parent / (stem.name + "_exebench_info.json"), "w") as fp:
            json.dump(exebench_info, fp)


    printable = [
        (
            fn.canonical_original_code + "\n\n" + "\n\n".join(udt.declaration("") for udt in fn.user_defined_types),
            prediction
        )
        for fn, prediction in results
    ]

    with open(stem.with_suffix(".c"), "w") as fp:
        for original, prediction in printable:
            fp.write(original)
            fp.write("\n// ----\n")
            fp.write(prediction)
            fp.write("\n\n")
            fp.write("// " + "*" * 40)
            fp.write("\n\n")

def read_predictions(existing_predictions_file: Path) -> list[tuple[MatchedFunction, str]]:
    with open(existing_predictions_file, "r") as fp:
        predictions = [
            (MatchedFunction.from_json(fn), prediction)
            for fn, prediction in json.load(fp)
        ]
    return predictions

def read_exebench_info(exebench_info_file: Path) -> list[dict[str, str | None]]:
    with open(exebench_info_file, "r") as fp:
        exebench_info = json.load(fp)
    return exebench_info


#### utils for prediction
def exebench_to_matched_function(example: dict[str, str]) -> MatchedFunction | None:
    """Convert an exebench entry to a matched_function. There isn't enough
    information to fill out some fields but there should be enough to make
    the evaluation work.
    """
    if example['hex-rays'] is None:
        return None
    types = FileTypeMapping()
    deps: str = example['synth_deps'] if example['synth_deps'] is not None else ""
    deps += (example['real_deps'] if example['real_deps'] is not None else "")
    deps = re.sub(r'(/\*.*?\*/)|(//.*)', '', deps) # remove comments
    root = parser.parse(bytes(deps, 'utf-8')).root_node
    for node in root.children:
        types.parse_type(node)

    tree = parser.parse(bytes(example['func_def'], 'utf-8'))
    root = tree.root_node
    assert len(root.children) == 1 and root.children[0].type == "function_definition", \
        f"Expected func_def field to contain one function_definition node but found: " + ", ".join(c.type for c in root.children)
    try:
        fn = PreprocessedFunction(root.children[0], types)
    except (AssertionError, TypeNotFoundError, TypeNotDefinedError, UnsupportedFeatureError):
        return None
    path = Path(example['path'])
    matched_function = MatchedFunction(
        name=example['fname'],
        canonical_name='func0', # there's only one function per binary so this must be the case. Also, this is not used in function mode.
        repo=path.parts[0] + "/" + path.parts[1],
        decompiled_code=example['hex-rays'], # TODO: switch this to a placeholder because we technically don't have the un-canonical decompiled code
        canonical_decompiled_code=example['hex-rays'],
        original_code=example['func_def'],
        canonical_original_code=fn.canonical_text,
        memory_layout={},
        variable_types=fn.variable_types,
        return_type=fn.return_type,
        user_defined_types=get_all_user_defined_types(fn),
        binary_hash=example['path'] # contains info on repo/function
    )
    # Attach the original example to the function so that we can get it later for the purposes of running the executable tests.
    setattr(matched_function, ORIGINAL_EXAMPLE_ATTR, example)
    return matched_function

def make_signature(fn: MatchedFunction) -> str:
    return fn.binary_hash + "_" + fn.name

def test_function_stringify(fninfo: tuple[MatchedBinary, int]):
    """Wrapper around causal_stringify_function_prompt to be used with MatchedBinaryFunctionWrapper.
    """
    return causal_stringify_function_prompt(fninfo[0].functions[fninfo[1]])

def test_neighbors_stringify(fninfo: tuple[MatchedBinary, int], nhops: int, tokenizer: PreTrainedTokenizerBase | None, max_context: int | None):
    """Wrapper around causal_stringify_neighbors_prompt to be used with MatchedBinaryFunctionWrapper.
    """
    binary, fn_index = fninfo
    return causal_stringify_neighbors_prompt(binary, binary.functions[fn_index], nhops, tokenizer, max_context)

def test_binary_stringify(fninfo: tuple[MatchedBinary, int]):
    """Wrapper around causal_stringify_binary_prompt to be used with MatchedFunctionBinaryWrapper.
    """
    binary, fn_index = fninfo
    return causal_stringify_binary_prompt(binary, binary.functions[fn_index])

def test_tokenize(batch: list[T], stringify: Callable[[T], str], tokenizer: PreTrainedTokenizerBase, max_length: int):
    """Run the tokenizer on a list of strings with several arguments pre-set.
    """
    return tokenizer([stringify(b) for b in batch], return_tensors='pt', max_length=max_length, padding=True, truncation=True)

def predict(
        model: PreTrainedModel | PeftModel, 
        tokenizer: PreTrainedTokenizerBase, 
        evaluation_set: Iterable[T],
        stringify_fn: Callable[[T], str], 
        batch_size: int, 
        max_context_length: int, 
        max_new_tokens: int, 
        limit: int | None, 
        device: str
    ) -> list[tuple[T, str]]:
    """For each example in evaluation_set, predict the original code from the decompiled.

    model: a Huggingface Transformers model.
    tokenizer: the tokenizer for `model`.
    evaluation_set: the dataset
    stringify_fn: a callable that converts an item of the dataset into a string which can be tokenized.
    batch_size: the number of examples to evaluate at once
    max_context_length: the total amount of decompiled information
    max_new_tokens: the total number of new tokens that can be generated for the solution.
    limit: generate solutions for exactly this many examples instead of all of them.
    device: where to put the tensors for predicting. Should match the device of the model.
    """
    results = []
    collate_fn = functools.partial(test_tokenize, stringify=stringify_fn, tokenizer=tokenizer, max_length=max_context_length)
    # By zipping the predictions with the predictions (in the return statement) we're loading the whole evaluation dataset into
    # memory anyway, so we might as well just do it here and fail fast if we run out of memory.
    eval_list = list(evaluation_set)
    if limit is not None:
        random.shuffle(eval_list)
        eval_list = eval_list[:limit]
    for batch in tqdm(DataLoader(eval_list, batch_size=batch_size, collate_fn=collate_fn, shuffle=False), desc="Predicting...", dynamic_ncols=True): # type: ignore  (a list is technically not a dataset but it has __getitem__ and __len__ methods like a dataset, so it duck-types.)
        batch.to(device)
        # Note: pad_token_id=tokenizer.pad_token_id suppresses the warning "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.""
        predictions = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        results.extend(decoded)
    return list(zip(eval_list, results))

def main(args: argparse.Namespace):
    random.seed(args.random_seed)
    eval_batch_size = args.batch_size
    max_prediction_length: int = args.max_prediction_length
    max_decompiled_function_size: int = args.max_decompiled_function_size
    missing_predictions_only: bool = args.missing_predictions_only
    do_exebench_tests: bool = not args.no_exebench_tests
    eval_partition: str = args.eval_partition
    assert not (args.evaluate_existing_predictions and missing_predictions_only), f"--evaluate-existing-predictions and --missing-predictions-only are contradictory."

    checkpoint_loc = Path(args.checkpoint)
    with open((checkpoint_loc.parent if checkpoint_loc.name[:10] == "checkpoint" else checkpoint_loc) / "idioms_config.json", "r") as fp:
        idioms_config = json.load(fp)
        # These argument must match those used in training; read them from config instead of asking for them.
        model_type: str = idioms_config["model_type"]
        dataset_path: str = idioms_config["dataset"]
        mode: str = idioms_config["mode"]
        nhops: int = idioms_config["nhops"] # only relevant in neighbors mode.
        max_seq_len: int = idioms_config["max_seq_len"]
        has_adapter: bool = idioms_config["adapter"]
        print(f"Model has adapter: {has_adapter}")

    if args.max_context_length is None:
        max_context_length = max_seq_len - max_prediction_length
    else:
        max_context_length: int = args.max_context_length
    assert max_context_length >= max_decompiled_function_size, f"--max-decompiled-function-size={max_decompiled_function_size} is too large for the available context: {max_context_length}"
    
    if args.dataset is not None:
        dataset_path = args.dataset
    dataset_dir = Path(dataset_path)

    # Is a huggingface datasets style dataset. Assume it is exebench, because that's the only
    # one we support (and that makes sense here.) The program will crash if it doesn't have the
    # required fields.
    dataset_is_exebench = (dataset_dir / "dataset_info.json").exists() or (dataset_dir / "dataset_dict.json").exists()
    assert not dataset_is_exebench or mode == "function", f"Exebench-derived information do not have binary-level info. Run in function mode."

    if dataset_is_exebench:
        eval_partition = f"test_{args.exebench_subpartition}" if eval_partition == "test" else f"valid_{args.exebench_subpartition}"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ) if has_adapter else None

    tokenizer = AutoTokenizer.from_pretrained(model_type, padding_side='left')
    print(f"{tokenizer.__class__}'s PAD and EOS IDs.")
    print("Pad token info", tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.pad_token_type_id)
    print("EOS token info", tokenizer.eos_token, tokenizer.eos_token_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting PAD TOKEN = EOS TOKEN. Now:")
        print("Pad token info", tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.pad_token_type_id)
        print("EOS token info", tokenizer.eos_token, tokenizer.eos_token_id)


    def function_size_filter(fn: MatchedFunction) -> bool:
        """Return True if the decompiled code fits within the allowed context size.
        """
        return len(tokenizer.encode(causal_stringify_function_prompt(fn))) <= max_decompiled_function_size

    def valid_missing_example_filter(fn: MatchedFunction, signatures: Container[str]) -> bool:
        return function_size_filter(fn) and make_signature(fn) not in signatures
    
    if checkpoint_loc.parts[0] == "runs":
        output_dir = Path("results", *checkpoint_loc.parts[1:])
    else:
        output_dir = Path("results", *checkpoint_loc.parts)
    if dataset_is_exebench:
        output_dir = output_dir / dataset_dir.name
    os.makedirs(output_dir, exist_ok=True)

    existing_predictions_file = output_dir / f"{eval_partition}_results.json"
    existing_exebench_info_file = output_dir / f"{eval_partition}_results_exebench_info.json"

    if args.evaluate_existing_predictions:
        assert existing_predictions_file.exists(), f"Specified --evaluate-existing-predictions but there are no existing predictions at {existing_predictions_file}. Generate them by running without --evaluate-existing-predictions."
        predictions = read_predictions(existing_predictions_file)
        if dataset_is_exebench:
            exebench_info = read_exebench_info(existing_exebench_info_file)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_type if has_adapter else checkpoint_loc,
            quantization_config=quantization_config,
            torch_dtype=None if has_adapter else torch.bfloat16
        )

        if has_adapter:
            # model = model.merge_and_unload()
            model = PeftModel.from_pretrained(model, checkpoint_loc / ADAPTER_NAME if (checkpoint_loc / ADAPTER_NAME).exists() else checkpoint_loc)
        else:
            print("Sending model to cuda")
            model.to("cuda") # type: ignore

        if missing_predictions_only:
            assert existing_predictions_file.exists(), f"Must have existing predictions to use --missing-predictions-only but none are found at {existing_predictions_file}. Rerun without --missing-predictions-only to generate."
            existing_predictions = read_predictions(existing_predictions_file)
            # Attach exebench info to existing_predictions if relevant.
            if dataset_is_exebench:
                existing_exebench_info = read_exebench_info(existing_exebench_info_file)
                assert len(existing_predictions) == len(existing_exebench_info), f"Number of predictions ({len(existing_predictions)}) do not match the number of exebench info entries ({len(existing_exebench_info)})"
                for prediction, info in zip(existing_predictions, existing_exebench_info):
                    setattr(prediction[0], ORIGINAL_EXAMPLE_ATTR, info) # prediction[0] is the ground truth MatchedFunction
                del prediction, info, existing_exebench_info
            # Filter out stuff that's too big
            existing_predictions = [p for p in existing_predictions if function_size_filter(p[0])]
            
            # Set up the filter that eliminates functions with existing predictions.
            function_validity_filter = functools.partial(valid_missing_example_filter, 
                signatures={make_signature(fn) for fn, _ in existing_predictions}
            )
        else:
            function_validity_filter = function_size_filter

        # It's important that this goes before the dataset_is_exebench if-statement below because that statement modifies stringify_fn
        if mode == "function":
            stringify_fn = test_function_stringify
        elif mode == "neighbors":
            stringify_fn = functools.partial(test_neighbors_stringify, nhops=nhops, tokenizer=tokenizer, max_context=max_context_length)
        else:
            assert mode == "binary", mode
            stringify_fn = test_binary_stringify

        if dataset_is_exebench:
            raw_dataset = load_from_disk(dataset_dir)
            if isinstance(raw_dataset, DatasetDict):
                raw_dataset = raw_dataset[eval_partition]
                
            # the inner filter removes invalid examples (those that are None and therefore not considered True in bool(...))
            # the outer filter removes examples that are too large, or in missing_examples_only mode, examples that already have predictions.
            holdout_set = filter(function_validity_filter, filter(None, map(exebench_to_matched_function, raw_dataset))) # type: ignore

            # Adjust other relevant settings now for dealing with an exebench dataset.
            # insead of the standard idioms dataset.
            stringify_fn = causal_stringify_function_prompt
        else:
            # Make a MatchedBinaryDataset regardless of the mode (function vs binary) and implement the correct
            # behavior in the stringify functions passed to `predict`. This is to use the length filter in MatchedBinaryFunctionWrapper.
            holdout_set = MatchedBinaryFunctionWrapper(
                MatchedBinaryDataset(dataset_dir.glob(f"{eval_partition}*.tar"), shuffle=False),
                function_filter=function_validity_filter
            )

        predictions = predict(model, tokenizer, holdout_set, stringify_fn, eval_batch_size, max_context_length, max_prediction_length, args.limit, "cuda") # type: ignore

        if dataset_is_exebench:
            get_matched_function = lambda x: x
        else:
            get_matched_function = lambda x: x[0].functions[x[1]]
        predictions = [(get_matched_function(fninfo), new_text.split(DECOMPILED_ORIG_SEP)[1]) for fninfo, new_text in predictions if new_text.count(DECOMPILED_ORIG_SEP) > 0] # type: ignore

        if missing_predictions_only:
            predictions = existing_predictions + predictions # type: ignore

        exebench_info = [getattr(r[0], ORIGINAL_EXAMPLE_ATTR) for r in predictions] if dataset_is_exebench else None # Save this exebench info so that it can be used later for executable-based testing.
        assert not dataset_is_exebench or len(predictions) == len(exebench_info), f"Size mismatch: {len(predictions)} predictions, but {len(exebench_info)} exebench entries." # type: ignore
        write_output_to_files(predictions, output_dir / f"{eval_partition}_results", exebench_info)

    evaluator = FunctionEvaluator()
    scores = evaluator(predictions)

    if dataset_is_exebench and do_exebench_tests and len(predictions) > 0:
        assert exebench_info is not None # for the benefit of mypy
        exebench_scores = calculate_executable_metrics(predictions, exebench_info)
        scores |= exebench_scores
        with open(output_dir / f"{eval_partition}_exebench_scores.json", "w") as fp:
            json.dump(exebench_scores, fp, indent=2)

    for metric, score in scores.items():
        print(metric, score)
    print()

    with open(output_dir / f"{eval_partition}_scores.json", "w") as fp:
        json.dump(scores, fp, indent=2)


if __name__ == "__main__":
    main(get_args())