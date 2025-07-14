
import os
import random
import torch
from pathlib import Path

from src.utility.utility import indicator
from src.utility.interpretability import NaiveInterpretability
import joblib
from transformers import set_seed, AutoTokenizer, T5EncoderModel, TapexTokenizer, BartForConditionalGeneration
import sys
import time
from typing import List, Dict, Any
import json
import csv
import multiprocessing
import traceback
import math

from multiprocessing import Pool, get_context
from typing import Optional

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v_lower = v.lower()
        if v_lower in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v_lower in ('no', 'false', 'f', 'n', '0'):
            return False

def run_with_timeout(target_func, timeout_sec, **kwargs):
    def wrapper(q):
        try:
            result = target_func(**kwargs)
            q.put(("success", result))
        except Exception as e:
            q.put(("error", traceback.format_exc()))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(q,))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout", None

    status, result = q.get()
    if status == "error":
        return "error", result
    return "success", result

def set_params(args):
    params = {
        "filename": getattr(args, "filename", None),
        "k": getattr(args, "k", None),
        "threshold": getattr(args, "threshold", None),
        "int_threshold": getattr(args, "int_threshold", None),
        "do_prune": getattr(args, "do_prune", None),
        "data_num": getattr(args, "data_num", None),
        "do_parallel": getattr(args, "do_parallel", None),
        "do_cache": getattr(args, "do_cache", None),
        "max_columns": getattr(args, "max_columns", None),
    }
    return params
    
def set_environment(transformer_path, hf_token_path):
    project_root = Path().resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.environ['TRANSFORMERS_CACHE'] = transformer_path
    os.environ['HF_HOME'] = transformer_path
    os.environ['HF_DATASETS_CACHE'] = transformer_path
    
    with open(hf_token_path, "r") as f:
        hf_token = f.read().strip() 

    os.environ["HF_TOKEN"] = hf_token
    print("HF_TOKEN has been set:", os.environ["HF_TOKEN"])

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)

def param_dict_to_folder(params: dict) -> str:
    return "_".join(f"{k}{str(v).replace('.', '-')}" for k, v in sorted(params.items()))

def get_experiment_folder(base_dir: str, params: dict) -> str:
    folder_name = param_dict_to_folder(params)
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_experiment_folders(base_output_dir: str, params: dict):
    def get_folder(subdir: str) -> str:
        folder_name = param_dict_to_folder(params)
        path = os.path.join(base_output_dir, subdir, folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    prune_dir = get_folder("prune")
    utility_dir = get_folder("utility")
    diversity_dir = get_folder("diversity")
    
    with open(os.path.join(prune_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
        
    with open(os.path.join(utility_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
        
    with open(os.path.join(diversity_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    return prune_dir, utility_dir, diversity_dir

def load_classifiers(clf_trd_path, clf_out_path):
    clf_trd = joblib.load(clf_trd_path)
    clf_out = joblib.load(clf_out_path)
    return clf_trd, clf_out

def load_encoders(query_checkpoint, table_checkpoint, device):
    query_tokenizer = AutoTokenizer.from_pretrained(query_checkpoint)
    query_model = T5EncoderModel.from_pretrained(query_checkpoint).to(device)

    table_tokenizer = TapexTokenizer.from_pretrained(table_checkpoint)
    table_model = BartForConditionalGeneration.from_pretrained(table_checkpoint).to(device)
    
    return query_tokenizer, query_model, table_tokenizer, table_model

def prune_combinations(
    combs: List[Dict[str, Any]],
    interesting_attrs: List[str],
    df,
    unique_values_dict: Dict[str, List[Any]],
    aggfunc_ranks: Dict[str, float],
    int_threshold: float,
    output_dir: str,
    d_size: int = 100,
    t_size: int = 100,
    verbose: bool = True
) -> List[Dict[str, Any]]:

    start_time = time.time()
    pruned_result = []

    for comb in combs:
        if not comb.get("aggfunc") or not comb.get("index") or not comb.get("value"):
            continue

        if not indicator(comb, interesting_attrs):
            continue

        int_model = NaiveInterpretability(
            d_size=d_size,
            t_size=t_size,
            original_table=df,
            query=comb,
            unique_dict=unique_values_dict,
            aggfunc_ranks=aggfunc_ranks,
            prune=True
        )
        int_model.calculate_scores()
        interpretability = int_model.interpretability_score

        if interpretability >= int_threshold:
            pruned_result.append(comb)

    elapsed_time = time.time() - start_time

    if pruned_result:
        keys = pruned_result[0].keys()
        csv_path = os.path.join(output_dir, "pruned.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(pruned_result)

    time_path = os.path.join(output_dir, "pruned_time.csv")
    with open(time_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Elapsed time (seconds)", f"{elapsed_time:.4f}"])
        writer.writerow(["Num pruned", len(pruned_result)])
        writer.writerow(["Total combinations", len(combs)])

    if verbose:
        print(f"[Prune] Saved {len(pruned_result)} results in {elapsed_time:.2f}s → {time_path}")

    return pruned_result

def _evaluate_combination(args):
    comb, interesting_attrs, df, unique_values_dict, aggfunc_ranks, d_size, t_size, int_threshold = args

    if not comb.get("aggfunc") or not comb.get("index") or not comb.get("value"):
        return None

    if not indicator(comb, interesting_attrs):
        return None

    int_model = NaiveInterpretability(
        d_size=d_size,
        t_size=t_size,
        original_table=df,
        query=comb,
        unique_dict=unique_values_dict,
        aggfunc_ranks=aggfunc_ranks,
        prune=True
    )
    int_model.calculate_scores()
    interpretability = int_model.interpretability_score

    if interpretability >= int_threshold:
        return comb
    return None


def prune_combinations_parallel(
    combs: List[Dict[str, Any]],
    interesting_attrs: List[str],
    df,
    unique_values_dict: Dict[str, List[Any]],
    aggfunc_ranks: Dict[str, float],
    int_threshold: float,
    output_dir: str,
    d_size: int = 100,
    t_size: int = 100,
    verbose: bool = True,
    num_workers: int = 8
) -> List[Dict[str, Any]]:

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    args_list = [
        (comb, interesting_attrs, df, unique_values_dict, aggfunc_ranks, d_size, t_size, int_threshold)
        for comb in combs
    ]
    
    chunk_size = math.ceil(len(combs) / num_workers)

    with get_context("spawn").Pool(num_workers) as pool:
        results = pool.imap_unordered(_evaluate_combination, args_list, chunksize=chunk_size)
        pruned_result = [res for res in results if res is not None]

    elapsed_time = time.time() - start_time

    if pruned_result:
        keys = pruned_result[0].keys()
        csv_path = os.path.join(output_dir, "pruned.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(pruned_result)

    time_path = os.path.join(output_dir, "pruned_time.csv")
    with open(time_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Elapsed time (seconds)", f"{elapsed_time:.4f}"])
        writer.writerow(["Num pruned", len(pruned_result)])
        writer.writerow(["Total combinations", len(combs)])

    if verbose:
        print(f"[Prune-MP] Saved {len(pruned_result)} results in {elapsed_time:.2f}s → {time_path}")

    return pruned_result