from typing import List, Dict, Any
import pandas as pd
import json
from src.utility.llama import *
from tqdm import tqdm
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def retrieve_unique_values(attr: pd.Series, max_unique: int = 100, bin_numeric: bool = True, bin_count: int = 5) -> List[Any]:
    attr = attr.dropna()
    unique_vals = attr.unique()
    
    if len(unique_vals) > max_unique:
        return []

    if bin_numeric and pd.api.types.is_numeric_dtype(attr) and len(unique_vals) > 10:
        try:
            binned = pd.qcut(attr, q=bin_count, duplicates='drop')
            return binned.astype(str).unique().tolist()
        except ValueError:
            return unique_vals.tolist()
    
    return unique_vals.tolist()

def retrieve_unique_values_from_df(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[Any]]:
    return {col: retrieve_unique_values(df[col]) for col in columns}

def retrieve_unique_values_from_df(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[Any]]:
    return {col: retrieve_unique_values(df[col]) for col in columns}

def generate_all_configs(unique_vals, index_cols, value_cols):
    aggfuncs = ['mean', 'count', 'min', 'max', 'sum']

    for group_col in index_cols:
        idx_vals = unique_vals[group_col]
        other_group_cols = [c for c in index_cols if c != group_col]
        for col in other_group_cols:
            col_vals = unique_vals[col]
            for idx_val in idx_vals:
                for col_val in col_vals:
                    for value_col, agg in product(value_cols, aggfuncs):
                            config = {
                                        'index_attr': group_col,
                                        'index_value': idx_val,
                                        'column_attr': col,
                                        'column_value': col_val,
                                        'value_attr': value_col,
                                        'aggfunc': agg,
                                    }
                            yield config

def config_key(config):
    return f"{config['index_attr']}|{config['index_value']}|{config['column_attr']}|{config['column_value']}|{config['value_attr']}|{config['aggfunc']}"

def build_prompt_task(config):
    key = config_key(config)
    prompt, instruction = build_prompt(config)
    return key, config, prompt, instruction

def build_prompt(config: Dict[str, Any]) -> str:
    instruction = (
        f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_value']}' and "
        f"'{config['column_attr']} '{config['column_value']}' shows an outlier"
        )
    
    prompt = f"""# Task Description: {instruction} Choose the appropriate likelihood from following scale. Return the result as JSON in the following format: {{"likelihood": "<one likelihood from the scale>"}}. Please return only the JSON output. Do not include explanations, code, or the full table.
    ## Input: 
    **Likelihood Scale:** 
    Very Likely
    Likely
    Neutral
    Unlikely
    Very Unlikely
            
    Return the result as JSON in the following format: {{"likelihood": "<one likelihood from the scale>"}}.
            
    ## Output:
    """
    
    return prompt, instruction

def sample_generator(generator, k):
    items = []
    for i, item in enumerate(generator):
        if len(items) < k:
            items.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                items[j] = item
    return items

def parallel_prompt_generation(df, interesting_attrs, already_saved, max_workers=8, sample_size=1000):
    unique_vals = retrieve_unique_values_from_df(df, interesting_attrs)
    group_cols = [col for col, vals in unique_vals.items() if 2 <= len(vals) <= 100]
    value_cols = df.select_dtypes(include='number').columns.tolist()

    sampled_configs = sample_generator(generate_all_configs(unique_vals, group_cols, value_cols), k=sample_size)

    prompt_tuples = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for config in sampled_configs:
            key = config_key(config)
            if key in already_saved:
                continue
            futures.append(executor.submit(build_prompt_task, config))

        for future in tqdm(as_completed(futures), total=len(futures)):
            key, config, prompt, instruction = future.result()
            prompt_tuples.append((key, config, prompt, instruction))

    return prompt_tuples

def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def save_json_lines(configs: List[Dict[str, Any]], filename: str, mode: str = 'a'):
    with open(filename, mode) as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')
            
def make_saver(save_path, checkpoint_every=100):
    buffer = []
    total_saved = 0

    def save(record):
        nonlocal buffer, total_saved
        buffer.append(record)
        if len(buffer) >= checkpoint_every:
            with open(save_path, 'a') as f:
                for item in buffer:
                    f.write(json.dumps(item) + '\n')
            total_saved += len(buffer)
            buffer = []

    def flush():
        nonlocal buffer, total_saved
        if buffer:
            with open(save_path, 'a') as f:
                for item in buffer:
                    f.write(json.dumps(item) + '\n')
            total_saved += len(buffer)
            buffer = []
        print(f"Total saved: {total_saved}")

    return save, flush