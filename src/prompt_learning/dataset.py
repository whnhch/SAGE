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

def generate_all_configs(unique_vals, group_cols, value_cols):
    aggfuncs = ['mean', 'count', 'min', 'max', 'sum']
    corr_buckets = ['very high', 'high', 'neutral', 'low', 'very low']
    rat_buckets = ['similar_magnitude', 'moderate difference', 'large difference', 'very large difference', 'extreme difference']

    for group_col in group_cols:
        group_vals = unique_vals[group_col]
        other_group_cols = [c for c in group_cols if c != group_col]
        for col in other_group_cols:
            for i in range(len(group_vals)):
                for j in range(i + 1, len(group_vals)):
                    val_pair = sorted([group_vals[i], group_vals[j]])
                    for value_col, agg, corr, rat in product(value_cols, aggfuncs, corr_buckets, rat_buckets):
                        for insight_type, bucket in [('corr', corr), ('rat', rat)]:
                            config = {
                                'index_attr': group_col,
                                'index_values': val_pair,
                                'column_attr': col,
                                'value_attr': value_col,
                                'aggfunc': agg,
                                'insight_type': insight_type,
                                'bucket': bucket
                            }
                            yield config


def config_key(config):
    return f"{config['index_attr']}|{config['index_values'][0]}|{config['index_values'][1]}|{config['column_attr']}|{config['value_attr']}|{config['aggfunc']}|{config['insight_type']}|{config['bucket']}"

def build_prompt_task(config):
    key = config_key(config)
    prompt, instruction = build_prompt(config, config['insight_type'])
    return key, config, prompt, instruction

def build_prompt(config: Dict[str, Any], insight_type: str) -> str:
    instruction = None
    if insight_type == 'corr':
        instruction = (
        f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_values'][0]}' and "
        f"'{config['index_values'][1]}' show a {config['bucket']} correlation "
        f"when grouped by '{config['column_attr']}'."
        )
    else:
        instruction = (
        f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_values'][0]}' and "
        f"'{config['index_values'][1]}' show a {config['bucket']} ratio difference "
        f"when grouped by '{config['column_attr']}'."
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