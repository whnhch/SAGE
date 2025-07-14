from typing import List, Dict, Any
import pandas as pd
import json
from src.utility.llama import *
from tqdm import tqdm
import os
    
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

def generate_all_groupby_configs(
    df: pd.DataFrame,
    interesting_attribute_names: List[str],
    llm: Llama3,
    save_path: str = "intermediate_configs.jsonl",
    checkpoint_every: int = 100,
    resume: bool = True,
    show_progress: bool = True
) -> List[Dict[str, Any]]:

    def is_valid_group_or_column(attr_name: str, values: List[Any]) -> bool:
        return len(values) >= 2 and len(values) <= 100

    already_saved = set()
    if resume and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                try:
                    cfg = json.loads(line)
                    key = json.dumps({
                        k: cfg[k] for k in ['index_attr', 'index_values', 'column_attr', 'value_attr', 'aggfunc', 'insight_type', 'bucket']
                    }, sort_keys=True)
                    already_saved.add(key)
                except:
                    continue

    # Determine candidate columns
    unique_values_dict = retrieve_unique_values_from_df(df, interesting_attribute_names)
    candidate_group_cols = [col for col, vals in unique_values_dict.items() if is_valid_group_or_column(col, vals)]
    candidate_value_cols = df.select_dtypes(include='number').columns.tolist()

    aggfuncs = ['mean', 'count', 'min', 'max', 'sum']
    correlation_buckets = ['very high', 'high', 'neutral', 'low', 'very low']
    ratio_buckets = ['similar_magnitude', 'moderate difference', 'large difference', 'very large difference', 'extreme difference']

    buffer = []
    total_saved = 0

    batch_prompts = []
    batch_keys = []
    batch_configs = []
    BATCH_SIZE = 64
    
    for group_col in tqdm(candidate_group_cols, desc="Group Cols", disable=not show_progress):
        group_vals = unique_values_dict[group_col]
        other_group_cols = [col for col in candidate_group_cols if col != group_col]

        for col in tqdm(other_group_cols, desc=f"  → Column Attrs ({group_col})", leave=False, disable=not show_progress):
            for i in range(len(group_vals)):
                for j in range(i + 1, len(group_vals)):
                    val_pair = sorted([group_vals[i], group_vals[j]])
                    if val_pair[0] == val_pair[1]:
                        continue
                    
                    for value_col in candidate_value_cols:
                        for agg in aggfuncs:
                            for corr_bucket, rat_bucket in zip(correlation_buckets, ratio_buckets):
                                for insight_type, bucket in [('corr', corr_bucket), ('rat', rat_bucket)]:
                                    config = {
                                        'index_attr': group_col,
                                        'index_values': val_pair,
                                        'column_attr': col,
                                        'value_attr': value_col,
                                        'aggfunc': agg,
                                        'insight_type': insight_type,
                                        'bucket': bucket
                                    }

                                    key = json.dumps({
                                        k: config[k] for k in ['index_attr', 'index_values', 'column_attr', 'value_attr', 'aggfunc', 'insight_type', 'bucket']
                                    }, sort_keys=True)

                                    if key in already_saved:
                                        continue

                                    prompt = build_prompt(config, insight_type)  
                                    batch_prompts.append(prompt)
                                    batch_configs.append((key, config))                                    
                                    already_saved.add(key)
                                    
                                    if len(batch_prompts) == BATCH_SIZE:
                                        results = run_batch(batch_prompts, batch_configs, llm)
                                        buffer+=results
                                        
                                        batch_prompts, batch_configs = [], []

                                        if len(buffer) >= checkpoint_every:
                                            with open(save_path, 'a') as f:
                                                for b in buffer:
                                                    f.write(json.dumps(b) + '\n')
                                            total_saved += len(buffer)
                                            buffer = []
    if batch_prompts:
        results = run_batch(batch_prompts, batch_configs, llm)
        buffer += results
    
    if buffer:
        with open(save_path, 'a') as f:
            for b in buffer:
                f.write(json.dumps(b) + '\n')
        total_saved += len(buffer)

    print(f"Total saved configs: {total_saved}")
    return buffer

def run_batch(prompts, configs, llm):
    responses = llm.chatbot(prompts)
    results = []
    for (key, config), response in zip(configs, responses):
        label = llm.find_value_in_response(response, "likelihood")
        config_with_label = {**config, "label": label}
        results.append(config_with_label)
    return results

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
    
    return prompt
  
def get_llm_label(config: Dict[str, Any], insight_type: str, llm:Llama3) -> str:
    prompt = None
    if insight_type == 'corr':
        prompt = (
        f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_values'][0]}' and "
        f"'{config['index_values'][1]}' show a {config['bucket']} correlation "
        f"when grouped by '{config['column_attr']}'."
        )
    else:
        prompt = (
        f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_values'][0]}' and "
        f"'{config['index_values'][1]}' show a {config['bucket']} ratio difference "
        f"when grouped by '{config['column_attr']}'."
        )
    
    def get_likelihood(instruction: str, llm: Llama3) -> str:
        """
        Get likelihood using llm and instruction.
        There is a function (load_prommpt) for this in prompt class but here we don't call that for simplicity.

        Args:
            instruction (str): instruction made above
        """
        prompt_ = f"""# Task Description: {instruction} Choose the appropriate likelihood from following scale. Return the result as JSON in the following format: {{"likelihood": "<one likelihood from the scale>"}}. Please return only the JSON output. Do not include explanations, code, or the full table.
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
        response = llm.chatbot(prompt_)
        result= llm.find_value_in_response(response, "likelihood")
        return result
    
    label = get_likelihood(prompt, llm)
    return label

def generate_outlier_dataset(
    df: pd.DataFrame,
    interesting_attribute_names: List[str],
    llm: Llama3,
    save_path: str = "outlier_intermediate_configs.jsonl",
    checkpoint_every: int = 100,
    resume: bool = True,
    show_progress: bool = True
) -> List[Dict[str, Any]]:

    already_saved = set()
    if resume and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                try:
                    cfg = json.loads(line)
                    key = json.dumps({
                        k: cfg[k] for k in ['index_attr', 'index_value', 'column_attr', 'column_value', 'value_attr', 'aggfunc']
                    }, sort_keys=True)
                    already_saved.add(key)
                except:
                    continue

    unique_values_dict = retrieve_unique_values_from_df(df, interesting_attribute_names)
    candidate_group_cols = [col for col, vals in unique_values_dict.items() if len(vals) >= 2]
    candidate_value_cols = df.select_dtypes(include='number').columns.tolist()

    aggfuncs = ['mean', 'count', 'min', 'max', 'sum']

    buffer = []
    total_saved = 0

    batch_prompts = []
    batch_keys = []
    batch_configs = []
    
    for group_col in tqdm(candidate_group_cols, desc="Group Cols", disable=not show_progress):
        idx_vals = unique_values_dict[group_col]
        other_group_cols = [col for col in candidate_group_cols if col != group_col]

        for col in tqdm(other_group_cols, desc=f"  → Column Attrs ({group_col})", leave=False, disable=not show_progress):
            col_vals = unique_values_dict[col]
            
            for idx_val in idx_vals:
                for col_val in col_vals:
                    for value_col in candidate_value_cols:
                        for agg in aggfuncs:
                                    config = {
                                        'index_attr': group_col,
                                        'index_value': idx_val,
                                        'column_attr': col,
                                        'column_value': col_val,
                                        'value_attr': value_col,
                                        'aggfunc': agg,
                                    }

                                    key = json.dumps({
                                        k: config[k] for k in ['index_attr', 'index_value', 'column_attr', 'column_value', 'value_attr', 'aggfunc']
                                    }, sort_keys=True)

                                    if key in already_saved:
                                        continue

                                    prompt = build_outlier_prompt(config)  
                                    batch_prompts.append(prompt)
                                    batch_configs.append((key, config))                                    
                                    already_saved.add(key)
                                    
                                    if len(batch_prompts) == checkpoint_every:
                                        results = run_batch(batch_prompts, batch_configs, llm)
                                        buffer+=results
                                        
                                        batch_prompts, batch_configs = [], []

                                        if len(buffer) >= checkpoint_every:
                                            with open(save_path, 'a') as f:
                                                for b in buffer:
                                                    f.write(json.dumps(b) + '\n')
                                            total_saved += len(buffer)
                                            buffer = []
    if batch_prompts:
        results = run_batch(batch_prompts, batch_configs, llm)
        buffer += results
    
    if buffer:
        with open(save_path, 'a') as f:
            for b in buffer:
                f.write(json.dumps(b) + '\n')
        total_saved += len(buffer)

    print(f"Total saved configs: {total_saved}")
    return buffer

def build_outlier_prompt(config: Dict[str, Any], llm:Llama3) -> str:
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
    
    return prompt

def bucketize_correlation(value: float) -> str:
    if value >= 0.9:
        return 'very high'
    elif value >= 0.7:
        return 'high'
    elif value >= 0.5:
        return 'neutral'
    elif value >= 0.3:
        return 'low'
    else:
        return 'very low'
    
def bucketize_ratio(value: float) -> str:
    if value <= 1.5:
        return 'similar magnitude'
    elif value <= 3.0:
        return 'moderate difference'
    elif value <= 10.0:
        return 'large difference'
    elif value <= 30.0:
        return 'very large difference'
    else:
        return 'extreme_difference'


def save_configs_to_json(configs: List[Dict[str, Any]], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(configs, f, indent=2)

def save_json_lines(configs: List[Dict[str, Any]], filename: str, mode: str = 'a'):
    with open(filename, mode) as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')