import itertools
import os
import numpy as np
import csv
from typing import List
import time
from scipy.spatial.distance import cdist
import pandas as pd
import json
from multiprocessing import Pool, get_context
from itertools import combinations, islice

def sqrt_cdist(matrix1: np.ndarray , matrix2: np.ndarray ):
    dist = cdist(matrix1, matrix2, metric="cosine")/2.0
    dist = np.clip(dist, 0, None)
    dist = np.sqrt(dist)

    return dist

def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each row to unit L2 norm."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)

def concat_embeddings(query_embeddings_list: List[np.ndarray], table_embeddings_list: List[np.ndarray], output_dir: str = "./embedding_distances"
) -> np.ndarray:
    """
    Concatenate query and table embeddings (with optional scaling) and save cosine distances.
    
    Args:
        query_embeddings_list: List of query embedding vectors (np.ndarray)
        table_embeddings_list: List of table embedding vectors (np.ndarray)
        output_dir: directory where CSVs are saved
    Returns:
        np.ndarray: Concatenated embeddings
    """
    os.makedirs(output_dir, exist_ok=True)

    query_np = np.vstack(query_embeddings_list)
    table_np = np.vstack(table_embeddings_list)

    table_np = table_np / 2.0

    def save_cdist(matrix: np.ndarray, name: str):
        dist = cdist(matrix, matrix, metric="cosine")/2.0
        df = pd.DataFrame(dist)
        df.to_csv(os.path.join(output_dir, f"{name}_cosine_distance.csv"), index=False)

    def save_sqrt_cdist(matrix: np.ndarray, name: str):
        dist = cdist(matrix, matrix, metric="cosine")/2.0
        dist = np.clip(dist, 0, None)
        dist = np.sqrt(dist)
        df = pd.DataFrame(dist)
        df.to_csv(os.path.join(output_dir, f"{name}_cosine_distance.csv"), index=False)


    combined = np.hstack([query_np, table_np])

    return combined

def get_all_embeddings(sql_queries, pivot_tables, query_tokenizer, query_model, table_tokenizer, table_model, device):
    """
    Compute and return embeddings for lists of SQL queries and pivot tables.

    Args:
        sql_queries (List[str]): List of SQL query strings.
        pivot_tables (List[str]): List of pivot table strings.
        query_tokenizer: Tokenizer for query model.
        query_model: Embedding model for SQL queries.
        table_tokenizer: Tokenizer for table model.
        table_model: Embedding model for tables.
        device: Device to run models on (e.g., 'cpu' or 'cuda').

    Returns:
        Tuple[List[Tensor], List[Tensor]]: 
            - query_embeddings_list: List of query embeddings
            - table_embeddings_list: List of table embeddings
    """
    query_embeddings_list = []
    table_embeddings_list = []

    for sql_query, pivot in zip(sql_queries, pivot_tables):
        query_vec, table_vec = get_embeddings(
            query_tokenizer, query_model,
            table_tokenizer, table_model,
            sql_query, pivot, device
        )
        query_embeddings_list.append(query_vec)
        table_embeddings_list.append(table_vec)

    return query_embeddings_list, table_embeddings_list


def get_embeddings(query_tokenizer, query_model, table_tokenizer, table_model,
                   sql_query, pivot,
                   device):
    query_inputs = query_tokenizer(sql_query, return_tensors="pt", padding=True, truncation=True).to(device)
    query_outputs = query_model(**query_inputs)

    query_vec = query_outputs.last_hidden_state.mean(dim=1)

    pivot.columns = [str(col) if col is not None else "" for col in pivot.columns]
    table_inputs = table_tokenizer(table=pivot.astype(str), query="Summarize the table", return_tensors="pt", padding=True, truncation=True).to(device)
    table_outputs = table_model.model.encoder(**table_inputs)

    table_vec = table_outputs.last_hidden_state.mean(dim=1)

    return query_vec.squeeze(0).detach().cpu().numpy(), table_vec.squeeze(0).detach().cpu().numpy()

def compute_and_concat_embeddings_with_timing(
    sql_queries,
    pivot_tables,
    query_tokenizer,
    query_model,
    table_tokenizer,
    table_model,
    device,
    output_dir,
    concat_embeddings_fn
):
    """
    Compute embeddings, concatenate them, and record timing.

    Args:
        sql_queries (List[str])
        pivot_tables (List[str])
        query_tokenizer, query_model
        table_tokenizer, table_model
        device (str)
        output_dir (str): Path to save timing CSV
        concat_embeddings_fn (function): Function to combine two lists of embeddings

    Returns:
        combined_embeddings: Result of concat_embeddings_fn
    """
    os.makedirs(output_dir, exist_ok=True)

    start_embed = time.time()
    query_embeddings_list, table_embeddings_list = get_all_embeddings(
        sql_queries, pivot_tables,
        query_tokenizer, query_model,
        table_tokenizer, table_model,
        device
    )
    end_embed = time.time()

    start_concat = time.time()
    combined_embeddings = concat_embeddings_fn(query_embeddings_list, table_embeddings_list)
    end_concat = time.time()

    # Save timing info
    time_path = os.path.join(output_dir, "embedding_time.csv")
    with open(time_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding_time_sec", "concat_time_sec"])
        writer.writeheader()
        writer.writerow({
            "embedding_time_sec": round(end_embed - start_embed, 4),
            "concat_time_sec": round(end_concat - start_concat, 4),
        })

    return combined_embeddings

def greedy_selection(queries, utility_results, embeddings, distance_threshold, k, output_dir):
    """
    Greedy selection with cosine distance threshold and logging.

    Saves:
    - selected_queries.json
    - diversity_result.csv: query, utility, min distance to others
    - selected_distance_matrix.csv
    - selected_distance_matrix_labeled.csv
    - full_distance_matrix.csv (before selection)
    - full_distance_matrix_labeled.csv
    - diversity_time.csv
    - params.json

    Returns:
        selected_indices: List[int]
    """
    os.makedirs(output_dir, exist_ok=True)

    utility_scores = [res["utility score"] for res in utility_results]

    print("Computing full distance matrix...")
    full_dist_matrix = sqrt_cdist(embeddings, embeddings) 

    np.fill_diagonal(full_dist_matrix, np.inf)

    np.savetxt(os.path.join(output_dir, "full_distance_matrix.csv"),
               full_dist_matrix, delimiter=",", fmt="%.6f")

    pd.DataFrame(full_dist_matrix, index=queries, columns=queries).to_csv(
        os.path.join(output_dir, "full_distance_matrix_labeled.csv"),
        float_format="%.6f"
    )

    print("Sorting utility scores...")
    start_sort = time.time()
    utility_sorted_indices = np.argsort(-np.array(utility_scores))
    end_sort = time.time()

    print("Running greedy diversity selection...")
    selected_indices = []
    selected_vectors = []
    start_select = time.time()

    for idx in utility_sorted_indices:
        if len(selected_indices) >= k:
            break

        candidate_vec = embeddings[idx]

        if not selected_vectors:
            selected_indices.append(idx)
            selected_vectors.append(candidate_vec)
            continue

        selected_matrix = np.vstack(selected_vectors)
        distances = sqrt_cdist([candidate_vec], selected_matrix) 
        
        min_dist = np.min(distances)

        if min_dist >= distance_threshold:
            selected_indices.append(idx)
            selected_vectors.append(candidate_vec)

    end_select = time.time()

    selected_queries = [queries[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]
    selected_dist_matrix = sqrt_cdist(selected_embeddings, selected_embeddings) 
        
    np.fill_diagonal(selected_dist_matrix, np.inf)
    query_min_dists = np.min(selected_dist_matrix, axis=1)

    with open(os.path.join(output_dir, "diversity_result.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "utility_score", "min_distance_to_others"])
        for idx, query, dist in zip(selected_indices, selected_queries, query_min_dists):
            score = utility_scores[idx]
            writer.writerow([query, round(float(score), 6), round(float(dist), 6)])

    np.savetxt(os.path.join(output_dir, "selected_distance_matrix.csv"),
               selected_dist_matrix, delimiter=",", fmt="%.6f")

    pd.DataFrame(selected_dist_matrix, index=selected_queries, columns=selected_queries).to_csv(
        os.path.join(output_dir, "selected_distance_matrix_labeled.csv"),
        float_format="%.6f"
    )

    with open(os.path.join(output_dir, "diversity_time.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sort_time_sec", "selection_time_sec"
        ])
        writer.writeheader()
        writer.writerow({
            "sort_time_sec": round(end_sort - start_sort, 4),
            "selection_time_sec": round(end_select - start_select, 4),
        })

    with open(os.path.join(output_dir, "selected_queries.json"), "w") as f:
        json.dump(selected_queries, f, indent=2)

    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump({
            "k": k,
            "distance_threshold": distance_threshold
        }, f, indent=2)

    print(f"Finished. Selected {len(selected_indices)} items.")
    return selected_indices

def _check_combinations_worker(args):
    comb_chunk, utility_scores, embeddings, distance_threshold = args
    best_subset = None
    best_utility = -np.inf

    for subset in comb_chunk:
        vectors = embeddings[list(subset)]
        distances = sqrt_cdist(vectors, vectors)
        np.fill_diagonal(distances, np.inf)
        if np.all(distances >= distance_threshold):
            utility = sum(utility_scores[i] for i in subset)
            if utility > best_utility:
                best_utility = utility
                best_subset = subset

    return best_subset, best_utility


def chunked_combinations(indices, k, chunk_size):
    """Yield chunks of combinations of size `chunk_size`"""
    buffer = []
    for comb in itertools.combinations(indices, k):
        buffer.append(comb)
        if len(buffer) == chunk_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def chunked_combinations_generator(indices, k, chunk_size):
    iterator = combinations(indices, k)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

def _check_combinations_worker(args):
    comb_chunk, utility_scores, embeddings, distance_threshold = args
    best_score = -np.inf
    best_subset = None
    for comb in comb_chunk:
        vectors = embeddings[list(comb)]
        dists = sqrt_cdist(vectors, vectors)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) >= distance_threshold:
            score = sum(utility_scores[i] for i in comb)
            if score > best_score:
                best_score = score
                best_subset = comb
    return best_subset, best_score

def bruteforce_selection_parallel(queries, utility_results, embeddings, distance_threshold, k, output_dir, num_workers=4, chunk_size=10000):
    os.makedirs(output_dir, exist_ok=True)

    utility_scores = [res["utility score"] for res in utility_results]
    num_candidates = len(utility_scores)
    indices = list(range(num_candidates))

    print(f"[BruteForce] Searching over {num_candidates} candidates (k={k}), multiprocessing with {num_workers} workers...")

    start_time = time.time()
    total_checked = 0
    best_global_subset = None
    best_global_utility = -np.inf
    
    with get_context("spawn").Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(
            _check_combinations_worker,
            ((chunk, utility_scores, embeddings, distance_threshold)
             for chunk in chunked_combinations_generator(indices, k, chunk_size))
        ):
            subset, score = result
            total_checked += chunk_size
            if subset and score > best_global_utility:
                best_global_utility = score
                best_global_subset = subset

    end_time = time.time()

    if best_global_subset is None:
        print("[BruteForce] No valid subset found.")
        return []

    selected_indices = list(best_global_subset)
    selected_queries = [queries[i] for i in selected_indices]
    selected_vectors = embeddings[selected_indices]
    selected_dist_matrix = sqrt_cdist(selected_vectors, selected_vectors)
    np.fill_diagonal(selected_dist_matrix, np.inf)
    query_min_dists = np.min(selected_dist_matrix, axis=1)

    dist_matrix_path = os.path.join(output_dir, "selected_distance_matrix.csv")
    np.savetxt(dist_matrix_path, selected_dist_matrix, delimiter=",", fmt="%.6f")

    with open(os.path.join(output_dir, "diversity_result.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "utility_score", "min_distance_to_others"])
        for idx, query, dist in zip(selected_indices, selected_queries, query_min_dists):
            score = utility_scores[idx]
            writer.writerow([query, round(float(score), 6), round(float(dist), 6)])

    elapsed = end_time - start_time
    with open(os.path.join(output_dir, "diversity_time.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Elapsed time (seconds)", f"{elapsed:.4f}"])
        writer.writerow(["combinations_checked", total_checked])

    print(f"[BruteForce] Done. Selected {len(selected_indices)} queries. Utility={best_global_utility:.4f}, Time={elapsed:.2f}s")
    return selected_indices

def bruteforce_selection(queries, utility_results, embeddings, distance_threshold, k, output_dir):
    """
    Brute-force selection that finds the best k-sized subset maximizing utility
    while ensuring all pairwise distances >= threshold.

    Saves:
    - diversity_result.csv: selected queries with utility and min pairwise distances
    - diversity_time.csv: time taken for brute-force
    - params.json (optional)

    Returns:
        selected_indices: List[int]
    """
    os.makedirs(output_dir, exist_ok=True)

    utility_scores = [res["utility score"] for res in utility_results]
    num_candidates = len(utility_scores)
    
    print(f"Running bruteforce over {num_candidates} candidates (selecting {k})...")

    start_time = time.time()

    best_subset = None
    best_total_utility = -np.inf

    indices = list(range(num_candidates))
    total_checked = 0

    for subset in itertools.combinations(indices, k):
        vectors = embeddings[list(subset)]
        distances = sqrt_cdist(vectors, vectors)
        np.fill_diagonal(distances, np.inf)

        if np.all(distances >= distance_threshold):
            total_utility = sum(utility_scores[i] for i in subset)
            if total_utility > best_total_utility:
                best_total_utility = total_utility
                best_subset = subset
        total_checked += 1
        if total_checked % 100000 == 0:
            print(f"Checked {total_checked:,} combinations...")

    end_time = time.time()

    if best_subset is None:
        print("No valid subset found satisfying diversity constraint.")
        return []

    selected_indices = list(best_subset)
    selected_queries = [queries[i] for i in selected_indices]
    selected_vectors = embeddings[selected_indices]
    selected_dist_matrix = sqrt_cdist(selected_vectors, selected_vectors)
    np.fill_diagonal(selected_dist_matrix, np.inf)
    query_min_dists = np.min(selected_dist_matrix, axis=1)

    dist_matrix_path = os.path.join(output_dir, "selected_distance_matrix.csv")
    np.savetxt(dist_matrix_path, selected_dist_matrix, delimiter=",", fmt="%.6f")

    with open(os.path.join(output_dir, "diversity_result.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "utility_score", "min_distance_to_others"])
        for idx, query, dist in zip(selected_indices, selected_queries, query_min_dists):
            score = utility_scores[idx]
            writer.writerow([query, round(float(score), 6), round(float(dist), 6)])

    elapsed_time = end_time - start_time
    time_path = os.path.join(output_dir, "diversity_time.csv")
    with open(time_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Elapsed time (seconds)", f"{elapsed_time:.4f}"])
        writer.writerow(["combinations_checked", total_checked])

    print(f"Finished bruteforce. Selected {len(selected_indices)} items with total utility {best_total_utility:.4f}")
    return selected_indices


def topk_selection(queries, utility_results, k, output_dir):
    """
    Greedy selection with cosine distance threshold and logging.

    Saves:
    - selected_queries.json
    - selected_distances.csv: each selected query with its min distance to others
    - diversity_timing.csv
    - params.json

    Returns:
        selected_indices: List[int]
    """
    os.makedirs(output_dir, exist_ok=True)

    utility_scores = [res["utility score"] for res in utility_results]
    print("Sorting utility scores...")
    start_sort = time.time()
    utility_sorted_indices = np.argsort(-np.array(utility_scores))
    end_sort = time.time()

    print("Running greedy diversity selection...")
    selected_indices = []
    selected_vectors = []
    start_select = time.time()

    for idx in utility_sorted_indices:
        if len(selected_indices) >= k:
            break
        
        if not selected_vectors:
            selected_indices.append(idx)
            continue

    end_select = time.time()

    selected_queries = [queries[i] for i in selected_indices]

    with open(os.path.join(output_dir, "diversity_result.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query"])
        for query, dist in zip(selected_queries):
            writer.writerow([query, round(float(dist), 6)])

    with open(os.path.join(output_dir, "diversity_timie.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sort_time_sec", "selection_time_sec"
        ])
        writer.writeheader()
        writer.writerow({
            "sort_time_sec": round(end_sort - start_sort, 4),
            "selection_time_sec": round(end_select - start_select, 4),
        })

    print(f"Finished. Selected {len(selected_indices)} items.")
    return selected_indices