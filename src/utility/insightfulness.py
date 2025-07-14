'''
Implementation for insightfulness.
Section 3.1 in the paper.
'''
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from numba import njit
from src.utility.llama import *

LIKELIHOOD_MAP={
    "Very Likely":0.2,
    "Likely":0.4,
    "Neutral":0.6,
    "Unlikely":0.8,
    "Very Unlikely":1.0
    }

from numba import njit
import numpy as np

@njit
def compute_high_ratio_pairs(arr, tau=0.8):
    n, d = arr.shape
    num_pairs = n * (n - 1) // 2
    scores = np.zeros(num_pairs)
    i_list = np.empty(num_pairs, dtype=np.int64)
    j_list = np.empty(num_pairs, dtype=np.int64)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            min_ab = np.inf
            min_ba = np.inf
            for k in range(d):
                a, b = arr[i, k], arr[j, k]

                if b != 0:
                    val_ab = a / b
                    if val_ab < min_ab:
                        min_ab = val_ab

                if a != 0:
                    val_ba = b / a
                    if val_ba < min_ba:
                        min_ba = val_ba

            score = max(min_ab, min_ba)
            scores[idx] = score
            i_list[idx] = i
            j_list[idx] = j
            idx += 1

    # Count number of scores > tau
    count = 0
    for i in range(num_pairs):
        if scores[i] > tau:
            count += 1

    # Allocate filtered output arrays
    indices = np.empty((count, 2), dtype=np.int64)  # for (i, j)
    scores_ = np.empty(count, dtype=np.float64)
    idx_out = 0
    for i in range(num_pairs):
        if scores[i] > tau:
            indices[idx_out, 0] = i_list[i]
            indices[idx_out, 1] = j_list[i]
            scores_[idx_out] = scores[i]
            idx_out += 1

    return indices, scores_

def convert_trend_vars_to_feature(vars: dict, index_attr: str, insight_type="corr", bucket="unknown") -> dict:
    return {
        "index_attr": index_attr,
        "index_values": '|'.join(sorted([str(vars["groupA"]), str(vars["groupB"])])),
        "column_attr": vars["column_attributes"],
        "value_attr": vars["value_attribute"],
        "aggfunc": vars["aggfunc"],
        "insight_type": insight_type,
        "bucket": bucket
    }


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
        return 'extreme difference'

class NaiveInsightfulness:
    def __init__(self, pt, d_size=None, t_size=100, original_table=None, query=None, llm: Llama3=None, pr:Prompt=None,
                 tau_cor:float=0.7, tau_rat:float=5.0, clf_trd=None, clf_out=None, gamma=None):
        # df : NxM
        self.d_size = d_size
        self.t_size = t_size
        
        self.query = query
        self.original_table = original_table
        
        if query: self.aggfunc = query['aggfunc']
        
        self.informativeness_score=0
        
        self.interesting_attributes = None
        self.correlation_score=0
        self.ratio_score=0
        self.trend_score=0
        
        self.surprise_score=0
        
        self.insightfulness_score=0
        
        self.llm = llm
        self.pr = pr
        
        self.tau_cor = tau_cor
        self.tau_rat = tau_rat
        
        self.clf_trd = clf_trd
        self.clf_out = clf_out
        
        self.gamma = gamma
        
    def calculate_scores(self, t):
        self.informativeness_score=self.calculate_inf(t.fillna(0))

        self.correlation_score = self.calculate_correlation(t.fillna(0), tau=self.tau_cor)
        self.ratio_score = self.calculate_ratio(t.fillna(0))
        self.trend_score = max(self.correlation_score, self.ratio_score)
                
        self.surprise_score=self.calculate_surprising(t)

        self.insightfulness_score = max(self.informativeness_score, self.trend_score,  self.surprise_score)

    def get_likelihood_clf(self, task_name: str = 'correlation', vars: dict = None) -> float:
        if vars is None:
            print("Warning: 'vars' is None, returning 0.")
            return 0

        x_df = pd.DataFrame([vars])

        if task_name != 'outlier':
            if self.clf_trd:
                try:
                    y_pred = self.clf_trd.predict(x_df)[0]
                    return LIKELIHOOD_MAP.get(y_pred, 0)
                except Exception as e:
                    return 0
            else:
                return 0

        else:
            if self.clf_out:
                try:
                    y_pred = self.clf_out.predict(x_df)[0]
                    return LIKELIHOOD_MAP.get(y_pred, 0)
                except Exception as e:
                    return 0
            else:
                return 0

        
    def get_likelihood(self, task_name: str = 'correlation', vars: dict = None) -> float:
        instruction = self.pr.load_instruction(task_name, vars)
        prompt = self.pr.load_prompt(task_name, {"instruction": instruction})

        response = self.llm.chatbot(prompt)
        result= self.llm.find_value_in_response(response, self.llm.completion_keys[task_name])
        final_result = LIKELIHOOD_MAP[result]
                
        return final_result
        
    def calculate_inf(self, t: pd.DataFrame) -> float:
        n, m = t.shape
        
        row_pairwise_distances, col_pairwise_distances = None, None
        
        def compute_pairwise_euclidean(arr, norm_size, gamma):
            pairwise_distance = pdist(arr, metric='euclidean')
            pairwise_distance /= norm_size
            pairwise_distance /= gamma
            
            return pairwise_distance
        
        if n == 1: row_informativeness_score = 0.0
        else:
            data_array = t.values
            row_pairwise_distances = compute_pairwise_euclidean(data_array, m, self.gamma)
            row_informativeness_score = np.sum(row_pairwise_distances)/ (n * (n - 1) / 2)
            
        if m == 1: col_informativeness_score = 0.0
        else:
            if n == 1:
                data_array = t.T[1:].to_numpy().reshape(-1, 1)
            else:
                data_array = t.T.values
            col_pairwise_distances = compute_pairwise_euclidean(data_array, n, self.gamma)
            col_informativeness_score = np.sum(col_pairwise_distances)/ (m * (m - 1) / 2)

        informative_score = max(row_informativeness_score, col_informativeness_score)
        
        return informative_score
    
    def calculate_correlation(self, t, tau=0.8):
        n, m = t.shape
        correlation_score = None
        row_corr_score, col_corr_score = None, None
        row_pairwise_corr, col_pairwise_corr = None, None
        data_array = t.values
        
        def compute_pairwise_correlation(arr, IsRow=True):
            corr = np.corrcoef(arr, rowvar=IsRow)
            corr_vals = corr[np.triu_indices(corr.shape[0], k=1)]
            corr_vals = np.abs(corr_vals)
    
            corr_vals = np.nan_to_num(corr_vals, nan=0)

            return corr_vals
        
        def compute_high_corr_pairs(arr, tau_cor, IsRow=True):
            if not IsRow:
                arr = arr.T

            corr = np.corrcoef(arr)
            corr = np.nan_to_num(corr, nan=0)

            i_indices, j_indices = np.triu_indices(corr.shape[0], k=1)
            corr_vals = corr[i_indices, j_indices]
            abs_corr_vals = np.abs(corr_vals)

            mask = abs_corr_vals > tau_cor

            return list(zip(
                i_indices[mask],
                j_indices[mask],
                corr_vals[mask],
                abs_corr_vals[mask]
            ))
    
        def get_correlation_likelihood(i, j, val, IsRow):
            if self.clf_trd:
                cor_vars=self.pr.prepare_correlation_clf(i,j,isRow=IsRow)
                cor_vars['bucket'] = bucketize_correlation(abs(val))
                result = self.get_likelihood_clf('correlation',cor_vars)
            else:
                sign = "positive" if val >= 0 else "negative"
                cor_vars=self.pr.prepare_correlation(i,j,val,sign,isRow=IsRow)
                result = self.get_likelihood('correlation',cor_vars)
            return result

        if n == 1: # Case with only one row
            row_corr_score = [0]
        elif np.all(np.std(data_array, axis=1) == 0):
            row_corr_score = [0]
        else:
            likelihoods, pairwise_corr = None, None
            high_corr_pairs_vals = compute_high_corr_pairs(data_array, tau_cor=tau, IsRow=True)
            if not high_corr_pairs_vals:
                row_pairwise_corr = 0.0
            else:
                likelihoods = [get_correlation_likelihood(a, b, c, True) for a, b, c, d in high_corr_pairs_vals]
                pairwise_corr = [d for a, b, c, d in high_corr_pairs_vals]
                row_pairwise_corr = [l * r for l, r in zip(likelihoods, pairwise_corr)]
            row_corr_score = np.sum(row_pairwise_corr) / (n * (n - 1) / 2)
            
        if m == 1:  # Case with only one column
            col_corr_score = [0]
        elif np.all(np.std(data_array, axis=0) == 0):
            col_corr_score = [0]
        else:
            high_corr_pairs_vals = compute_high_corr_pairs(data_array, tau_cor=tau, IsRow=False)
            if not high_corr_pairs_vals:
                col_pairwise_corr = 0.0
            else:
                likelihoods = [get_correlation_likelihood(a, b, c, False) for a, b, c, d in high_corr_pairs_vals]
                pairwise_corr = [d for a, b, c, d in high_corr_pairs_vals]
                col_pairwise_corr = [l * r for l, r in zip(likelihoods, pairwise_corr)]
            col_corr_score = np.sum(col_pairwise_corr) / ((m * (m - 1) / 2))

        correlation_score = max(row_corr_score, col_corr_score)

        return correlation_score
                
    def calculate_ratio(self, t, tau=2.0):
        n, m = t.shape
        
        row_rat_score, col_rat_score = None, None
        row_mins, col_mins = None, None 
    
        def get_inverese_ratio(arr):
            with np.errstate(divide='ignore', invalid='ignore'):
                inf_mask = arr != np.inf
                null_mask = arr != 0.0
                result = np.zeros_like(arr, dtype=np.float64)
                result[inf_mask & null_mask] = 1 - 1 / arr[inf_mask & null_mask]
                
                result = np.sum(result)
            return result
        
        def get_ratio_likelihood(i, j, val, isRow):
            if self.clf_trd:
                rat_vars=self.pr.prepare_ratio_clf(i,j,isRow=isRow)
                rat_vars['bucket'] = bucketize_ratio(abs(val))
                result = self.get_likelihood_clf('ratio',rat_vars)
            else:
                rat_vars=self.pr.prepare_ratio(i,j,val,isRow)
                result = self.get_likelihood('ratio', rat_vars)
            return result
        
        if n == 1: 
            row_rat_score = [0]
        else:
            data_array = t.values
            high_rat_pairs, high_rat_vals = compute_high_ratio_pairs(data_array, tau=self.tau_rat)

            if len(high_rat_pairs) == 0:
                row_rat_score = [0]
            else:
                high_rat_pairs = np.atleast_2d(high_rat_pairs)
                likelihoods = np.array([get_ratio_likelihood(a, b, c, isRow=True) for (a, b), c in zip(high_rat_pairs, high_rat_vals)])
                row_mins = np.array([l * r for l, r in zip(likelihoods, high_rat_vals)])
                
                row_rat_score = get_inverese_ratio(row_mins)/(n * (n - 1) / 2)
            
        if m == 1: 
            col_rat_score = [0]
        else:
            data_array = t.T.values
            high_rat_pairs, high_rat_vals = compute_high_ratio_pairs(data_array, tau=self.tau_rat)

            if len(high_rat_pairs) == 0:
                col_rat_score = [0]
            else:
                high_rat_pairs = np.atleast_2d(high_rat_pairs)
                likelihoods = np.array([get_ratio_likelihood(a, b, c, isRow=False) for (a, b), c in zip(high_rat_pairs, high_rat_vals)])
                col_mins = np.array([l * r for l, r in zip(likelihoods, high_rat_vals)])
                
                col_rat_score = get_inverese_ratio(col_mins)/(m * (m - 1) / 2)
            
        ratio_score = max(row_rat_score, col_rat_score)
        
        return ratio_score

    def calculate_surprising(self, t, threshold=4.0):
        n, m = t.shape
        
        row_sur_score, col_sur_score = None, None
        
        def compute_outlier(arr, threshold):
            mean = np.nanmean(arr, axis=1, keepdims=True) 
            std = np.nanstd(arr, axis=1, keepdims=True) 
            deviation = np.abs(arr - mean)
            
            mask = np.zeros_like(arr, dtype=bool)
            valid_std = std != 0  
            valid_std = np.broadcast_to(valid_std, arr.shape)
            
            deviation_mask = deviation >= std * threshold
            mask = deviation_mask & valid_std            
            
            return mask

        def get_outlier_likelihood(i, j, isRow):
            if self.clf_trd:
                outlier_vars=self.pr.prepare_outlier_clf(i,j,isRow=isRow)
                result = self.get_likelihood_clf('outlier',outlier_vars)
            else:
                outlier_vars=self.pr.prepare_outlier(i,j, isRow)
                result = self.get_likelihood('outlier', outlier_vars)
            return result
        
        if m == 1: 
            row_sur_score = 0.0
        else:
            data_array = t.values
            row_outlier_mask = compute_outlier(data_array, threshold)
            row_scores = []
            for i in range(n):
                outlier_cols = np.where(row_outlier_mask[i])[0]
                if len(outlier_cols) > 0:
                    likelihood_sum = 0.0
                    for j in outlier_cols:
                        likelihood_sum += get_outlier_likelihood(i, j, True)
                    score = 1.0 - (likelihood_sum / (len(outlier_cols) + 1))
                else:
                    score = 0.0
                row_scores.append(score)

            row_sur_score = sum(row_scores) / n
            
        if n == 1: 
            col_sur_score = 0.0
        else:
            data_array = t.T.values
            col_outlier_mask = compute_outlier(data_array, threshold) 
            col_scores = []
            for j in range(m):
                outlier_rows = np.where(col_outlier_mask[j])[0]
                if len(outlier_rows) > 0:
                    likelihood_sum = 0.0
                    for i in outlier_rows:
                        likelihood_sum += get_outlier_likelihood(j, i, isRow=False)
                    score = 1.0 - (likelihood_sum / (len(outlier_rows) + 1))
                else:
                    score = 0.0
                col_scores.append(score)

            col_sur_score = sum(col_scores) / m
                    
        surprise_score = max(row_sur_score, col_sur_score)
        
        return surprise_score