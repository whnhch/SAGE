import json
import random
from typing import List, Dict, Any, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from src.prompt_learning.train_words import *
from sklearn.neural_network import MLPClassifier

def run_scaling_experiment(filename:str, configs: List[Dict[str, Any]], sizes: Union[int, List[int]]):
    if isinstance(sizes, int):
        sizes = [sizes]

    results = []

    for n in sizes:
        sampled_configs = sample_configs(configs, n)
        X, y = convert_configs_to_features_and_labels(sampled_configs)
        clf, train_acc, test_acc, report_dict = train_decision_tree_classifier(X, y)
        
        df_report = pd.DataFrame(report_dict).transpose()
        df_report.to_csv(f"result/{filename}_{n}_decisiontree.csv")
        
        results.append((n, train_acc, test_acc))
        if n == sizes[-1]:  # safe since sizes is now always a list
            joblib.dump(clf, f"model/{filename}_{n}_decision_tree.pkl")

    return results

def run_outlier_scaling_experiment(filename:str, configs: List[Dict[str, Any]], sizes: Union[int, List[int]]):
    if isinstance(sizes, int):
        sizes = [sizes]

    results = []

    for n in sizes:
        sampled_configs = sample_configs(configs, n)
        X, y = convert_outlier_configs_to_features_and_labels(sampled_configs)
        clf, train_acc, test_acc, report_dict = train_outlier_decision_tree_classifier(X, y)
        
        df_report = pd.DataFrame(report_dict).transpose()
        df_report.to_csv(f"result/{filename}_{n}_outlier_decisiontree.csv")
        
        results.append((n, train_acc, test_acc))
        if n == sizes[-1]:  # safe since sizes is now always a list
            joblib.dump(clf, f"model/{filename}_{n}_outlier_decision_tree.pkl")

    return results

def extract_trained_values_from_clf(clf, fields):
    trained_values = {}

    col_transformer = clf.named_steps['columntransformer']
    ohe = col_transformer.named_transformers_['cat']

    for field, cats in zip(fields, ohe.categories_):
        trained_values[field] = list(cats)

    return trained_values

def run_fasttext_experiment(configs: List[Dict[str, Any]], sizes: Union[int, List[int]], ft:KeyedVectors):
    if isinstance(sizes, int):
        sizes = [sizes]

    results = []

    for n in sizes:
        sampled_configs = sample_configs(configs, n)
        
        X, y = convert_configs_to_features_and_labels(sampled_configs)
        clf, X_test, y_test, train_acc, test_ignore_test = train_only_decision_tree_classifier(X, y)
        
        fields = ["index_attr", "index_values", "column_attr", "value_attr", "aggfunc", "insight_type", "bucket"]
        trained_values = extract_trained_values_from_clf(clf, fields)

        preds = predict_with_fasttext_normalization(clf, X_test, trained_values, fields, ft)
        test_acc = classification_report(y_test, preds)
        print(classification_report(y_test, preds))

        results.append((n, train_acc, test_ignore_test, test_acc))
        if n == sizes[-1]:  # safe since sizes is now always a list
            joblib.dump(clf, "decision_tree.pkl")

    return results

def run_mlp_experiment(configs: List[Dict[str, Any]], sizes: Union[int, List[int]], ft:KeyedVectors):
    if isinstance(sizes, int):
        sizes = [sizes]

    results = []

    for n in sizes:
        sampled_configs = sample_configs(configs, n)
        fields = ["index_attr", "index_values", "column_attr", "value_attr", "aggfunc", "insight_type", "bucket"]
        
        X, y = convert_configs_to_features_and_labels(sampled_configs)
        embedder = FastTextEmbedder(ft, fields)
        X_embedded = embedder.transform(X)
    
        mlp, train_acc, test_acc = train_mlp_with_fasttext(X_embedded, y)
        results.append((n, train_acc, test_acc))
        
        if n == sizes[-1]:  # safe since sizes is now always a list
            joblib.dump(mlp, "mlp.pkl")

    return results

def load_configs(filename: str) -> List[Dict[str, Any]]:
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def sample_configs(configs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return random.sample(configs, n)

def convert_configs_to_features_and_labels(configs: List[Dict[str, Any]]):
    X = []
    y = []

    for cfg in configs:
        label = cfg['label']
        feature = {
            'index_attr': cfg['index_attr'],
            'index_values': '|'.join(str(v) for v in sorted(cfg['index_values'])),
            'value_attr': cfg['value_attr'],
            'aggfunc': cfg['aggfunc'],
            'insight_type': cfg['insight_type'],
            'column_attr': cfg['column_attr'],
            'bucket': cfg['bucket'],
        }
        X.append(feature)
        y.append(label)
    
    return X, y

def convert_outlier_configs_to_features_and_labels(configs: List[Dict[str, Any]]):
    X = []
    y = []

    for cfg in configs:
        label = cfg['label']
        feature = {
            'index_attr': cfg['index_attr'],
            'index_value': str(cfg['index_value']),
            'value_attr': cfg['value_attr'],
            'aggfunc': cfg['aggfunc'],
            'column_attr': cfg['column_attr'],
            'column_value': str(cfg['column_value']),
        }
        X.append(feature)
        y.append(label)
    
    return X, y

def train_mlp_with_fasttext(X: List[Dict[str, Any]], y: List[str]):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)

    # Evaluate
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    return mlp, train_acc, test_acc

def train_decision_tree_classifier(X: List[Dict[str, Any]], y: List[str], max_depth=15):
    categorical_features = ['index_attr', 'index_values', 'column_attr', 'value_attr', 'aggfunc', 'insight_type', 'bucket']
    df_X = pd.DataFrame(X)  

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    clf = make_pipeline(preprocessor, DecisionTreeClassifier(max_depth=15, random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_dict["overall_train_accuracy"] = {
        "precision": train_acc,
        "recall": train_acc,
        "f1-score": train_acc,
        "support": len(y_train)
    }
    return clf, train_acc, test_acc, report_dict


def train_outlier_decision_tree_classifier(X: List[Dict[str, Any]], y: List[str], max_depth=15):
    categorical_features = ['index_attr', 'index_value', 'column_attr', 'column_value', 'value_attr', 'aggfunc']
    df_X = pd.DataFrame(X)  

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    clf = make_pipeline(preprocessor, DecisionTreeClassifier(max_depth=15, random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_dict["overall_train_accuracy"] = {
        "precision": train_acc,
        "recall": train_acc,
        "f1-score": train_acc,
        "support": len(y_train)
    }
    return clf, train_acc, test_acc, report_dict


def train_only_decision_tree_classifier(X: list, y: list, max_depth=15):
    categorical_features = ['index_attr', 'index_values', 'column_attr', 'value_attr', 'aggfunc', 'insight_type', 'bucket']
    df_X = pd.DataFrame(X)

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    clf = make_pipeline(preprocessor, DecisionTreeClassifier(max_depth=max_depth, random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    return clf, X_test, y_test, train_acc, test_acc

def fasttext_closest_match(token: str, candidates: List[str], model: KeyedVectors):
    def get_vec(t):
        try:
            return model[t]
        except KeyError:
            return np.zeros(model.vector_size)

    token_vec = get_vec(token)
    if not np.any(token_vec):
        return candidates[0] if candidates else token

    sims = []
    for cand in candidates:
        cand_vec = get_vec(cand)
        if not np.any(cand_vec): continue
        sim = np.dot(token_vec, cand_vec) / (np.linalg.norm(token_vec) * np.linalg.norm(cand_vec) + 1e-8)
        sims.append((sim, cand))

    if not sims:
        return candidates[0] if candidates else token

    return max(sims)[1]

def normalize_example_with_fasttext(example: Dict[str, Any], trained_values: Dict[str, List[str]], fields: List[str], model: KeyedVectors):
    normalized = {}

    for field in fields:
        val = str(example.get(field, ""))
        candidates = trained_values.get(field, [])

        if val in candidates:
            normalized[field] = val
        else:
            normalized[field] = fasttext_closest_match(val, candidates, model)

    return normalized

def predict_with_fasttext_normalization(clf, X_test, trained_values, fields, ft_model):
    normed_rows = []
    for row in X_test.to_dict(orient="records"):
        normed_row = normalize_example_with_fasttext(row, trained_values, fields, ft_model)
        normed_rows.append(normed_row)

    df_normed = pd.DataFrame(normed_rows)
    preds = clf.predict(df_normed)
    return preds

def plot_scaling_results(results: List[Tuple[int, float, float]]):
    sizes = [r[0] for r in results]
    train_accs = [r[1] for r in results]
    test_accs = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_accs, marker='o', label='Train Accuracy')
    plt.plot(sizes, test_accs, marker='s', label='Test Accuracy')
    plt.xlabel('Sample Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sample Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def plot_fasttext_scaling_results(results: List[Tuple[int, float, float, float]]):
    sizes = [r[0] for r in results]
    train_accs = [r[1] for r in results]
    test_ignore_accs = [r[2] for r in results]
    test_accs = [r[3] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_accs, marker='o', label='Train Accuracy')
    plt.plot(sizes, test_accs, marker='s', label='Test Ingore Accuracy')
    plt.plot(sizes, test_accs, marker='x', label='Test Accuracy')
    plt.xlabel('Sample Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sample Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def run_pipeline(json_path: str, filename:str, sizes: Union[int, List[int]], ft=None):
    configs = load_configs(json_path)
    if ft: 
        results = run_fasttext_experiment(configs, sizes, ft)
        plot_fasttext_scaling_results(results)
    else: 
        results = run_scaling_experiment(filename, configs, sizes)
        plot_scaling_results(results)

def run_outlier_pipeline(json_path: str, filename:str, sizes: Union[int, List[int]]):
    configs = load_configs(json_path)
    results = run_outlier_scaling_experiment(filename, configs, sizes)
    plot_scaling_results(results)
        
def run_mlp_pipeline(json_path: str, sizes: Union[int, List[int]], ft=None):
    configs = load_configs(json_path)
    results = run_mlp_experiment(configs, sizes, ft)
    plot_scaling_results(results)