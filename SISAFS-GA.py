# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:04:10 2026

@author: Kamyar
"""

################ SISAFS ########################

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_DIR = r"D:\DataS Projects\DataS\dataset2"
DATASETS = {
    
    "GLI": "GLI.csv", 
    # "SMK": "SMK.csv",
    # "Covid_19": "Covid.csv", 
    # "Leukemia": "Leukemia_3c.xlsx", 
    # "Prostate": "Prostate.xlsx",
    # "ALLAML": "ALLAML.csv",
    # "nci9": "nci9.csv",
    # "Tox_171": "TOX_171.csv",
    
}
TEST_SIZE, SEED = 0.10, 123

# --- 1. Dataset Loader & Global Normalization ---
def load_and_normalize_data(name, source):
    file_path = os.path.join(DATA_DIR, source)
    if source.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=0)
    elif source.endswith('.csv'):
        df = pd.read_csv(file_path, header=0)
    else:
        raise ValueError(f"Unsupported file format for {source}")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if y.dtype == object or isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)
        
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    
    # Strict Train/Test Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Normalize once: fit on X_train, transform both X_train and X_test to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


# --- Helper: Process a Single Sample for DISAFS (Optimized for Parallel Execution) ---
def process_single_sample(i, X_train, y_train, classes):
    n_samples, n_features = X_train.shape
    x_i = X_train[i]
    y_i = y_train[i]
    
    mask_i = np.ones(n_samples, dtype=bool)
    mask_i[i] = False
    X_loo = X_train[mask_i]
    y_loo = y_train[mask_i]
    
    distances = np.abs(X_loo - x_i)
    sample_retained = np.zeros(n_features, dtype=bool)
    
    for f in range(n_features):
        d_f = distances[:, f]
        class_aucs = {}
        for c in classes:
            y_binary = (y_loo == c).astype(int)
            n_pos = np.sum(y_binary == 1)
            n_neg = np.sum(y_binary == 0)
            
            if n_pos == 0 or n_neg == 0:
                class_aucs[c] = 0.5
                continue
            
            sample_weights = np.zeros_like(y_binary, dtype=float)
            sample_weights[y_binary == 1] = 1.0 / (2.0 * n_pos)
            sample_weights[y_binary == 0] = 1.0 / (2.0 * n_neg)
            
            try:
                auc_c = roc_auc_score(y_binary, d_f, sample_weight=sample_weights)
                class_aucs[c] = auc_c
            except ValueError:
                class_aucs[c] = 0.5
        
        predicted_class = min(class_aucs, key=class_aucs.get)
        if predicted_class == y_i:
            sample_retained[f] = True
            
    return i, sample_retained


# --- 2. DISAFS & SISAFS Pipeline Engine ---
def run_disafs_and_sisafs(X_train, y_train):
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    
    print(f"  Running DISAFS across {n_samples} training samples and {n_features} features (using multi-core parallelization)...")
    
    results_parallel = Parallel(n_jobs=-1)(
        delayed(process_single_sample)(i, X_train, y_train, classes) for i in range(n_samples)
    )
    
    retained_matrix = np.zeros((n_samples, n_features), dtype=bool)
    for i, sample_retained in results_parallel:
        retained_matrix[i] = sample_retained
                
    per_sample_counts = np.sum(retained_matrix, axis=1)
    avg_per_sample = np.mean(per_sample_counts)
    min_per_sample = np.min(per_sample_counts)
    max_per_sample = np.max(per_sample_counts)
    
    correct_predictions = 0
    for i in range(n_samples):
        selected_feats = np.where(retained_matrix[i] == True)[0]
        if len(selected_feats) == 0:
            selected_feats = [0]
            
        mask_i = np.ones(n_samples, dtype=bool)
        mask_i[i] = False
        X_loo = X_train[mask_i][:, selected_feats]
        y_loo = y_train[mask_i]
        x_test_sample = X_train[i, selected_feats].reshape(1, -1)
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_loo, y_loo)
        pred = clf.predict(x_test_sample)[0]
        if pred == y_train[i]:
            correct_predictions += 1
            
    disafs_train_acc = correct_predictions / n_samples
    
    retention_counts = np.sum(retained_matrix, axis=0)
    valid_features = np.where(retention_counts > 0)[0]
    sorted_sisafs_features = valid_features[np.argsort(retention_counts[valid_features])[::-1]]
    total_aggregated_count = len(sorted_sisafs_features)
    
    stats = {
        "avg_retained": round(avg_per_sample, 2),
        "min_retained": int(min_per_sample),
        "max_retained": int(max_per_sample),
        "total_aggregated_count": total_aggregated_count,
        "disafs_train_acc": disafs_train_acc
    }
    
    return sorted_sisafs_features, stats


# --- 3. Helper: Evaluate Classifier Metrics ---
def evaluate_metrics(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_metrics = {
        "Acc": accuracy_score(y_train, y_train_pred),
        "Prec": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        "Rec": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }
    
    test_metrics = {
        "Acc": accuracy_score(y_test, y_test_pred),
        "Prec": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "Rec": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    }
    
    return train_metrics, test_metrics


# --- 4. Main Execution Pipeline ---
def main():
    results = []
    dataset_topk_records = {}  # Dictionary to record top1 to top100 test accuracies per dataset for later use
    
    for dataset_name, source in DATASETS.items():
        print(f"\n{'='*55}\nProcessing Dataset: {dataset_name}\n{'='*55}")
        try:
            X_train_scaled, X_test_scaled, y_train, y_test = load_and_normalize_data(dataset_name, source)
        except Exception as e:
            print(f"Could not load/normalize {dataset_name}: {e}")
            continue
            
        print(f"Train shape -> Samples: {X_train_scaled.shape[0]}, Features: {X_train_scaled.shape[1]}")
        
        train_orig, test_orig = evaluate_metrics(X_train_scaled, y_train, X_test_scaled, y_test)
        
        start_time = time.time()
        sorted_sisafs_features, disafs_stats = run_disafs_and_sisafs(X_train_scaled, y_train)
        fs_duration = time.time() - start_time
        
        print(f"  Feature selection completed in {fs_duration:.2f}s")
        print(f"  Per-sample retained features (DISAFS) -> Avg: {disafs_stats['avg_retained']}, Min: {disafs_stats['min_retained']}, Max: {disafs_stats['max_retained']}")
        print(f"  Total Aggregated Features (Freq > 0): {disafs_stats['total_aggregated_count']}")
        print(f"  DISAFS Train Accuracy: {disafs_stats['disafs_train_acc']:.2f}")
        
        print(f"\n--- Results for {dataset_name} ---")
        print(f"  [DISAFS] Train Accuracy: {disafs_stats['disafs_train_acc']:.2f}")
        print(f"  [All Features] Test Accuracy: {test_orig['Acc']:.2f} | Prec: {test_orig['Prec']:.2f} | Rec: {test_orig['Rec']:.2f} | F1: {test_orig['F1']:.2f}")
        
        # --- Evaluate SISAFS across Top-10 to Top-100 (Steps of 10) ---
        print(f"\n  [SISAFS Test Performance Across Top-K Subsets (10 to 100)]")
        print(f"  {'Top-K':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print(f"  {'-'*55}")
        
        dataset_topk_results = []
        for k in range(10, 101, 10):
            actual_k = min(k, len(sorted_sisafs_features))
            if actual_k == 0:
                continue
                
            sisafs_subset = sorted_sisafs_features[:actual_k]
            X_train_sub = X_train_scaled[:, sisafs_subset]
            X_test_sub = X_test_scaled[:, sisafs_subset]
            
            _, test_sisafs_sub = evaluate_metrics(X_train_sub, y_train, X_test_sub, y_test)
            
            print(f"  Top-{actual_k:<5} | {test_sisafs_sub['Acc']:<10.2f} | {test_sisafs_sub['Prec']:<10.2f} | {test_sisafs_sub['Rec']:<10.2f} | {test_sisafs_sub['F1']:<10.2f}")
            
            dataset_topk_results.append({
                "Dataset": dataset_name,
                "Top_K": actual_k,
                "SISAFS_Test_Prec": round(test_sisafs_sub["Prec"], 2),
                "SISAFS_Test_Rec": round(test_sisafs_sub["Rec"], 2),
                "SISAFS_Test_F1": round(test_sisafs_sub["F1"], 2),
            })
            
        results.extend(dataset_topk_results)
        
        # --- Calculate and Record Top-1 to Top-100 Test Accuracies for Later Use ---
        max_possible_k = min(100, len(sorted_sisafs_features))
        top1_to_100_accuracies = []
        
        for k in range(1, max_possible_k + 1):
            sisafs_subset = sorted_sisafs_features[:k]
            X_train_sub = X_train_scaled[:, sisafs_subset]
            X_test_sub = X_test_scaled[:, sisafs_subset]
            _, test_sub = evaluate_metrics(X_train_sub, y_train, X_test_sub, y_test)
            top1_to_100_accuracies.append(test_sub["Acc"])
            
        dataset_topk_records[dataset_name] = top1_to_100_accuracies
        avg_top1_100_acc = np.mean(top1_to_100_accuracies) if top1_to_100_accuracies else 0.0
        
        print(f"\n  Average Test Accuracy (Top-1 to Top-{max_possible_k} features with SISAFS): {avg_top1_100_acc:.2f}")
        
        # =========================================================
        # --- SAVING BLOCK (Integrated inside main loop) ---
        # =========================================================
        save_dir = r"D:\SISAFS\NORMALIZED\GLI"
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "disafs_stats.pkl"), "wb") as f:
            pickle.dump(disafs_stats, f)

        with open(os.path.join(save_dir, "sorted_sisafs_features.pkl"), "wb") as f:
            pickle.dump(sorted_sisafs_features, f)

        with open(os.path.join(save_dir, "topk_records.pkl"), "wb") as f:
            pickle.dump(top1_to_100_accuracies, f)

        print(f"\n[Saved] All files successfully written to {save_dir}")
        
    print(f"\n{'*'*20} FINAL COMPARATIVE SUMMARY TABLE (ALL DATASETS & TOP-K) {'*'*20}")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    return dataset_topk_records

if __name__ == "__main__":
    main()





###################### Wrapper Ensemble GA ###########################
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_DIR = r"D:\DataS Projects\DataS\dataset2"
DATASETS = {
    
    "GLI": "GLI.csv", 
    # "SMK": "SMK.csv",
    # "Covid_19": "Covid.csv", 
    # "Leukemia": "Leukemia_3c.xlsx", 
    # "Prostate": "Prostate.xlsx",
    # "ALLAML": "ALLAML.csv",
    # "nci9": "nci9.csv",
    # "Tox_171": "TOX_171.csv",
    
}
TEST_SIZE, SEED = 0.10, 123
GA_SEED = 42
FEATURE_BUDGET = 100
ALPHA_PENALTY = 0.01  # Parsimony penalty weight to favor smaller subsets while prioritizing accuracy
B_RESAMPLES = 100      # Number of bootstrap resamples for ensemble feature selection

# Automatically pick the active dataset from the DATASETS dictionary
ACTIVE_DATASET_NAME = list(DATASETS.keys())[0]
ACTIVE_DATASET_FILE = DATASETS[ACTIVE_DATASET_NAME]

# --- Dataset Loader ---
def load_data(name, source):
    file_path = os.path.join(DATA_DIR, source)
    df = pd.read_excel(file_path, header=0) if source.endswith('.xlsx') else pd.read_csv(file_path, header=0)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    if y.dtype == object or isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)
    return np.array(X, dtype=float), np.array(y, dtype=int)

# --- 1. Load Saved Top-K Features ---
save_dir = rf"D:\SISAFS\NORMALIZED\{ACTIVE_DATASET_NAME}"
with open(os.path.join(save_dir, "sorted_sisafs_features.pkl"), "rb") as f:
    sorted_sisafs_features = pickle.load(f)

top_features = sorted_sisafs_features[:100] if isinstance(sorted_sisafs_features, (list, np.ndarray)) else list(sorted_sisafs_features.keys())[:100]
# top_features = sorted_sisafs_features[-100:] if isinstance(sorted_sisafs_features, (list, np.ndarray)) else list(sorted_sisafs_features.keys())[-100:]


# --- 2. Load Dataset & Strict Train/Test Split ---
X, y = load_data(ACTIVE_DATASET_NAME, ACTIVE_DATASET_FILE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
)

# --- Data Normalization (Z-score standardization) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Baseline Evaluation: All Features on Test Set ---
dt_baseline = DecisionTreeClassifier(random_state=GA_SEED)
dt_baseline.fit(X_train, y_train)
y_pred_baseline = dt_baseline.predict(X_test)
baseline_test_acc = accuracy_score(y_test, y_pred_baseline)
print(f"Dataset: {ACTIVE_DATASET_NAME}")
print(f"Baseline Test Accuracy (All {X.shape[1]} Features): {baseline_test_acc:.2f}")


# --- 3. Budgeted GA Feature Selector with Fitness History Tracking ---
class BudgetedGAFeatureSelector:
    def __init__(self, feature_pool, feature_budget=10, pop_size=30, generations=100, patience=20, alpha=ALPHA_PENALTY, seed=GA_SEED):
        self.feature_pool = np.array(feature_pool)
        self.n_features = len(self.feature_pool)
        self.feature_budget = feature_budget
        self.pop_size = pop_size
        self.generations = generations
        self.patience = patience
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.nfe = 0  

    def fit(self, X_train, y_train):
        # Initialize fresh tracking lists locally per run instance
        best_fitness_history = []
        avg_fitness_history = []
        self.nfe = 0

        population = np.zeros((self.pop_size, self.n_features), dtype=int)
        for i in range(self.pop_size):
            k = self.rng.integers(1, self.feature_budget + 1)
            chosen_idx = self.rng.choice(self.n_features, size=k, replace=False)
            population[i, chosen_idx] = 1

        best_overall_fitness = -float('inf')
        best_chromosome = population[0].copy()
        patience_counter = 0

        for gen in range(self.generations):
            fitnesses = []
            for ind in population:
                selected_mask = (ind == 1)
                selected_idx = np.where(selected_mask)[0]

                if len(selected_idx) == 0 or len(selected_idx) > self.feature_budget:
                    fitnesses.append(-999.0)
                    continue

                actual_features = self.feature_pool[selected_idx]

                dt = DecisionTreeClassifier(random_state=GA_SEED)
                cv_scores = cross_val_score(dt, X_train[:, actual_features], y_train, cv=5, scoring='accuracy')
                mean_cv_acc = np.mean(cv_scores)

                self.nfe += 1

                size_ratio = len(selected_idx) / self.feature_budget
                fitness_val = mean_cv_acc - (self.alpha * size_ratio)
                fitnesses.append(fitness_val)

            fitnesses = np.array(fitnesses)
            valid_fitnesses = fitnesses[fitnesses != -999.0]
            current_avg_fitness = np.mean(valid_fitnesses) if len(valid_fitnesses) > 0 else 0.0

            gen_best_idx = np.argmax(fitnesses)
            current_gen_best_fitness = fitnesses[gen_best_idx]

            if current_gen_best_fitness > best_overall_fitness:
                best_overall_fitness = current_gen_best_fitness
                best_chromosome = population[gen_best_idx].copy()
                patience_counter = 0
            else:
                patience_counter += 1

            best_fitness_history.append(best_overall_fitness)
            avg_fitness_history.append(current_avg_fitness)

            if patience_counter >= self.patience:
                break

            min_fit = np.min(fitnesses)
            shifted_fitnesses = fitnesses - min_fit if min_fit < 0 else fitnesses.copy()
            fit_sum = np.sum(shifted_fitnesses)
            probs = np.ones(self.pop_size) / self.pop_size if fit_sum == 0 else shifted_fitnesses / fit_sum

            selected_indices = self.rng.choice(self.pop_size, size=self.pop_size, p=probs, replace=True)
            new_population = population[selected_indices].copy()

            for i in range(0, self.pop_size - 1, 2):
                if self.rng.random() < 0.8:
                    mask = self.rng.integers(0, 2, size=self.n_features).astype(bool)
                    child1 = np.where(mask, new_population[i], new_population[i+1])
                    child2 = np.where(mask, new_population[i+1], new_population[i])
                    new_population[i] = child1
                    new_population[i+1] = child2

            for i in range(self.pop_size):
                for j in range(self.n_features):
                    if self.rng.random() < (1.0 / self.n_features):
                        new_population[i, j] = 1 - new_population[i, j]

            for ind in new_population:
                active = np.where(ind == 1)[0]
                if len(active) == 0:
                    ind[self.rng.integers(0, self.n_features)] = 1
                elif len(active) > self.feature_budget:
                    to_drop = self.rng.choice(active, size=len(active) - self.feature_budget, replace=False)
                    ind[to_drop] = 0

            new_population[0] = best_chromosome.copy()
            population = new_population

        best_mask = (best_chromosome == 1)
        return self.feature_pool[best_mask], best_overall_fitness, self.nfe, best_fitness_history, avg_fitness_history


# --- 4. Run GA B Times on Bootstrap Resamples ---
feature_counts = {feat: 0 for feat in top_features}
total_nfe = 0
nfe_per_run = []                  # List to save individual NFE values run by run
features_per_run = []             # List to save the number of features selected per run

all_best_histories = []
all_avg_histories = []

print(f"\n[Ensemble of Wrappers Feature Selection: Running GA across B={B_RESAMPLES} Bootstrap Resamples]")
rng_bootstrap = np.random.default_rng(SEED)

for b in range(B_RESAMPLES):
    resample_indices = rng_bootstrap.choice(len(X_train), size=len(X_train), replace=True)
    X_resample = X_train[resample_indices]
    y_resample = y_train[resample_indices]

    ga_selector = BudgetedGAFeatureSelector(
        feature_pool=top_features,
        feature_budget=FEATURE_BUDGET,
        pop_size=30,
        generations=100,
        patience=20,
        alpha=ALPHA_PENALTY,
        seed=GA_SEED + b
    )

    run_features, run_fitness, run_nfe, best_hist, avg_hist = ga_selector.fit(X_resample, y_resample)
    
    # Store histories and metric tracking arrays per run
    all_best_histories.append(best_hist)
    all_avg_histories.append(avg_hist)
    
    nfe_per_run.append(run_nfe)
    total_nfe += run_nfe

    num_selected = len(run_features)
    features_per_run.append(num_selected)

    for f in run_features:
        feature_counts[f] += 1
    
    clf_resample = DecisionTreeClassifier(random_state=GA_SEED)
    clf_resample.fit(X_resample[:, run_features], y_resample)
    train_acc_resample = accuracy_score(y_resample, clf_resample.predict(X_resample[:, run_features]))

    print(f"  Resample {b+1}/{B_RESAMPLES} completed | Best Fitness: {run_fitness:.4f} | Train Acc: {train_acc_resample:.2f} | Selected {num_selected} features | NFE: {run_nfe}")

# Print NFE and Feature Count Summary Statistics across the runs
print(f"\n[NFE Summary across {B_RESAMPLES} Runs]")
print(f"  Total NFE (Sum): {total_nfe}")
print(f"  Mean NFE per Run: {np.mean(nfe_per_run):.2f} ± {np.std(nfe_per_run):.2f}")
print(f"  Min NFE: {np.min(nfe_per_run)} | Max NFE: {np.max(nfe_per_run)}")

print(f"\n[Selected Features Count Summary across {B_RESAMPLES} Runs]")
print(f"  Mean Features Selected: {np.mean(features_per_run):.2f} ± {np.std(features_per_run):.2f}")
print(f"  Min Features: {np.min(features_per_run)} | Max Features: {np.max(features_per_run)}")

# Sort features by selection frequency descending
sorted_features_by_freq = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)


# --- 5. Cumulative Top-K Test Accuracy Evaluation (Top 1 to Top 100) ---
top_k_ranks = list(range(1, 101))
top_k_test_accuracies = []

print("\n[Cumulative Top-K Feature Evaluation]")
for k in top_k_ranks:
    selected_k_features = np.array([feat for feat, count in sorted_features_by_freq[:k]])
    
    X_train_k = X_train[:, selected_k_features]
    X_test_k = X_test[:, selected_k_features]

    clf = DecisionTreeClassifier(random_state=GA_SEED)
    clf.fit(X_train_k, y_train)
    y_pred_k = clf.predict(X_test_k)
    acc_k = accuracy_score(y_test, y_pred_k)
    top_k_test_accuracies.append(acc_k)
    print(f"  Top {k} Features Test Accuracy: {acc_k:.2f}")
    
    
# --- 6. Separate Visualizations & Plotting ---

# --- Figure 1: Feature Selection Frequencies Histogram Across 100 Runs for top 10 features (Feature Names on X-Axis) ---
from matplotlib.ticker import MaxNLocator
plt.figure(figsize=(10, 5))
features_list = [str(feat) for feat, count in sorted_features_by_freq[0:10]]
counts_list = [count for feat, count in sorted_features_by_freq[0:10]]

plt.bar(features_list, counts_list, color='skyblue', edgecolor='black')
plt.title("TOX", fontsize = 38)
plt.xlabel("Feature Index",fontsize = 32)
plt.ylabel("Frequency", fontsize = 32)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# Increased tick label sizes
plt.xticks(rotation = 45, fontsize=24)
plt.yticks(fontsize=24)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# # --- Figure 2: Top-K Test Accuracies vs. Baseline Accuracy ---


from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

# Select Resample 99 (0-indexed as index 98)
run_idx = 3

plt.figure(figsize=(8, 5))
plt.plot(all_best_histories[run_idx], label='Best Fitness', color='green', linewidth=2)
plt.plot(all_avg_histories[run_idx], label='Average Fitness', color='orange', linestyle='--', linewidth=2)

plt.title("TOX", fontsize=34)
plt.xlabel("Iteration", fontsize=28)
plt.ylabel("Fitness", fontsize=28)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=18, loc="lower right")
# plt.ylim(0.76,0.965)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#################################################3
# --- Figure: Cumulative Top-K Test Accuracy Evaluation (Top 1 to Top 100) ---
plt.figure(figsize=(8, 5))
plt.plot(top_k_ranks, top_k_test_accuracies, marker='o', linestyle='-', color='b', label='Top-100 Features')

# Baseline and Average lines
plt.axhline(y=baseline_test_acc, color='r', linestyle='--', label=f'Baseline ({baseline_test_acc:.2f})')
avg_top_k_acc = np.mean(top_k_test_accuracies)
plt.axhline(y=avg_top_k_acc, color='g', linestyle='-.', label=f'Avg Top-100 ({avg_top_k_acc:.2f})')

plt.title("TOX", fontsize=34)
plt.xlabel("Top Features", fontsize=28)
plt.ylabel("Test Accuracy", fontsize=28)

# Set x-ticks explicitly starting at 1, then from 10 up to 100 stepping by 10
custom_xticks = [1] + list(range(10, max(top_k_ranks) + 1, 10))
plt.xticks(custom_xticks, fontsize=20)

plt.yticks(fontsize=20)
plt.legend(loc="lower right", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.ylim(0.2, 0.92)
plt.show()




##############  Additional Plot Generation 
######################################
##########  Box Plot #################
######################################

import numpy as np
import matplotlib.pyplot as plt

# --- NFE Data Dictionary across 50 runs for each dataset ---
nfe_data = {
    # "Prostate": [1560, 2970, 1950, 1500, 630, 2310, 1260, 750, 1470, 2190, 990, 1020, 1590, 3000, 690, 660, 630, 1650, 1380, 960, 930, 720, 660, 1800, 810, 690, 1620, 3000, 1020, 1800, 1500, 1740, 900, 1200, 630, 630, 720, 870, 1110, 2130, 3000, 1800, 2160, 750, 660, 660, 810, 1260, 1560, 780, 3000, 990, 1050, 630, 780, 630, 2100, 660, 1380, 1020, 1410, 870, 660, 990, 1020, 810, 1200,1560, 690, 1710, 1740, 1680, 2850, 810, 990, 1560, 990, 630, 2310, 1080, 3000, 1890, 750, 1080, 1950, 1410, 690, 720, 2310, 630, 690, 750, 1560, 930, 3000, 1470, 1050, 630, 840, 1680],
    "Prostate":  [18, 30, 41, 42, 24, 32, 32, 52, 39, 35, 51, 31, 40, 31, 34, 35, 52, 54, 48, 48, 32, 36, 35, 38, 29, 48, 35, 36, 40, 36, 36, 42, 44, 38, 15, 18, 36, 22, 40, 36, 40, 36,27, 23, 22, 28, 24, 43, 29, 26, 37, 34, 22, 10, 31, 26, 45, 18, 30, 34, 47, 24, 17, 42, 30, 31, 38, 34, 48, 33, 40, 43, 37, 36, 49, 34, 40, 10, 29, 29, 32, 35, 47, 35, 31, 40, 40, 47, 41, 2, 36, 47, 38, 44, 30, 43, 41, 23, 38, 36],
    
    # "ALLAML": [630, 720, 1230, 630, 1260, 630, 660, 690, 630, 870, 630, 630, 1110, 1020, 720, 930, 630, 2640, 630, 900, 720, 630, 630, 630, 630, 660, 630, 630, 720, 780, 630, 750, 630, 660, 870, 1290, 1440, 840, 630, 630, 630, 1080, 960, 720, 1110, 630, 630, 660, 990, 630, 2760, 1380, 1890, 660, 2610, 840, 870, 1110, 690, 660, 1230, 630, 660, 720, 840, 990, 630, 720, 660, 990, 660, 630, 630, 660, 630, 660, 1020, 3000, 660, 630, 690, 840, 630, 2820, 810, 630, 690, 750, 630, 1170, 1170, 630, 630, 660, 660, 690, 660,1170, 1530, 1290],
    "ALLAML": [4, 16, 39, 32, 35, 13, 18, 13, 11, 15, 19, 4, 18, 36, 28, 14, 8, 33, 18, 38, 9, 16, 3, 17, 1, 22, 3, 4, 21, 28, 13, 16,3, 13, 10, 35, 34, 21, 8, 7, 15, 22, 10, 12, 25, 10, 19, 19, 17, 6, 30, 36, 30, 12, 37, 34, 16, 35, 13, 12, 18, 12, 17, 32, 35, 5, 1, 30, 25, 35, 17, 25, 33, 15, 21, 13, 37, 34, 3, 6, 31, 52, 12, 34, 28, 11, 29, 10, 28, 18, 27, 12, 4, 22, 13, 8, 29, 25, 32, 34],
    
    # "SMK": [2880,  1170,  1560,  2250,  1740,  2970, 2430, 1110, 1230, 1680, 1530, 1470, 630, 1140, 1890, 810, 1380, 1620, 2610, 1440, 1230, 1380, 1350, 840, 1590, 810, 3000, 1020, 1920, 1440, 1290, 2070, 1080, 2580, 1470, 1110, 1230, 1200, 870, 1680, 1560, 630, 1770, 2400, 1980, 1140, 690, 1260, 1290, 1260, 2970, 840, 1860, 900, 1200, 630, 990, 2880, 1440, 3000, 1620, 1380, 750, 1170, 1350, 960, 1410, 2370, 2430, 1080, 1800, 1380, 780, 1170, 1350, 1200, 1140, 1470, 960, 1590, 1680, 930, 1260, 1560, 1350, 840, 2190, 1590, 930, 630, 1620, 630, 2490, 630, 2610, 2610, 1410, 1110, 1740, 1530],
    "SMK": [44, 51, 54, 43, 44, 42, 34, 58, 56, 40, 52, 44, 7, 64, 52, 43, 36,51, 55, 30, 45, 38, 55, 42, 46, 4, 52, 46, 40, 46, 46, 39, 49, 38, 41, 41, 44, 30, 65, 51, 55, 26, 32, 28, 44, 43, 43, 50, 34, 55, 47, 44, 45, 45, 43, 6, 51, 43, 50, 37, 66, 38, 26, 36, 84, 40, 47, 40, 61, 63, 43, 52, 65, 35, 70, 61, 40, 49, 35, 43, 45, 30, 58, 49, 46, 54, 41, 46, 58, 23, 47, 54, 48, 10, 46, 36, 50, 52, 45, 50],
    
    # "GLI": [960,  1050,  2730,  1920,  1140,  1110,  1230,  720,  1590,  630,  630,  630,  1710,   900,  1260,  960,  990,  1410,  1500,  1770,  750,  660,  630,  1410,  990,  990,  960,  2970,  630,  630,  2610, 660,  1140,  1230,  720,  690,  1350,  1020,  1860,  630,  1170,  1230,  2070,  1140,  1140,  930,  750,  690,  1260,  2010,  630,  960,  660,  720,  1140,  840,  1740,  1770,  1170,  900,  1470,  930,  1770,  720,  630,  720,  660,  630,  960,  690,  690,  1080,  1290, 1050,  1350,  1320,  1200,  990,  1860,  720,  1080, 1110,  690,  960,  690, 960,  2430,  1170,  660,  1710,  690,  900,  2370,  2040,  900,  1200,  630,  1890,  660,  630],
    "GLI":  [7,  38, 27, 48, 39, 39, 32, 46, 39, 35, 43, 32, 34, 7, 38, 41, 54, 21, 37, 20, 26, 5, 36, 37, 54,14, 31, 28, 32, 24, 40, 2, 48, 21, 40, 27, 46, 27, 36, 7, 54, 35, 40, 23, 17, 38, 30, 16, 25, 47, 14, 38, 26, 29, 21, 37, 28, 29, 30, 27, 31, 44, 30, 30, 8, 24, 44, 16, 49, 13, 22, 33, 45, 46, 38, 42, 29, 57, 34, 17, 30, 45, 51, 54, 33, 50, 32, 30, 25, 37, 28, 49, 39, 41, 41, 29, 10, 51, 31, 28],
    
    # "TOX": [870, 1500, 1320, 690, 1260, 2130, 2010, 1980, 1590, 750, 1170, 1620, 1380, 1860, 720, 2010, 1410, 750, 1350, 900, 1290, 1440, 1110, 1230, 1320, 1620, 1620, 1890, 840, 1380, 1170, 2910, 1440, 930, 1440, 1170, 2010, 840, 1710, 750, 900, 1830, 1590, 1920, 1860, 870, 870, 780, 2100, 1500, 1920, 1770, 1410, 690, 990, 1050, 1440, 1350, 1260, 900, 2340, 1590, 1290, 1170, 1350, 840, 1380, 1860, 2040, 1380, 2160, 1590, 810, 660, 1200, 1650, 1350, 660, 1650, 1290, 1080, 1200, 1350, 1980, 1800, 990, 870, 840, 1770, 2160, 1440, 870, 2400, 1380, 2010, 1140, 900, 1440, 1590, 1800],
    "TOX": [40, 35, 61, 76, 37, 43, 47, 43, 71, 71, 56, 38, 31, 63, 42, 44, 45, 44, 48, 30, 45, 41, 50, 65, 51, 53, 46, 41, 40, 43, 61, 48, 42, 26, 50, 48, 41, 56, 57, 42, 61, 45, 42, 30, 40, 55, 48, 70, 33, 50, 55, 49, 40, 27, 50, 50, 48, 44, 49, 39, 50, 45, 28, 46, 40, 42, 53, 57, 48, 55, 47, 35, 18, 36, 52, 47, 38, 67, 32, 41, 54, 40, 53, 51, 49, 58, 56, 59, 43, 47, 53, 31, 48, 63, 42, 37, 50, 58, 45, 47],
    
    # "Leukemia": [900, 810, 870, 810, 1740, 990, 1440, 3000, 3000, 1800, 630, 630, 630, 3000, 690, 630, 630, 750, 1500, 2100, 660, 780, 3000, 720, 1980, 2280, 630, 630, 810, 2130, 1020, 660, 2070, 990, 1200, 1740, 1440, 720, 630, 630, 1500, 1230, 1500, 750, 630, 2010, 1530, 750, 690, 720, 630, 1080, 1650, 660, 630, 1440, 630, 2130, 1230, 660, 630, 870, 1080, 780, 660, 690, 930, 2640, 660, 1230, 1980, 1320, 660, 1320, 1110, 660, 990, 2460, 630, 1200, 750, 1440, 840, 960, 1800, 2280, 2250, 810, 750, 870, 630, 630, 630, 1410, 1410, 1440, 3000, 2310, 750, 1170],,
    "Leukemia": [17, 20, 40, 12, 31, 43, 22, 23, 36, 39, 43, 8, 6, 34, 40, 25, 5, 46, 33, 33, 31, 29, 32, 44, 35, 26, 9, 2, 35, 29, 31, 19, 32, 32, 39, 23, 39, 21, 41, 6, 27, 30, 14, 11, 19, 34, 45, 31, 21, 36, 22, 43, 36, 12, 16, 37, 8, 26, 32, 16, 20, 32, 28, 32, 37, 21, 26, 25, 31, 31, 40, 30, 43, 32, 40, 28, 40, 31, 4, 14, 9, 34, 35, 48, 39, 31, 41, 30, 37, 14, 31, 45, 8, 45, 34, 31, 26, 26, 33, 22],
    
    # "NCI": [2400, 3000, 1860, 1530, 1200, 1590, 1920, 1740, 1110, 1380, 1260, 1170, 1740, 1890, 1350, 1350, 1440, 1710, 1170, 1620, 1350, 2880,1020, 3000,1860, 1680, 1920, 1020, 1020, 1800, 840, 1200, 2280, 2580, 630, 1260, 1440, 1410, 2550, 920, 630, 1830, 1170, 750, 1470, 990, 930, 1140, 960, 1110, 3000, 1770, 720, 810, 2070, 2130, 1380, 1740,2040, 1140, 2400, 630, 1800, 630, 2550, 2010, 1110, 1800, 3000, 1770, 1410, 1110, 1500, 870, 1320, 1260, 1290, 1440, 1050, 1440, 1710, 690, 2520, 1830, 1560, 1440, 1530, 900, 1740, 1530, 1200, 720, 1740, 2340, 660, 1920, 2520,1650, 1050, 2040],
    "NCI":[43, 38, 47, 52, 51, 9, 47, 53, 65, 53, 59, 44, 38, 57, 47, 51, 54, 49, 37, 35, 37, 25, 40, 54, 51, 47, 47, 48, 43, 43, 49, 42, 39, 36, 76, 41, 50, 36, 44, 43, 19, 35, 37, 17, 34, 39, 47,38, 43, 50, 44, 55, 45, 49, 39, 45, 42, 24, 30, 34, 50, 17, 23, 7, 46, 43, 46, 42, 38, 44, 48, 19, 55, 49, 31, 33, 47, 57, 33, 45, 43, 27, 55, 41, 51, 47, 45, 44, 46, 49, 47, 50, 43, 50, 10, 41, 48, 37,55, 48],
    
    # "Covid_19": [660, 960, 1500, 1320, 1110, 960, 1260, 630, 1200, 630, 2250, 960, 1650, 780, 440, 1230, 660, 1230, 1620, 1620, 1470, 1920, 660, 780, 690, 1860, 1230, 1350, 2280, 930, 870, 720, 2520, 1170, 750, 930, 1110, 1500, 840, 1470, 660, 840, 1500, 2550, 1080, 930, 1800, 2220, 780, 960, 750, 1950, 1260, 1260, 720, 1290, 870, 1440, 1230, 900, 1260, 1590, 840, 660, 690, 1200, 1830, 1500, 1410, 1380, 1020, 1680, 630, 990, 1500, 1710, 1230, 630, 3000, 3000, 630, 930, 780, 1500, 660, 1320, 640, 960, 1830, 1260, 900, 2160, 2100, 750, 2250, 1110, 2370, 1980, 870, 840]
    "Covid_19":  [42, 42, 41, 46, 51, 41, 37, 14, 53, 36, 46, 48, 40, 63, 49, 52, 38, 57, 46, 41, 55, 49, 42, 51, 48, 39, 29, 46, 45, 42, 55, 54, 43, 35, 41, 39, 49, 43, 39, 37, 63, 38, 36, 47, 50, 23, 57, 50, 38, 63, 66, 41, 39, 49, 36, 61, 16, 45, 42, 40, 55, 52, 35, 96, 25, 41, 53, 43, 49, 49, 53, 49, 35, 45, 56, 32, 40, 25, 41, 42, 15, 54, 46, 44, 47, 43, 48, 56, 56, 43, 41, 48, 51, 47, 47, 36, 35, 43, 43, 48]
}

datasets = list(nfe_data.keys())
values = list(nfe_data.values())

# --- Plotting the Box Plot ---
plt.figure(figsize=(12, 6))

# Create box plot with customized visual styling
bp = plt.boxplot(values, labels=datasets, patch_artist=True, 
                 boxprops=dict(facecolor='skyblue', color='navy'),
                 capprops=dict(color='navy', linewidth=1.5),
                 whiskerprops=dict(color='navy', linewidth=1.5),
                 flierprops=dict(marker='o', color='red', alpha=0.5, markersize=5),
                 medianprops=dict(color='darkred', linewidth=2))

plt.title("Subset Size Distribution Across Datasets", fontsize=34)
plt.xlabel("Datasets", fontsize=32)
plt.ylabel("Subset Size", fontsize=32)

plt.xticks(fontsize=22, rotation=25)
plt.yticks(fontsize=22)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()







###########   Filter Comparisons  ###################

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif, chi2

# --- Configuration ---
DATA_DIR = r"D:\DataS Projects\DataS\dataset2"
DATASETS = {
    # "Colon": "Colon.xlsx", 
    # "CNS": "CNS.xlsx", 
    # "GLI": "GLI.csv", 
    # "SMK": "SMK.csv",
    # "Covid_19": "Covid.csv", 
    # "Leukemia": "Leukemia_3c.xlsx", 
    # "MLL": "MLL.xlsx", 
    # "SRBCT": "SRBCT.xlsx",
    # "Lung": "Lung.xlsx",
    # "lymphoma": "lymphoma.csv",
    # "Prostate": "Prostate.xlsx",
    # "ALLAML": "ALLAML.csv",
    # "nci9":"nci9.csv",
    # "TOX_171":"TOX_171.csv",
    # "CLL_SUB": "CLL_SUB_111.csv",
    "CARCINOM": "CARCINOM.csv"




}
TEST_SIZE, SEED = 0.10, 123
GA_SEED = 42

# --- 1. Dataset Loader & Split ---
def load_and_split_data(name, source):
    file_path = os.path.join(DATA_DIR, source)
    df = pd.read_excel(file_path, header=0) if source.endswith('.xlsx') else pd.read_csv(file_path, header=0)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if y.dtype == object or isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)
        
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
    )
    return X_train, X_test, y_train, y_test


# --- 2. Filter Methods (Computed strictly on X_train) ---
def compute_pearson_correlation(X_train, y_train):
    # Vectorized absolute Pearson correlation coefficient with target y
    y_centered = y_train - np.mean(y_train)
    y_denom = np.sqrt(np.sum(y_centered ** 2))
    
    if y_denom == 0:
        return np.arange(X_train.shape[1])
        
    X_centered = X_train - np.mean(X_train, axis=0)
    X_denom = np.sqrt(np.sum(X_centered ** 2, axis=0))
    X_denom[X_denom == 0] = 1e-8
    
    correlations = np.abs(np.sum(X_centered * y_centered[:, np.newaxis], axis=0) / (X_denom * y_denom))
    correlations = np.nan_to_num(correlations)
    return np.argsort(correlations)[::-1]  # Descending order

def compute_mutual_info(X_train, y_train):
    mi_scores = mutual_info_classif(X_train, y_train, random_state=SEED)
    return np.argsort(mi_scores)[::-1]

def compute_chi2(X_train, y_train):
    # Chi2 requires non-negative features; scale X_train using MinMaxScaler just for Chi2
    scaler_mm = MinMaxScaler()
    X_train_mm = scaler_mm.fit_transform(X_train)
    chi2_scores, _ = chi2(X_train_mm, y_train)
    chi2_scores = np.nan_to_num(chi2_scores)
    return np.argsort(chi2_scores)[::-1]

def compute_fisher_ratio(X_train, y_train):
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    global_mean = np.mean(X_train, axis=0)
    
    between_var = np.zeros(n_features)
    within_var = np.zeros(n_features)
    
    for c in classes:
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        between_var += n_c * (mean_c - global_mean) ** 2
        within_var += np.sum((X_c - mean_c) ** 2, axis=0)
        
    within_var[within_var == 0] = 1e-8
    fisher_scores = between_var / within_var
    return np.argsort(fisher_scores)[::-1]

def compute_relieff(X_train, y_train, sample_size=200, k_neighbors=5):
    n_samples, n_features = X_train.shape
    sample_size = min(sample_size, n_samples)
    indices = np.random.RandomState(SEED).choice(n_samples, sample_size, replace=False)
    
    scores = np.zeros(n_features)
    for idx in indices:
        x_i = X_train[idx]
        y_i = y_train[idx]
        
        diffs = X_train - x_i
        distances = np.linalg.norm(diffs, axis=1)
        
        same_class_mask = (y_train == y_i)
        same_class_mask[idx] = False
        diff_class_mask = (y_train != y_i)
        
        if not np.any(same_class_mask) or not np.any(diff_class_mask):
            continue
            
        hit_indices = np.where(same_class_mask)[0]
        miss_indices = np.where(diff_class_mask)[0]
        
        hit_dist = distances[hit_indices]
        miss_dist = distances[miss_indices]
        
        h_idx = hit_indices[np.argsort(hit_dist)[:k_neighbors]]
        m_idx = miss_indices[np.argsort(miss_dist)[:k_neighbors]]
        
        for f in range(n_features):
            diff_hit = np.sum(np.abs(X_train[idx, f] - X_train[h_idx, f]))
            diff_miss = np.sum(np.abs(X_train[idx, f] - X_train[m_idx, f]))
            scores[f] += (diff_miss - diff_hit) / k_neighbors
            
    return np.argsort(scores)[::-1]


# --- 3. Main Evaluation Pipeline ---
def main():
    filters = {
        "Pearson Correlation": compute_pearson_correlation,
        "Mutual Info": compute_mutual_info,
        "Chi-Squared": compute_chi2,
        "Fisher Ratio": compute_fisher_ratio,
        "ReliefF": compute_relieff
    }
    
    for dataset_name, source in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        
        # Load and split
        X_train, X_test, y_train, y_test = load_and_split_data(dataset_name, source)
        
        # Normalize (Z-score standardization) strictly on train, transform test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        top_k_ranks = list(range(1, 101))
        
        for filter_name, filter_func in filters.items():
            print(f"\n--- Evaluating Filter: {filter_name} ---")
            
            # 1. Rank features using training data only
            sorted_features = filter_func(X_train_scaled, y_train)
            top_100_features = sorted_features[:100]
            
            # 2. Cumulative Top-K Test Accuracy Evaluation, Average, and Maximum Calculation
            top_k_test_accuracies = []
            for k in top_k_ranks:
                actual_k = min(k, len(top_100_features))
                selected_k_features = top_100_features[:actual_k]
                
                clf = DecisionTreeClassifier(random_state=GA_SEED)
                clf.fit(X_train_scaled[:, selected_k_features], y_train)
                acc = accuracy_score(y_test, clf.predict(X_test_scaled[:, selected_k_features]))
                top_k_test_accuracies.append(acc)
                
            avg_test_acc = np.mean(top_k_test_accuracies)
            max_test_acc = np.max(top_k_test_accuracies)
            max_acc_idx = np.argmax(top_k_test_accuracies)
            best_cumulative_k = top_k_ranks[max_acc_idx]
            
            print(f"  Average Test Accuracy (Top-1 to Top-100): {avg_test_acc:.2f}")
            print(f"  Maximum Test Accuracy (Top-1 to Top-100): {max_test_acc:.2f} (Achieved with K = {best_cumulative_k} features)")
            
            # 3. Find optimal K via internal 5-fold CV on X_train
            cv_accuracies_per_k = []
            for k in top_k_ranks:
                selected_k_features = top_100_features[:k]
                X_train_k = X_train_scaled[:, selected_k_features]
                
                scores = cross_val_score(DecisionTreeClassifier(random_state=GA_SEED), X_train_k, y_train, cv=5, scoring='accuracy')
                cv_accuracies_per_k.append(np.mean(scores))
                
            best_k_idx = np.argmax(cv_accuracies_per_k)
            optimal_k = top_k_ranks[best_k_idx]
            print(f"  Optimal K selected via Training CV: {optimal_k}")
            
            # 4. Final evaluation on the strictly blind test set using optimal K
            final_features = top_100_features[:optimal_k]
            clf_final = DecisionTreeClassifier(random_state=GA_SEED)
            clf_final.fit(X_train_scaled[:, final_features], y_train)
            
            y_pred_test = clf_final.predict(X_test_scaled[:, final_features])
            final_test_acc = accuracy_score(y_test, y_pred_test)
            print(f"  Unbiased Test Accuracy with Optimal K={optimal_k}: {final_test_acc:.2f}")

if __name__ == "__main__":
    main()
    

##################  PRE, REC, F1
################################
############  Prostate #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.92, 0.91, 0.91]
after_values = [0.92, 0.91, 0.91]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("Prostate", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18,loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


############  ALLAML #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.85, 0.75, 0.75]
after_values = [1, 1, 1]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("ALLAML", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


############  SMK #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.52, 0.53, 0.52]
after_values = [0.79, 0.79, 0.79]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("SMK", fontsize=34,  pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  GLI #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.83, 0.78, 0.74]
after_values = [0.83, 0.78, 0.74]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("GLI", fontsize=34,  pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  Leukemia #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.75, 0.74, 0.75]
after_values = [0.9, 0.88, 0.87]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("Leukemia", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  TOX #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.43, 0.44, 0.42]
after_values = [0.69, 0.61, 0.59]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("TOX", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


############  NCI #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.54, 0.42, 0.44]
after_values = [0.54, 0.67, 0.58]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("NCI", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  Covid_19 #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.66, 0.62, 0.63]
after_values = [0.83, 0.79, 0.78]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("Covid_19", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()




##################  PRE, REC, F1
################################
############  Prostate #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.92, 0.91, 0.91]
after_values = [0.92, 0.91, 0.91]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("Prostate", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18,loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


############  ALLAML #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.85, 0.75, 0.75]
after_values = [1, 1, 1]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("ALLAML", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


############  SMK #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.52, 0.53, 0.52]
after_values = [0.79, 0.79, 0.79]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("SMK", fontsize=34,  pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  GLI #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.83, 0.78, 0.74]
after_values = [0.83, 0.78, 0.74]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("GLI", fontsize=34,  pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  Leukemia #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.75, 0.74, 0.75]
after_values = [0.9, 0.88, 0.87]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("Leukemia", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  TOX #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.43, 0.44, 0.42]
after_values = [0.69, 0.61, 0.59]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("TOX", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


############  NCI #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.54, 0.42, 0.44]
after_values = [0.54, 0.67, 0.58]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("NCI", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############  Covid_19 #########

import matplotlib.pyplot as plt
import numpy as np

# Data metrics for Prostate
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [0.66, 0.62, 0.63]
after_values = [0.83, 0.79, 0.78]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

# Create grouped bars
bars1 = plt.bar(x - width/2, before_values, width, label='Baseline', color='skyblue', edgecolor='navy')
bars2 = plt.bar(x + width/2, after_values, width, label='SISAFS', color='salmon', edgecolor='darkred')


# Chart formatting
plt.title("Covid_19", fontsize=34, pad=15)
plt.xlabel("Metrics", fontsize=32, labelpad=10)
plt.ylabel("Score", fontsize=32, labelpad=10)


plt.xticks(fontsize=22)
plt.yticks(fontsize=20)

plt.xticks(x, metrics, fontsize=22)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize = 18, loc = "lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
