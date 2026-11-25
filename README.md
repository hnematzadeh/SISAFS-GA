# SISAFS-GA
SISAFS-GA: Three-stage feature selection framework (DISAFS local filter, SISAFS global filter, and GA wrapper ensemble) for small-sample, high-dimensional datasets.




# SISAFS-GA: Feature Selection & Evaluation Pipeline (`SISAFS-GA.py`)

`SISAFS-GA.py` implements the **SISAFS-GA** feature selection workflow and comparative/visual evaluation blocks for high-dimensional, small-sample datasets. The script combines per-instance local filtering (**DISAFS**), dataset-level global ranking (**SISAFS**), and a bootstrap-based **Genetic Algorithm (GA)** wrapper ensemble over top-ranked features.

## Pipeline Structure in `SISAFS-GA.py`

* **Stage 1 — DISAFS & SISAFS (`load_and_normalize_data`, `process_single_sample`, `run_disafs_and_sisafs`, `evaluate_metrics`, `main`):**
  * Loads `.csv` or `.xlsx` datasets from `DATA_DIR` (`D:\DataS Projects\DataS\dataset2`), encodes target labels, splits data (`TEST_SIZE = 0.10`, `SEED = 123`) with stratification when possible, and fits a `StandardScaler` strictly on the training split.
  * Runs **DISAFS** in parallel (`joblib.Parallel(n_jobs=-1)`) across training instances using leave-one-out feature distances (`|X_loo - x_i|`) and one-vs-rest weighted ROC-AUC to retain features whose minimum-AUC class matches the sample's true label.
  * Computes leave-one-out `DecisionTreeClassifier(random_state=42)` training accuracy (`disafs_train_acc`), aggregates non-zero feature retention counts across training instances to produce `sorted_sisafs_features` (**SISAFS**), evaluates `Acc`, `Prec`, `Rec`, and `F1` across top-$K$ subsets (`10` to `100` in steps of `10`) and cumulative `1` to `100` accuracies, and saves `disafs_stats.pkl`, `sorted_sisafs_features.pkl`, and `topk_records.pkl` to `save_dir`.
* **Stage 2 — Bootstrap Wrapper Ensemble GA (`BudgetedGAFeatureSelector`):**
  * Loads `sorted_sisafs_features.pkl` from `rf"D:\SISAFS\NORMALIZED\{ACTIVE_DATASET_NAME}"` and takes the top 100 features (`FEATURE_BUDGET = 100`).
  * Runs across `B_RESAMPLES = 100` bootstrap resamples of `X_train` (`pop_size=30`, `generations=100`, `patience=20`, `ALPHA_PENALTY = 0.01`) using 5-fold cross-validation accuracy of `DecisionTreeClassifier(random_state=GA_SEED)` minus a subset-size penalty (`alpha * (len(selected_idx) / feature_budget)`).
  * Tracks per-run best/average fitness histories, Number of Function Evaluations (`nfe`), selected subset sizes, and feature selection frequencies, then evaluates cumulative top-1 to top-100 test accuracy.
* **Visualization & Additional Blocks:**
  * Plots top-10 feature selection frequency bar charts, GA fitness convergence trajectories for a selected run (`run_idx = 3`), and cumulative top-100 test accuracy against all-features baseline and average top-100 accuracy.
  * Includes standalone sections for box plots across datasets, classical filter comparisons (`Pearson Correlation`, `Mutual Info`, `Chi-Squared`, `Fisher Ratio`, `ReliefF`), and per-dataset `Precision`, `Recall`, and `F1-Score` comparison bar charts.

## Dependencies

* Python 3.9+
* `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `openpyxl`

## Configuration & Practical Notes

* **Dataset selection:** Configure `DATA_DIR` and uncomment/select the target entry in `DATASETS` before running.
* **Output directory alignment:** In the main SISAFS block, `save_dir` is hardcoded to `r"D:\SISAFS\NORMALIZED\GCM"`, whereas the GA section reads from `rf"D:\SISAFS\NORMALIZED\{ACTIVE_DATASET_NAME}"`—update `save_dir` to match your active dataset folder.
* **Script layout:** Because multiple standalone experiment and plotting blocks are concatenated in a single file, inspect and run or modularize each block as needed (and fix the trailing comma on the `Leukemia` entry in `nfe_data` if executing the box plot block directly).
````









######################
######################













This repository implements the **SISAFS-GA** feature selection pipeline for high-dimensional, small-sample datasets. It combines:
1. **DISAFS (Dynamic Instance-Specific Area-Based Feature Selection):** Per-instance local filtering using leave-one-out feature distance and weighted class ROC-AUC separation.
2. **SISAFS (Static Instance-Specific Area-Based Feature Selection):** Dataset-level aggregation and ranking of features retained across training samples.
3. **Wrapper Ensemble GA (`BudgetedGAFeatureSelector`):** A parsimony-regularized Genetic Algorithm run across bootstrap resamples of the training data over the top-100 SISAFS-ranked features.

## Repository Pipeline Overview

The provided script is organized into two main stages followed by visualization and summary plotting blocks:

* **Stage 1 — DISAFS & SISAFS (`load_and_normalize_data`, `process_single_sample`, `run_disafs_and_sisafs`):**
  * Performs a stratified train/test split (`TEST_SIZE = 0.10`, `SEED = 123`) and fits a `StandardScaler` strictly on the training split to prevent data leakage.
  * Evaluates each training instance `i` in a leave-one-out manner (`joblib.Parallel(n_jobs=-1)`) across all features `f`, computing one-vs-rest weighted ROC-AUC on absolute feature distances `|X_loo[:, f] - x_i[f]|` and retaining features whose lowest-AUC class matches `y_train[i]`.
  * Computes leave-one-out training accuracy per sample (`disafs_train_acc`), ranks features with non-zero retention frequency (`sorted_sisafs_features`), evaluates `DecisionTreeClassifier(random_state=42)` across subsets, and saves `disafs_stats.pkl`, `sorted_sisafs_features.pkl`, and `topk_records.pkl`.
* **Stage 2 — Bootstrap Wrapper Ensemble GA (`BudgetedGAFeatureSelector`):**
  * Loads `sorted_sisafs_features.pkl` and restricts the search pool to the top 100 ranked features (`FEATURE_BUDGET = 100`).
  * Runs across `B_RESAMPLES = 100` bootstrap resamples of `X_train` (`pop_size=30`, `generations=100`, `patience=20`, `ALPHA_PENALTY = 0.01`, 5-fold cross-validation fitness with a subset-size penalty).
  * Tracks per-run fitness histories, Number of Function Evaluations (`NFE`), selected feature counts, and global feature selection frequencies.
* **Evaluation & Plotting:**
  * Evaluates cumulative top-1 to top-100 feature subsets on the held-out test set using `DecisionTreeClassifier`.
  * Generates plots for top-10 feature selection frequencies, GA fitness convergence trajectories, cumulative top-100 test accuracy against a baseline, and box plots for NFE / subset statistics.

## Requirements

* Python 3.9+
* `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `openpyxl` (for `.xlsx` datasets)

## Usage & Configuration Notes

1. **Dataset paths:** Update `DATA_DIR` and the active entries in `DATASETS` to match your local dataset directory and filenames.
2. **Serialization path consistency:** In Stage 1, `save_dir` is currently set to `r"D:\SISAFS\NORMALIZED\GCM"`, whereas Stage 2 loads from `rf"D:\SISAFS\NORMALIZED\{ACTIVE_DATASET_NAME}"`. Align both paths (and make sure the final dictionary literal in the plotting block has valid syntax before running).
3. **Run the pipeline:** Execute the script from your Python environment or IDE after configuring the target dataset.
