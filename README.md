# Santander Customer Transaction Prediction

**Machine Learning · Python · Exploratory Data Analysis · Imbalanced Classification · XGBoost · Ensemble Learning**

An end-to-end machine-learning study of the [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) dataset, conducted as part of the **M.Sc. Data Analytics** program at **Justus Liebig University Giessen** (summer semester 2026).

The objective is to predict a binary transaction-related outcome from **200 anonymized numerical features**. The project covers exploratory data analysis (EDA), memory-efficient preprocessing, model training, hyperparameter tuning, stratified cross-validation, ensemble learning, and classification-threshold selection.

> **Portfolio focus:** My contributions centered on the EDA and data preparation, the evaluation and tuning of decision trees, random forests and XGBoost, stacking, and the synthesis of model-comparison results. This was a university group project; complementary models and externally supplied results are identified separately below.

## At a glance

| | |
|---|---|
| **Dataset** | 200,000 observations; 200 anonymized numerical predictors; binary target |
| **Class balance** | Approximately 89.95% negative / 10.05% positive |
| **Split** | Stratified 80% training / 20% held-out test (`random_state=42`) |
| **Validation** | Stratified 5-fold cross-validation |
| **Main evaluation metrics** | ROC-AUC and PR-AUC; precision, recall and F1 at selected thresholds |
| **Best directly comparable individual model** | Tuned XGBoost (1,500 trees) |
| **Selected-model test performance** | ROC-AUC **0.8959**, PR-AUC **0.6002**, F1 **0.5621** at threshold **0.24** |

## Questions investigated

1. How do a single decision tree, a random forest, and XGBoost differ in predictive performance and generalization?
2. Does stacking logistic regression, random forest, and XGBoost outperform the strongest individual model?
3. How does changing the decision threshold affect the trade-off between precision and recall?

## Methodology

### 1. Exploratory data analysis

- Checked missing and infinite values, duplicate rows and features, near-constant predictors, and unusual value encodings.
- Examined feature distributions, skewness, kurtosis, potential outliers, and relationships with the target.
- Investigated all **19,900 unique pairwise Pearson correlations** among the 200 features: even the strongest absolute correlations were below approximately 0.01.
- Studied mutual information and class-specific feature distributions. No single feature provided an obvious strong separation between the classes.

### 2. Data preparation

- Converted numerical predictors from `float64` to `float32`, reducing their memory footprint from approximately **305 MB to 153 MB**.
- Encoded the binary target as `int8`.
- Created a reproducible, stratified **80/20 train-test split**.
- Stored prepared datasets as Parquet files. Applied model-specific scaling rather than scaling all predictors globally.

### 3. Modeling and validation

Models evaluated included a dummy baseline, decision tree, random forest, XGBoost, and a stacking ensemble of logistic regression, random forest, and XGBoost. Logistic regression and other complementary approaches were also included in the broader group comparison.

Hyperparameters were explored with randomized and targeted searches. Some searches used stratified subsets for computational efficiency, followed by validation on the full training set.

**Why ROC-AUC and PR-AUC?** With only ~10% positive cases, a classifier predicting the negative class every time would still achieve almost 90% accuracy. PR-AUC helps assess how well models identify the minority class, alongside ROC-AUC.

## Results

Selected **approximate stratified cross-validation results** from my project report:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Dummy classifier | 0.500 | 0.101 |
| Decision tree — baseline | 0.559 | 0.122 |
| Decision tree — tuned | 0.653 | 0.181 |
| Random forest — baseline | 0.821 | 0.407 |
| Random forest — tuned | 0.848 | 0.475 |
| XGBoost — 1,500 trees | **0.896** | **0.605** |
| Stacking ensemble | 0.892 | 0.603 |

The tuned **XGBoost** model was the strongest directly comparable individual model. Stacking produced similar, but slightly lower, performance at a higher computational cost; I therefore did not select it as the final model.

**Held-out test evaluation (XGBoost, 1,500 trees):** ROC-AUC ≈ **0.8959**, PR-AUC ≈ **0.6002**, and F1 ≈ **0.5621** using a threshold of **0.24**, selected from out-of-fold predictions on the training data.

A separately developed group-member XGBoost approach with frequency-encoded features achieved approximately **0.905 ROC-AUC** and **0.642 PR-AUC**. This was an **exploratory external comparison**, not a controlled experiment: feature engineering, training configuration, and hyperparameters differed, so the improvement cannot be attributed to frequency encoding alone.

## Key findings

- Tuning reduced severe overfitting in a single decision tree, but it still performed substantially worse than ensemble methods.
- Random forests substantially improved performance over a single decision tree; XGBoost performed best among the directly comparable individual models.
- A good ROC-AUC or PR-AUC does not guarantee useful positive predictions at the default **0.50** threshold.
- Out-of-fold threshold analysis indicated that a threshold near **0.24** yielded a better precision–recall balance for the selected XGBoost model under the F1 criterion.
- With anonymized predictors and no business-specific error costs, the chosen threshold is **statistically motivated**, not necessarily optimal for a real banking decision.

## Repository structure

The notebooks are intentionally kept at the **repository root**, because the current code uses `Path.cwd()` to locate `data/` and `results/`. The `data/` folders are generated locally and are not necessarily present in the public repository.

```text
santander-customer-transaction-prediction/
├── README.md
├── 01_EDA_Santander.ipynb
├── 02_data_preparation.ipynb
├── 03_dummy_classifier.ipynb
├── 04_RandomForests_Santander.ipynb
├── 05_DecisionTree.ipynb
├── 06_XGBOOST.ipynb
├── 07_Stacking.ipynb
├── data/
│   ├── raw/                # local dataset; not required in GitHub
│   └── processed/          # generated train/test Parquet files
└── results/
    ├── cv/                 # cross-validation summary CSVs
    └── external/           # optional, clearly attributed group comparisons
```

The provided notebooks should be renamed to the filenames above when publishing (removing version suffixes such as `(5)`), or the README links below should be adjusted to match the uploaded filenames.

## Notebooks

| Notebook | Content |
|---|---|
| [`01_EDA_Santander.ipynb`](01_EDA_Santander.ipynb) | Data loading, validation, distributions, outliers, correlations, mutual information |
| [`02_data_preparation.ipynb`](02_data_preparation.ipynb) | Data types, target encoding, stratified split, Parquet export |
| [`03_dummy_classifier.ipynb`](03_dummy_classifier.ipynb) | Baseline and evaluation metrics |
| [`04_RandomForests_Santander.ipynb`](04_RandomForests_Santander.ipynb) | Random-forest baseline, tuning, CV |
| [`05_DecisionTree.ipynb`](05_DecisionTree.ipynb) | Tree baseline, overfitting analysis, tuning |
| [`06_XGBOOST.ipynb`](06_XGBOOST.ipynb) | XGBoost searches, final model, test evaluation, threshold analysis |
| [`07_Stacking.ipynb`](07_Stacking.ipynb) | Ensemble with internal and external CV, comparison with XGBoost |

## Data source and reproducibility

**Data:** [OpenML dataset 45566](https://www.openml.org/search?type=data&status=active&id=45566), originally used in the [2019 Kaggle Santander competition](https://www.kaggle.com/competitions/santander-customer-transaction-prediction).

The first notebook retrieves the dataset with `sklearn.datasets.fetch_openml(data_id=45566, target_column="target", as_frame=True)`. Subsequent notebooks expect the raw CSV at `data/raw/santander_openml.csv` and the prepared files at `data/processed/train.parquet` and `data/processed/test.parquet`.

To explore or rerun the project:

1. Install Python and JupyterLab, plus the packages used in the notebooks (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `pyarrow`, `joblib`).
2. Open JupyterLab **from the repository root**, so `Path.cwd()` resolves to this folder.
3. Run notebook **01** to retrieve and inspect the data, and save the raw dataset where notebook **02** expects it (if it has not already been saved).
4. Run notebook **02** to generate the prepared Parquet files.
5. Run the modeling notebooks (**03–07**) as needed. Their results are summarized in the table above; available cross-validation summaries are stored in `results/cv/`.

**Reproducibility note:** Some analyses involve long tuning or cross-validation runs. The consolidated model-comparison notebook is not included because part of its external group-model comparison depends on `.npy` prediction files that are not available in this repository. The model results are summarized above and in the available `results/cv/` CSV files; some results may require rerunning individual modeling notebooks to reproduce.

**Data publication:** The complete source dataset and generated training/test files are intentionally excluded from the public repository; obtain them from the original data provider and review the applicable usage terms. Small aggregate metric CSVs can be included, provided they contain no private data.

## My contribution and collaboration

This project was completed within a **three-person university group**. My documented work covered data exploration and preprocessing, decision-tree/random-forest/XGBoost modeling and tuning, out-of-fold threshold analysis, stacking, and the synthesis and interpretation of model-comparison results. The broader team also contributed complementary modeling approaches and shared results. External predictions and frequency-encoded model results are discussed as **collaborators' work**, not presented as independently developed models of mine.

## Limitations and next steps

- Anonymized features restrict domain-specific interpretation.
- Some hyperparameter searches were conducted on stratified subsets due to computational limitations.
- Comparisons with external models are exploratory because configurations and validation procedures were not always identical.
- Potential next steps: controlled feature-engineering comparisons, calibrated probabilities, domain-specific cost-sensitive threshold selection, and a fully self-contained reproducibility pipeline.

## Tools

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter` · `Parquet`

---

**Project context:** Scientific Programming and Data Analysis / Advanced Data Analytics, Justus Liebig University Giessen, summer semester 2026.
