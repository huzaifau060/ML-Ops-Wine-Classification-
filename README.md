# Wine Cultivar Classification - MLOps Pipeline

[![CI Pipeline](https://github.com/REPLACE_WITH_YOUR_USERNAME/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE_WITH_YOUR_USERNAME/wine-mlops-pipeline/actions/workflows/ci.yml)

Automated, reproducible end-to-end MLOps pipeline for multi-class chemical cultivar classification using `sklearn.datasets.load_wine` (178 samples, 13 features, 3 target classes).

---

## 1. Repository Architecture

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow
├── data/
│   └── .gitkeep               # Preserves data directory structure
├── src/
│   ├── __init__.py
│   ├── data.py                # Data loading, validation, and stratified splitting
│   ├── train.py               # Dual classifier 5-fold CV training & MLflow tracking
│   └── evaluate.py            # Champion model inference verification on test split
├── tests/
│   ├── __init__.py
│   ├── test_data.py           # Unit tests for data shapes, features, and null checks
│   └── test_model_gate.py     # Quality Gates: F1 threshold, latency, output schema
├── .gitignore                 # Excludes caches, venvs, and local MLflow DBs
├── Makefile                   # GNU Makefile automating install, lint, test, train, clean
├── requirements.txt           # Pinned dependency versions for reproducible environments
└── README.md                  # Project documentation and execution instructions
```

---

## 2. Environment Setup & Automation (`Makefile`)

All workflows are automated using standard GNU Makefile targets to ensure strict developer-to-CI parity:

```bash
# 1. Install pinned dependencies
make install

# 2. Lint codebase with flake8 (max-line-length 100)
make lint

# 3. Run all unit tests and MLOps Quality Gate
make test

# 4. Execute MLflow training across all 6 hyperparameter configurations
make train

# 5. Clean compiled bytecode and test caches
make clean
```

---

## 3. Data Pipeline & Model Training

- **Dataset:** Scikit-Learn Wine dataset (13 chemical constituents, 3 cultivar classes).
- **Split:** Stratified 80/20 train/test split with fixed random seed (`random_state=42`).
- **Validation:** Automated checks ensure 0 null values and exactly 13 input features.
- **Model Families Evaluated:**
  1. `RandomForestClassifier` (3 distinct configurations)
  2. `GradientBoostingClassifier` (3 distinct configurations)
- **Validation Strategy:** 5-fold Stratified Cross-Validation evaluating Macro F1, Accuracy, and Log Loss.

---

## 4. MLflow Experiment Tracking & Model Registry

1. **Tracking Backend:** Configured using local SQLite database (`sqlite:///mlflow.db`).
2. **Experiment Name:** `Wine-Cultivar-Classification`.
3. **Artifact Logging:** Each candidate run logs hyperparameters, CV metrics, model signature (`mlflow.models.infer_signature`), and input examples.
4. **Champion Model Promotion:** Automatically identifies the best-performing run by validation Macro F1-score, registers it in the MLflow Model Registry under `WineClassifier`, and assigns the alias `champion`.

To view the interactive MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://localhost:5000` to inspect experiment comparisons, metrics charts, and the registered champion model.

---

## 5. Automated CI/CD & MLOps Quality Gates

GitHub Actions (`.github/workflows/ci.yml`) runs on every `push` and `pull_request` to `main`:
1. **Metric Threshold Gate:** Validation Macro F1-Score $\ge 0.88$.
2. **Inference Latency Gate:** Test batch inference time $\le 30\text{ ms}$.
3. **Output Schema Gate:** Model output strictly confined to class indices $\{0, 1, 2\}$.

---

## 6. Git Collaboration & Conflict Resolution

This repository follows a feature-branch workflow (`feature/mlflow-tracking`) and documents an engineered merge conflict simulation (`conflict-simulation`) demonstrating conflict marker resolution on `main`.
