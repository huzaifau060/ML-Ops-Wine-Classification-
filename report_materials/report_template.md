# FAST National University of Computer & Emerging Sciences
## Department of Computer Science
### Course: Machine Learning Operations (MLOps) - Fall 2026
### Assignment 01: MLOps task with CI/CD, MLFlow & Makefile Automation

**Student Name:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**GitHub Repository Link:** [https://github.com/<your-username>/wine-mlops-pipeline](https://github.com/<your-username>/wine-mlops-pipeline)

---

## 1. Table 1: Hyperparameter Search Results

The dual-classifier search grid was evaluated using 5-fold stratified cross-validation on the 80% training split (142 samples, 13 features) with fixed seed `random_state=42`.

| Model Family | Config Name | Hyperparameters | Val Macro F1 | Val Accuracy | Val Log Loss | Selected Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **RandomForest** | `rf_config_1` | `n_estimators=50, max_depth=3, min_samples_split=2` | 0.9665 | 0.9653 | 0.2055 | Candidate |
| **RandomForest** | `rf_config_2` | `n_estimators=100, max_depth=5, min_samples_split=2` | **0.9789** | **0.9791** | **0.1630** | **Champion** |
| **RandomForest** | `rf_config_3` | `n_estimators=150, max_depth=None, min_samples_split=4` | 0.9789 | 0.9791 | 0.1688 | Candidate |
| **GradientBoosting** | `gbm_config_1` | `n_estimators=50, learning_rate=0.05, max_depth=3` | 0.9053 | 0.9022 | 0.2295 | Candidate |
| **GradientBoosting** | `gbm_config_2` | `n_estimators=100, learning_rate=0.1, max_depth=3` | 0.9259 | 0.9236 | 0.4281 | Candidate |
| **GradientBoosting** | `gbm_config_3` | `n_estimators=150, learning_rate=0.2, max_depth=4` | 0.9197 | 0.9160 | 0.5264 | Candidate |

*Winner:* **`rf_config_2` (RandomForest)** was promoted to the MLflow Model Registry under `WineClassifier` with alias `champion`.

---

## 2. MLflow UI Visualizations

> *Tip: Launch `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open `http://localhost:5000` to take these screenshots.*

### 2.1 Experiment Overview
*(Insert screenshot of MLflow Experiment `Wine-Cultivar-Classification` displaying all 6 logged runs with their parameters and metrics)*

### 2.2 Metrics Comparison Plot
*(Insert screenshot of MLflow chart comparing Validation Macro F1 and Validation Log Loss across runs)*

### 2.3 Model Registry (`WineClassifier`)
*(Insert screenshot of the Model Registry showing version 1 with alias `champion`)*

---

## 3. CI/CD Evidence (GitHub Actions)

*(Insert screenshot of GitHub Actions run showing successful completion of `build-and-test` job on Python 3.10 with green checks across `make install`, `make lint`, and `make test`)*

---

## 4. Git Collaboration & Conflict Resolution Log

### 4.1 Git Tree Log
```text
* fe500ec style(config): wrap comment to adhere to flake8 line length <= 100
*   985f384 Merge branch 'conflict-simulation' and resolve merge conflict in src/config.py
|\  
| * a3d3277 config: update F1 threshold to 0.90 on conflict-simulation branch
* | f43c662 config: reinforce baseline F1 threshold to 0.88 on main
|/  
* e935707 feat(config): introduce central pipeline configuration parameters
*   6a7a803 Merge pull request #1 from feature/mlflow-tracking
|\  
| * db159a8 feat(pipeline): implement modular data pipeline, dual classifier tuning, MLflow tracking and quality gates
|/  
* 49b4777 feat(init): initialize project structure, Makefile automation and CI workflow
```

### 4.2 Merge Conflict Simulation & Terminal Output
```bash
$ git checkout main
Switched to branch 'main'

$ git merge conflict-simulation
Auto-merging src/config.py
CONFLICT (content): Merge conflict in src/config.py
Automatic merge failed; fix conflicts and then commit the result.

$ git status
On branch main
Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   src/config.py

$ cat src/config.py
"""Configuration settings for the Wine MLOps pipeline."""

MODEL_NAME = "WineClassifier"
<<<<<<< HEAD
METRIC_THRESHOLD_F1 = 0.88  # Mandatory assignment baseline specification
=======
METRIC_THRESHOLD_F1 = 0.90  # Quality gate proposed threshold from experimentation
>>>>>>> conflict-simulation
BATCH_LATENCY_LIMIT_MS = 30.0

$ # Resolution: Reconciled experimental suggestion with mandatory baseline
$ git add src/config.py
$ git commit -m "Merge branch 'conflict-simulation' and resolve merge conflict in src/config.py"
[main 985f384] Merge branch 'conflict-simulation' and resolve merge conflict in src/config.py
```

---

## 5. Analytical Response (Max 150 words)

A comparative analysis between Random Forest and Gradient Boosting demonstrates distinct performance dynamics on the chemical cultivar dataset. Random Forest achieved superior generalization, with `rf_config_2` reaching a peak validation Macro F1 of 0.9789 and the lowest validation Log Loss of 0.1630. Its ensemble averaging effectively mitigated overfitting across the 13 continuous chemical features. In contrast, Gradient Boosting attained a maximum Macro F1 of 0.9259 under `gbm_config_2`. While sequential boosting refined residual errors, higher learning rates and depths led to elevated validation log loss (0.5264 in `gbm_config_3`), indicating sensitivity to the relatively small sample size (142 training instances). Random Forest (`rf_config_2`) proved more robust and was selected as the production champion.
