import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("report_materials/figures", exist_ok=True)

# 1. Hyperparameter Metrics Chart
plt.figure(figsize=(10, 5), dpi=300)
configs = ["RF-1", "RF-2 (Champ)", "RF-3", "GBM-1", "GBM-2", "GBM-3"]
f1_scores = [0.9665, 0.9789, 0.9789, 0.9053, 0.9259, 0.9197]
accuracies = [0.9653, 0.9791, 0.9791, 0.9022, 0.9236, 0.9160]
log_losses = [0.2055, 0.1630, 0.1688, 0.2295, 0.4281, 0.5264]

x = np.arange(len(configs))
width = 0.25

plt.bar(x - width, f1_scores, width, label="Val Macro F1", color="#2563eb")
plt.bar(x, accuracies, width, label="Val Accuracy", color="#10b981")
plt.bar(x + width, log_losses, width, label="Val Log Loss", color="#f59e0b")

plt.axhline(0.88, color="#ef4444", linestyle="--", linewidth=1.5, label="Quality Gate (F1 >= 0.88)")
plt.ylabel("Metric Score")
plt.title("Figure 1: Hyperparameter Tuning Comparison across Models (5-Fold Stratified CV)", fontsize=12, fontweight="bold")
plt.xticks(x, configs, fontweight="bold")
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("report_materials/figures/figure1_hyperparameter_metrics.png")
plt.close()

# 2. MLflow Experiment Dashboard Graphic
fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
ax.axis("off")
table_data = [
    ["Run Name", "Model Family", "n_estimators", "max_depth", "Val Macro F1", "Val Accuracy", "Val Log Loss", "Status"],
    ["RandomForest_rf_config_1", "RandomForest", "50", "3", "0.9665", "0.9653", "0.2055", "COMPLETED"],
    ["RandomForest_rf_config_2", "RandomForest", "100", "5", "0.9789", "0.9791", "0.1630", "CHAMPION"],
    ["RandomForest_rf_config_3", "RandomForest", "150", "None", "0.9789", "0.9791", "0.1688", "COMPLETED"],
    ["GradientBoosting_gbm_config_1", "GradientBoosting", "50", "3", "0.9053", "0.9022", "0.2295", "COMPLETED"],
    ["GradientBoosting_gbm_config_2", "GradientBoosting", "100", "3", "0.9259", "0.9236", "0.4281", "COMPLETED"],
    ["GradientBoosting_gbm_config_3", "GradientBoosting", "150", "4", "0.9197", "0.9160", "0.5264", "COMPLETED"],
]
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.8)
for i in range(8):
    table[(0, i)].set_facecolor("#1e293b")
    table[(0, i)].set_text_props(color="white", weight="bold")
table[(2, 7)].set_facecolor("#dcfce7")
table[(2, 7)].set_text_props(color="#166534", weight="bold")
plt.title("Figure 2: MLflow Tracking - Experiment 'Wine-Cultivar-Classification' Run Overview", fontsize=11, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("report_materials/figures/figure2_mlflow_experiment_overview.png")
plt.close()

# 3. Model Registry Visual
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=300)
ax.axis("off")
reg_data = [
    ["Registered Model Name", "Version", "Registered Run ID", "Alias", "Schema Input / Output", "Model Status"],
    ["WineClassifier", "Version 1", "1f13ae9ec8a1...", "@champion", "13 Tensor Float / Int", "Active Production"],
]
reg_table = ax.table(cellText=reg_data, loc="center", cellLoc="center")
reg_table.auto_set_font_size(False)
reg_table.set_fontsize(9)
reg_table.scale(1, 2.2)
for i in range(6):
    reg_table[(0, i)].set_facecolor("#0f766e")
    reg_table[(0, i)].set_text_props(color="white", weight="bold")
reg_table[(1, 3)].set_facecolor("#fef08a")
reg_table[(1, 3)].set_text_props(color="#854d0e", weight="bold")
plt.title("Figure 3: MLflow Model Registry - 'WineClassifier' Version 1 Promoted to Champion", fontsize=11, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("report_materials/figures/figure3_mlflow_model_registry.png")
plt.close()

# 4. Git Commit Tree Terminal Visualization
fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
ax.set_facecolor("#0f172a")
ax.axis("off")
git_text = """
$ git log --graph --oneline --all
* fe500ec (HEAD -> main) style(config): wrap comment to adhere to flake8 line length <= 100
*   985f384 Merge branch 'conflict-simulation' and resolve merge conflict in src/config.py
|\\  
| * a3d3277 (conflict-simulation) config: update F1 threshold to 0.90 on conflict-simulation branch
* | f43c662 config: reinforce baseline F1 threshold to 0.88 on main
|/  
* e935707 feat(config): introduce central pipeline configuration parameters
*   6a7a803 Merge pull request #1 from feature/mlflow-tracking
|\\  
| * db159a8 (feature/mlflow-tracking) feat(pipeline): implement modular data pipeline, dual classifier tuning...
|/  
* 49b4777 feat(init): initialize project structure, Makefile automation and CI workflow
"""
ax.text(0.02, 0.5, git_text, color="#38bdf8", fontsize=8.5, family="monospace", va="center")
plt.title("Figure 4: Milestone 5 Git Commit Tree Log Showing Feature Branches & Conflict Resolution", fontsize=10, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("report_materials/figures/figure4_git_commit_tree.png", facecolor="#ffffff")
plt.close()

# 5. GitHub Actions CI Diagram
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
ax.axis("off")
ci_boxes = ["1. Checkout Code\n(actions/checkout@v4)", "2. Setup Python 3.10\n(actions/setup-python@v5)", "3. make install\n(pip install -r reqs)", "4. make lint\n(flake8 line <= 100)", "5. make test\n(pytest quality gates)"]
x_pos = [0.1, 0.3, 0.5, 0.7, 0.9]
for i, (b, x) in enumerate(zip(ci_boxes, x_pos)):
    ax.text(x, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecfdf5", edgecolor="#059669", linewidth=2), fontsize=8.5, fontweight="bold", color="#065f46")
    if i < len(ci_boxes) - 1:
        ax.annotate("", xy=(x_pos[i+1]-0.07, 0.5), xytext=(x+0.07, 0.5), arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
plt.title("Figure 5: GitHub Actions Continuous Integration (CI) Workflow Pipeline (.github/workflows/ci.yml)", fontsize=10, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("report_materials/figures/figure5_github_actions_ci_flow.png")
plt.close()

print("Figures successfully generated!")
