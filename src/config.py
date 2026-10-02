"""Configuration settings for the Wine MLOps pipeline."""

MODEL_NAME = "WineClassifier"
METRIC_THRESHOLD_F1 = 0.90  # Quality gate proposed threshold from experimentation
BATCH_LATENCY_LIMIT_MS = 30.0
