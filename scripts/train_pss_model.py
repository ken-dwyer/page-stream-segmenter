"""Train and save the production Page Stream Segmentation (PSS) 30-feature classifier.

Saves:
  models/pss_classifier.pkl (GBDT + LogisticRegression Ensemble + StandardScaler + Feature Metadata)
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"
MODEL_PKL = MODELS_DIR / "pss_classifier.pkl"

from run_pss_benchmark_tournament import (
    TABMEPP_FILE,
    DOCSPLIT_FILE,
    build_supervised_dataset,
)


def main():
    print("Loading benchmark training suites for production model fit...")
    with open(TABMEPP_FILE, encoding="utf-8") as f:
        tabme_data = json.load(f)
    with open(DOCSPLIT_FILE, encoding="utf-8") as f:
        docsplit_data = json.load(f)

    # Combine both training suites for maximum domain generalization:
    # 1. Real-world business archival folders (TABME++)
    # 2. Multi-class & same-class document streams (DocSplit-Seq: RVL-CDIP-N-MP)
    combined_train = tabme_data["train_packets"] + docsplit_data["train_packets"]
    tot_pgs = sum(p["total_pages"] for p in combined_train)
    print(f"Total training packets: {len(combined_train)} ({tot_pgs} pages)")

    X_train, y_train = build_supervised_dataset(combined_train)
    print(f"Extracted {X_train.shape[0]} pairwise training transitions with {X_train.shape[1]} features.")
    print(f"Positive splits: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%), Continuation: {np.sum(y_train == 0)}")

    print("Fitting HistGradientBoostingClassifier...")
    gbdt = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    gbdt.fit(X_train, y_train)

    print("Fitting Calibrated LogisticRegression on scaled features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=1.0, max_iter=600, random_state=42)
    lr.fit(X_train_scaled, y_train)

    artifact = {
        "gbdt": gbdt,
        "lr": lr,
        "scaler": scaler,
        "num_features": X_train.shape[1],
        "training_samples": X_train.shape[0],
    }

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Successfully saved production model artifact to {MODEL_PKL} ({MODEL_PKL.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
