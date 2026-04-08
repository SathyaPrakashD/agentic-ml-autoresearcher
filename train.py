"""
train.py — the editable asset
The agent modifies ONLY the CONFIG block at the top.
Everything below the divider is fixed infrastructure — never touch it.
"""

# ==============================================================
# CONFIG — agent modifies this block only
# ==============================================================
CONFIG = {
    "n_estimators": 185,
    "max_depth": 21,
    "min_samples_split": 20,
    "min_samples_leaf": 8
}
# ==============================================================
# FIXED INFRASTRUCTURE — do not modify below this line
# ==============================================================

import json
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def evaluate(config: dict) -> dict:
    """
    Single source of truth for the ratchet.
    Returns a result dict — score + config + metadata.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    model = RandomForestClassifier(
        n_estimators      = config["n_estimators"],
        max_depth         = config["max_depth"],
        min_samples_split = config["min_samples_split"],
        min_samples_leaf  = config["min_samples_leaf"],
        random_state      = 42
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    return {
        "cv_accuracy" : round(float(scores.mean()), 6),
        "cv_std"      : round(float(scores.std()), 6),
        "config"      : config,
    }


if __name__ == "__main__":
    result = evaluate(CONFIG)
    # Print as JSON — agent.py reads this stdout
    print(json.dumps(result))
