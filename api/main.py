"""
api/main.py — FastAPI wrapper around the AutoResearch ratchet loop.

Endpoints:
  POST /run        — start a ratchet run (n_experiments in body)
  GET  /status     — current best score + progress
  GET  /results    — full experiment log

Usage:
  cd api
  uvicorn main:app --reload
"""

import json
import random
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Agentic ML AutoResearcher",
    description = "AutoResearch ratchet loop as a REST API — inspired by Karpathy's autoresearch",
    version     = "1.0.0",
)

# ── Shared state ───────────────────────────────────────────────────────────
state = {
    "running"          : False,
    "experiment_log"   : [],
    "current_best_cfg" : None,
    "current_best_score": -1.0,
    "baseline_score"   : None,
    "started_at"       : None,
    "finished_at"      : None,
    "stop_reason"      : None,
}

# ── Request schema ─────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    n_experiments   : int   = 30
    accuracy_ceiling: float = 0.975
    stuck_window    : int   = 15

# ── Constants ──────────────────────────────────────────────────────────────
BOUNDS = {
    "n_estimators"    : (5,  500),
    "max_depth"       : (2,  25),
    "min_samples_split": (2, 20),
    "min_samples_leaf" : (1, 10),
}
STEP_SIZES = {
    "n_estimators"    : [10, 30, 50, 100],
    "max_depth"       : [1, 2, 3],
    "min_samples_split": [1, 2, 4],
    "min_samples_leaf" : [1, 2],
}
PARAMS = list(BOUNDS.keys())

# ── Evaluator ──────────────────────────────────────────────────────────────
def evaluate(config: dict) -> dict:
    data = load_breast_cancer()
    X, y = data.data, data.target
    model = RandomForestClassifier(
        n_estimators      = config["n_estimators"],
        max_depth         = config["max_depth"],
        min_samples_split = config["min_samples_split"],
        min_samples_leaf  = config["min_samples_leaf"],
        random_state      = 42,
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return {
        "cv_accuracy": round(float(scores.mean()), 6),
        "cv_std"     : round(float(scores.std()),  6),
    }

# ── Agent ──────────────────────────────────────────────────────────────────
def is_stuck(log: list, window: int) -> bool:
    recent = [e for e in log if e["id"] > 0][-window:]
    return len(recent) == window and not any(e["kept"] for e in recent)

def propose_next(current_best: dict, log: list) -> tuple[dict, str]:
    tried = {json.dumps(e["config"], sort_keys=True) for e in log}

    if is_stuck(log, window=5):
        least_tried = min(
            PARAMS,
            key=lambda p: sum(
                1 for e in log
                if e["id"] > 0
                and log[e["id"] - 1]["config"][p] != e["config"][p]
            )
        )
        mn, mx = BOUNDS[least_tried]
        for _ in range(10):
            if current_best[least_tried] < (mn + mx) / 2:
                new_val = int(mn + (mx - mn) * random.uniform(0.6, 0.9))
            else:
                new_val = int(mn + (mx - mn) * random.uniform(0.1, 0.4))
            new_config = current_best.copy()
            new_config[least_tried] = new_val
            if json.dumps(new_config, sort_keys=True) not in tried:
                break
        return new_config, f"ESCAPE: large jump on {least_tried} ({current_best[least_tried]} → {new_val})"

    recent_kept = [e for e in log[-6:] if e["kept"] and e["id"] > 0]
    if recent_kept:
        last_win = recent_kept[-1]
        prev_cfg = log[last_win["id"] - 1]["config"]
        curr_cfg = last_win["config"]
        changed  = [(p, curr_cfg[p] - prev_cfg[p]) for p in PARAMS if curr_cfg[p] != prev_cfg[p]]
        if changed:
            param, delta = changed[0]
            step      = random.choice(STEP_SIZES[param])
            direction = 1 if delta > 0 else -1
            new_val   = int(np.clip(current_best[param] + direction * step, *BOUNDS[param]))
            if new_val != current_best[param]:
                new_config = current_best.copy()
                new_config[param] = new_val
                return new_config, (
                    f"Exploit: {param} {'up' if direction>0 else 'down'} "
                    f"({current_best[param]} → {new_val})"
                )

    recent_params = []
    for e in reversed(log[-8:]):
        if e["id"] > 0:
            prev = log[e["id"] - 1]["config"]
            for p in PARAMS:
                if e["config"][p] != prev[p] and p not in recent_params:
                    recent_params.append(p)

    untouched = [p for p in PARAMS if p not in recent_params]
    target    = untouched[0] if untouched else random.choice(PARAMS)
    mn, mx    = BOUNDS[target]
    direction = 1 if current_best[target] < (mn + mx) / 2 else -1
    step      = random.choice(STEP_SIZES[target])
    new_val   = int(np.clip(current_best[target] + direction * step, mn, mx))
    new_config = current_best.copy()
    new_config[target] = new_val
    return new_config, (
        f"Explore: {target} {'up' if direction>0 else 'down'} "
        f"({current_best[target]} → {new_val})"
    )

# ── Ratchet loop (runs in background thread) ───────────────────────────────
def ratchet_loop(n_experiments: int, accuracy_ceiling: float, stuck_window: int):
    s = state
    s["running"]           = True
    s["experiment_log"]    = []
    s["started_at"]        = datetime.now().isoformat()
    s["finished_at"]       = None
    s["stop_reason"]       = None

    baseline_cfg = {
        "n_estimators": 5, "max_depth": 2,
        "min_samples_split": 20, "min_samples_leaf": 8,
    }
    result = evaluate(baseline_cfg)
    s["current_best_cfg"]   = baseline_cfg.copy()
    s["current_best_score"] = result["cv_accuracy"]
    s["baseline_score"]     = result["cv_accuracy"]

    entry = {
        "id": 0, "timestamp": datetime.now().isoformat(),
        "config": baseline_cfg, **result,
        "kept": True, "reason": "Baseline — starting point",
    }
    s["experiment_log"].append(entry)

    for i in range(1, n_experiments + 1):
        proposed_cfg, reason = propose_next(s["current_best_cfg"], s["experiment_log"])
        result   = evaluate(proposed_cfg)
        score    = result["cv_accuracy"]
        kept     = score > s["current_best_score"]

        if kept:
            s["current_best_score"] = score
            s["current_best_cfg"]   = proposed_cfg.copy()

        s["experiment_log"].append({
            "id": i, "timestamp": datetime.now().isoformat(),
            "config": proposed_cfg, **result,
            "kept": kept, "reason": reason,
        })

        if s["current_best_score"] >= accuracy_ceiling:
            s["stop_reason"] = f"Accuracy ceiling {accuracy_ceiling*100:.1f}% reached"
            break

        recent = s["experiment_log"][-stuck_window:]
        if (len(recent) == stuck_window
                and not any(e["kept"] for e in recent)):
            s["stop_reason"] = f"No improvement in last {stuck_window} experiments"
            break

    s["running"]     = False
    s["finished_at"] = datetime.now().isoformat()
    if not s["stop_reason"]:
        s["stop_reason"] = f"Completed {n_experiments} experiments"

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/run", summary="Start the ratchet loop")
def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    if state["running"]:
        return JSONResponse(
            status_code=409,
            content={"error": "A run is already in progress. Check /status."}
        )
    background_tasks.add_task(
        ratchet_loop,
        req.n_experiments,
        req.accuracy_ceiling,
        req.stuck_window,
    )
    return {
        "message"         : "Ratchet loop started",
        "n_experiments"   : req.n_experiments,
        "accuracy_ceiling": req.accuracy_ceiling,
        "stuck_window"    : req.stuck_window,
    }

@app.get("/status", summary="Current best score and progress")
def get_status():
    log   = state["experiment_log"]
    kept  = sum(1 for e in log if e["kept"] and e["id"] > 0)
    total = len(log)
    gain  = (
        round((state["current_best_score"] - state["baseline_score"]) * 100, 4)
        if state["baseline_score"] else None
    )
    return {
        "running"           : state["running"],
        "experiments_run"   : total,
        "improvements_kept" : kept,
        "baseline_accuracy" : state["baseline_score"],
        "best_accuracy"     : state["current_best_score"],
        "gain_pct"          : gain,
        "best_config"       : state["current_best_cfg"],
        "started_at"        : state["started_at"],
        "finished_at"       : state["finished_at"],
        "stop_reason"       : state["stop_reason"],
    }

@app.get("/results", summary="Full experiment log")
def get_results():
    return {
        "total"      : len(state["experiment_log"]),
        "experiments": state["experiment_log"],
    }

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "agentic-ml-autoresearcher"}
