"""
agent.py — the ratchet loop orchestrator
Reads:    program.md  (research directions — human authored)
Runs:     train.py    (editable asset — agent modifies this)
Writes:   results.jsonl (experiment log — append only)

Usage: python agent.py
"""

import json
import re
import subprocess
import sys
import copy
import random
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
TRAIN_SCRIPT  = "train.py"
RESULTS_FILE  = "results.jsonl"
PROGRAM_FILE  = "program.md"

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

# ── Program.md parser ───────────────────────────────────────────────────────
def parse_program_md() -> dict:
    """
    Reads program.md and extracts stopping criteria.
    This is the human's control surface — agent reads it, never writes it.
    """
    text = Path(PROGRAM_FILE).read_text()

    # Extract max experiments
    match = re.search(r"Stop after (\d+) experiments", text)
    max_experiments = int(match.group(1)) if match else 50

    # Extract accuracy ceiling
    match = re.search(r"cv_accuracy exceeds ([\d.]+)", text)
    accuracy_ceiling = float(match.group(1)) if match else 0.975

    # Extract stuck window
    match = re.search(r"no improvement found in the last (\d+)", text)
    stuck_window = int(match.group(1)) if match else 15

    return {
        "max_experiments"  : max_experiments,
        "accuracy_ceiling" : accuracy_ceiling,
        "stuck_window"     : stuck_window,
    }

# ── train.py runner ─────────────────────────────────────────────────────────
def run_experiment(config: dict) -> dict:
    """
    Writes config into train.py, runs it as subprocess,
    reads JSON result from stdout. This is the evaluate() call.
    """
    # Read current train.py
    source = Path(TRAIN_SCRIPT).read_text()

    # Replace CONFIG block
    new_config_str = "CONFIG = " + json.dumps(config, indent=4)
    new_source = re.sub(
        r"CONFIG\s*=\s*\{[^}]*\}",
        new_config_str,
        source,
        flags=re.DOTALL
    )
    Path(TRAIN_SCRIPT).write_text(new_source)

    # Run train.py and capture JSON output
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"train.py failed:\n{result.stderr}")

    return json.loads(result.stdout.strip())

# ── Agent logic ─────────────────────────────────────────────────────────────
def is_stuck(log: list, window: int) -> bool:
    recent = [e for e in log if e["id"] > 0][-window:]
    return len(recent) == window and not any(e["kept"] for e in recent)

def propose_next(current_best: dict, log: list) -> tuple[dict, str]:
    """
    Three-strategy agent — same logic as the notebook, now in production shape.
    """
    # Strategy 0 — Escape if stuck
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
        # Try up to 5 random jumps — pick one not already in log
        tried = {json.dumps(e["config"], sort_keys=True) for e in log}
        for _ in range(5):
            if current_best[least_tried] < (mn + mx) / 2:
                new_val = int(mn + (mx - mn) * random.uniform(0.6, 0.9))
            else:
                new_val = int(mn + (mx - mn) * random.uniform(0.1, 0.4))
            new_config = current_best.copy()
            new_config[least_tried] = new_val
            if json.dumps(new_config, sort_keys=True) not in tried:
                break
        return new_config, f"ESCAPE: large jump on {least_tried} ({current_best[least_tried]} → {new_val})"

    # Strategy 1 — Exploit recent wins
    recent_kept = [e for e in log[-6:] if e["kept"] and e["id"] > 0]
    if recent_kept:
        last_win = recent_kept[-1]
        prev_cfg = log[last_win["id"] - 1]["config"]
        curr_cfg = last_win["config"]
        changed = [
            (p, curr_cfg[p] - prev_cfg[p])
            for p in PARAMS if curr_cfg[p] != prev_cfg[p]
        ]
        if changed:
            param, delta = changed[0]
            step = random.choice(STEP_SIZES[param])
            direction = 1 if delta > 0 else -1
            new_val = int(np.clip(
                current_best[param] + direction * step,
                *BOUNDS[param]
            ))
            if new_val != current_best[param]:
                new_config = current_best.copy()
                new_config[param] = new_val
                return new_config, (
                    f"Exploit: {param} improved last time — "
                    f"continuing {'up' if direction > 0 else 'down'} "
                    f"({current_best[param]} → {new_val})"
                )

    # Strategy 2 — Explore least-recently-touched param
    recent_params = []
    for e in reversed(log[-8:]):
        if e["id"] > 0:
            prev = log[e["id"] - 1]["config"]
            for p in PARAMS:
                if e["config"][p] != prev[p] and p not in recent_params:
                    recent_params.append(p)

    untouched = [p for p in PARAMS if p not in recent_params]
    target = untouched[0] if untouched else random.choice(PARAMS)
    mn, mx  = BOUNDS[target]
    direction = 1 if current_best[target] < (mn + mx) / 2 else -1
    step = random.choice(STEP_SIZES[target])
    new_val = int(np.clip(current_best[target] + direction * step, mn, mx))
    new_config = current_best.copy()
    new_config[target] = new_val
    return new_config, (
        f"Explore: {target} nudged {'up' if direction > 0 else 'down'} "
        f"({current_best[target]} → {new_val})"
    )

# ── Results logger ───────────────────────────────────────────────────────────
def log_experiment(entry: dict):
    """Append-only JSONL log — every experiment recorded, nothing deleted."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Main ratchet loop ────────────────────────────────────────────────────────
def main():
    program  = parse_program_md()
    max_exp  = program["max_experiments"]
    ceiling  = program["accuracy_ceiling"]
    stuck_w  = program["stuck_window"]

    print(f"{'='*65}")
    print(f"  Agentic ML AutoResearcher")
    print(f"  Program : {PROGRAM_FILE}")
    print(f"  Max experiments : {max_exp}")
    print(f"  Accuracy ceiling: {ceiling*100:.1f}%")
    print(f"  Stuck window    : {stuck_w} experiments")
    print(f"{'='*65}")

    experiment_log   = []
    current_best_cfg = None
    current_best_score = -1.0

    # ── Baseline ──
    print("\nRunning baseline...")
    baseline_result = run_experiment({
        "n_estimators"    : 5,
        "max_depth"       : 2,
        "min_samples_split": 20,
        "min_samples_leaf" : 8,
    })
    current_best_cfg   = baseline_result["config"].copy()
    current_best_score = baseline_result["cv_accuracy"]

    baseline_entry = {
        "id"        : 0,
        "timestamp" : datetime.now().isoformat(),
        "config"    : baseline_result["config"],
        "cv_accuracy": baseline_result["cv_accuracy"],
        "cv_std"    : baseline_result["cv_std"],
        "kept"      : True,
        "reason"    : "Baseline — starting point",
    }
    experiment_log.append(baseline_entry)
    log_experiment(baseline_entry)

    print(f"Baseline: {current_best_score*100:.4f}%")
    print(f"{'-'*65}")

    # ── Ratchet loop ──
    for i in range(1, max_exp + 1):
        proposed_cfg, reason = propose_next(current_best_cfg, experiment_log)
        result = run_experiment(proposed_cfg)
        score  = result["cv_accuracy"]
        kept   = score > current_best_score

        if kept:
            current_best_score = score
            current_best_cfg   = proposed_cfg.copy()

        entry = {
            "id"         : i,
            "timestamp"  : datetime.now().isoformat(),
            "config"     : proposed_cfg,
            "cv_accuracy": score,
            "cv_std"     : result["cv_std"],
            "kept"       : kept,
            "reason"     : reason,
        }
        experiment_log.append(entry)
        log_experiment(entry)

        verdict = "KEPT   ✓" if kept else "DISCARDED ✗"
        print(
            f"Exp #{i:02d} | {verdict} | "
            f"{score*100:.4f}% | Best: {current_best_score*100:.4f}% | "
            f"{reason[:50]}..."
        )

        # ── Stopping criteria (from program.md) ──
        if current_best_score >= ceiling:
            print(f"\nStopping: accuracy ceiling {ceiling*100:.1f}% reached.")
            break

        recent = experiment_log[-stuck_w:]
        if len(recent) == stuck_w and not any(e["kept"] for e in recent):
            print(f"\nStopping: no improvement in last {stuck_w} experiments.")
            break

    # ── Summary ──
    kept_count = sum(1 for e in experiment_log if e["kept"] and e["id"] > 0)
    print(f"\n{'='*65}")
    print(f"  Run complete")
    print(f"  Baseline  : {baseline_result['cv_accuracy']*100:.4f}%")
    print(f"  Best      : {current_best_score*100:.4f}%")
    print(f"  Gain      : +{(current_best_score - baseline_result['cv_accuracy'])*100:.4f}%")
    print(f"  Kept      : {kept_count} / {i} experiments")
    print(f"  Best config: {current_best_cfg}")
    print(f"  Log       : {RESULTS_FILE}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
