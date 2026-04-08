# Agentic ML AutoResearcher

An autonomous ML experiment loop inspired by 
[Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — 
built with sklearn, FastAPI, and a hill-climbing agent.

## What it does

Runs a ratchet loop: propose a config → evaluate → keep if better → 
discard if not → repeat. No human involvement after the first run.

**Dataset:** Breast Cancer Wisconsin (sklearn built-in)  
**Model:** Random Forest classifier  
**Metric:** 5-fold cross-validation accuracy  
**Result:** Baseline 94.55% → Best 95.61% (+1.06%) in 30 experiments

## The 3-file architecture (Phase 2)

| File | Role |
|---|---|
| `program.md` | Human control surface — research directions + stopping criteria |
| `train.py` | Editable asset — agent modifies CONFIG block per experiment |
| `agent.py` | Ratchet loop — orchestrates everything, writes `results.jsonl` |

## Quickstart

```bash
git clone https://github.com/SathyaPrakashD/agentic-ml-autoresearcher.git
cd agentic-ml-autoresearcher
pip install -r requirements.txt

# Run from terminal (Phase 2)
python agent.py

# Run as API (Phase 3)
uvicorn api.main:app --reload
```

## API endpoints (Phase 3)

| Endpoint | Method | Description |
|---|---|---|
| `/run` | POST | Start ratchet loop |
| `/status` | GET | Live progress + current best |
| `/results` | GET | Full experiment log |

```bash
# Start a run
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"n_experiments": 30, "accuracy_ceiling": 0.975, "stuck_window": 15}'

# Check progress
curl http://localhost:8000/status

# Get full log
curl http://localhost:8000/results
```

## Phases built

- **Phase 1** — Jupyter notebook with staircase chart
- **Phase 2** — 3-file CLI architecture (program.md + train.py + agent.py)
- **Phase 3** — FastAPI service with background task execution

## Why this exists

Most ML engineers tune models manually — one run at a time. This project 
automates that loop: the agent proposes, evaluates, and keeps only 
validated improvements. The human's only job is writing `program.md`.

> *"The role of the human shifts from experimenter to experimental designer."*  
> — Karpathy, March 2026
