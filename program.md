# AutoResearch Program — Breast Cancer Classifier

## Objective
Maximise 5-fold cross-validation accuracy on the sklearn Breast Cancer Wisconsin dataset
using a Random Forest classifier. The metric is `cv_accuracy` — higher is better.

## Search directions
- Increase `n_estimators` — more trees generally reduce variance up to a point
- Increase `max_depth` — current baseline is severely under-fitted at depth=2
- Decrease `min_samples_split` — lower threshold allows finer splits
- Decrease `min_samples_leaf` — smaller leaves capture more local structure

## Constraints — do not change these
- Dataset: `sklearn.datasets.load_breast_cancer()` — fixed, never swap it
- Evaluator: 5-fold CV with `random_state=42` — fixed, never modify
- Parameter bounds:
  - n_estimators: 5 to 500
  - max_depth: 2 to 25
  - min_samples_split: 2 to 20
  - min_samples_leaf: 1 to 10

## Stopping criteria
- Stop after 50 experiments, OR
- Stop if cv_accuracy exceeds 97.5%, OR
- Stop if no improvement found in the last 15 consecutive experiments

## What counts as improvement
Strictly greater than current best cv_accuracy. Ties are discarded.

## Notes for the agent
- Change one parameter per experiment — isolates causality
- If stuck for 5 experiments, make a large exploratory jump
- Record every experiment — wins and losses both matter
