"""
cost_simulator.py

Turns the classifier's held-out test predictions into a decision-support
tool: for a grid of (risk threshold x downtime-to-replacement cost ratio),
project the total cost of running the fleet at that threshold, using REAL
outcomes from the test set (not a hypothetical). This answers the question
every real predictive-maintenance deployment has to answer -- "what
threshold should we actually alert at?" -- and shows how the right answer
DEPENDS on your organization's specific cost structure, which is the genuinely
useful, non-obvious result: there is no single "best" threshold in the
abstract.

Cost model (deliberately simple and disclosed, not hidden):
    - False positive (flagged a drive that was actually healthy): the
      organization pays an unnecessary replacement_cost.
    - False negative (missed a drive that actually failed): the
      organization pays a downtime_cost (the expensive one).
    - True positive (correctly caught before failure): still pays
      replacement_cost (you do have to replace the drive) but AVOIDS the
      downtime_cost -- that avoided cost is the entire value proposition.
    - True negative: no cost.

Why threshold x cost-ratio (not threshold x time horizon): scaling the same
cost curve by a longer time horizon doesn't change WHERE the minimum sits,
only how tall it is -- a flat, uninteresting surface. The cost RATIO between
downtime and replacement is what actually moves the optimal threshold, which
is the useful, teachable result: an organization with cheap replacements but
catastrophic downtime should alert earlier (lower threshold) than one with
expensive replacements and mild downtime. The horizon is still used, as a
separate control, to scale the dollar total to a real planning period.
"""

import numpy as np
import pandas as pd

TEST_PREDICTIONS_PATH = "model/test_predictions.csv"
WINDOW_DAYS = 20  # the test set's implicit observation window


def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PREDICTIONS_PATH)


def cost_at_threshold(df: pd.DataFrame, threshold: float, replacement_cost: float, downtime_cost: float) -> dict:
    flagged = df["y_proba"] >= threshold
    tp = int((flagged & (df["y_true"] == 1)).sum())
    fp = int((flagged & (df["y_true"] == 0)).sum())
    fn = int((~flagged & (df["y_true"] == 1)).sum())
    tn = int((~flagged & (df["y_true"] == 0)).sum())
    total_cost = fp * replacement_cost + fn * downtime_cost + tp * replacement_cost
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": total_cost}


def build_cost_surface(
    replacement_cost: float,
    horizon_days: int,
    thresholds=None,
    cost_ratios=None,
):
    """
    Returns (thresholds, cost_ratios, Z, optimal) where:
      Z[i][j]  = projected total cost at thresholds[i], cost_ratios[j],
                 scaled to horizon_days
      optimal  = per cost-ratio row: the threshold that minimizes cost
    cost_ratios are downtime_cost / replacement_cost multiples.
    """
    df = load_test_predictions()
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    if cost_ratios is None:
        cost_ratios = np.array([2, 5, 10, 20, 35, 50])

    horizon_multiplier = horizon_days / WINDOW_DAYS
    Z = np.zeros((len(thresholds), len(cost_ratios)))
    for j, ratio in enumerate(cost_ratios):
        downtime_cost = replacement_cost * ratio
        row_costs = np.array([
            cost_at_threshold(df, t, replacement_cost, downtime_cost)["cost"] for t in thresholds
        ])
        Z[:, j] = row_costs * horizon_multiplier

    optimal = []
    for j, ratio in enumerate(cost_ratios):
        best_idx = int(np.argmin(Z[:, j]))
        optimal.append({
            "cost_ratio": float(ratio),
            "optimal_threshold": float(thresholds[best_idx]),
            "min_cost": float(Z[best_idx, j]),
        })

    return thresholds, cost_ratios, Z, optimal
