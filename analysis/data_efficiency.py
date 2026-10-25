"""
The data-efficiency statistic g(L), and the bookkeeping that maps experiment
IDs to the five configurations of the study.

    D_v^(s)(L) = first number of training examples at which the running-best
                 validation loss of variant v, seed s, reaches L
                 (linear interpolation between 100K-example checkpoints)

    g_v^(s)(L) = 1 - D_v^(s)(L) / D_b^(s)(L)        (paired per seed vs baseline b)

g = 0.10 means "variant v needed 10% less data than the baseline to reach L".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CURVES_CSV = ROOT / "data" / "curves.csv"

# ---- which exp_ids form each experimental condition -------------------------
# Seeds were trained in several batches (exp_ids), so a condition is a group.
CONDITIONS = {
    "d512 EN-FR":        ["exp1", "exp4", "exp12"],   # 5 seeds, peak LR ~1.1e-3
    "d256 different LR": ["exp2", "exp3"],            # 5 seeds, peak LR ~1.6e-3
    "d256 matched LR":   ["exp7", "exp10", "exp11"],  # 5 seeds, peak LR ~1.1e-3
    "d512 modern arch":  ["exp8", "exp9"],            # 3 seeds
    "d512 DE-EN":        ["exp6"],                    # 3 seeds (En->De direction)
    "d512 mechanism":    ["exp5", "exp13"],           # learned, nope, t5 @ d512
    "d256 t5":           ["exp14"],                   # t5 @ d256, peak LR ~1.6e-3
}

# (group, display label, condition key) -- rows of the forest plot / Table 1
FOREST_ROWS = [
    ("Width",        "d=512, En-Fr",        "d512 EN-FR"),
    ("Width",        "d=256, higher LR",    "d256 different LR"),
    ("Optimization", "d=256, matched LR",   "d256 matched LR"),
    ("Architecture", "d=512, modern arch",  "d512 modern arch"),
    ("Language",     "d=512, De-En",        "d512 DE-EN"),
]

BAND_WIDTH = 0.25   # headline scalar: average g over [floor, floor + 0.25]
CONF = 0.95


def load_curves(path=CURVES_CSV):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} checkpoints from {df['run_id'].nunique()} runs.")
    return df


def runs_by_seed(df, exp_ids, variant):
    """
    {seed: (pairs_seen, val_loss)} for one variant in one condition.
    Keyed by SEED so that pairing across arms is by seed, never by position.
    """
    mask = df["exp_id"].isin(exp_ids) & (df["variant"] == variant)
    out = {}
    for _, g in df[mask].groupby("run_id"):
        g = g.sort_values("pairs_seen")
        out[int(g["seed"].iloc[0])] = (g["pairs_seen"].to_numpy(float),
                                       g["val_loss"].to_numpy(float))
    return out


def data_to_reach(pairs, loss, L):
    """D(L): examples needed for the running-best loss to first reach L (NaN if never)."""
    best = np.minimum.accumulate(loss)
    if best[-1] > L:
        return np.nan
    i = int(np.argmax(best <= L))
    if i == 0:
        return float(pairs[0])
    y0, y1 = float(pairs[i - 1]), float(pairs[i])
    x0, x1 = float(best[i - 1]), float(best[i])
    if x0 == x1:
        return y1
    return y0 + (x0 - L) / (x0 - x1) * (y1 - y0)


def run_floor(pairs_loss):
    return float(np.minimum.accumulate(pairs_loss[1])[-1])


def common_floor(*run_dicts):
    """Deepest loss reached by EVERY run in all given arms (the worst run's floor)."""
    return max(run_floor(r) for d in run_dicts for r in d.values())


def gap_matrix(runs_a, runs_b, Ls):
    """g(L) per seed: array (len(Ls), n_seeds), paired on the seed intersection."""
    seeds = sorted(set(runs_a) & set(runs_b))
    G = np.array([[1 - data_to_reach(*runs_a[s], L) / data_to_reach(*runs_b[s], L)
                   for s in seeds] for L in Ls])
    return G, seeds


def gap_curve(runs_a, runs_b, Ls):
    G, seeds = gap_matrix(runs_a, runs_b, Ls)
    return np.nanmean(G, axis=1), np.nanstd(G, axis=1, ddof=1), seeds


def forest_row(runs_a, runs_b, band_width=BAND_WIDTH, conf=CONF):
    """
    Headline scalar: average g over [floor, floor + band_width] WITHIN each seed,
    then mean across seeds with a Student-t interval.
    """
    floor = common_floor(runs_a, runs_b)
    Ls = np.linspace(floor, floor + band_width, 120)
    G, seeds = gap_matrix(runs_a, runs_b, Ls)
    per_seed = G.mean(axis=0)
    assert not np.isnan(per_seed).any(), "a seed does not cover this band"
    n = len(per_seed)
    mean, sd = per_seed.mean(), per_seed.std(ddof=1)
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)
    return dict(estimate=100 * mean, lo=100 * (mean - half), hi=100 * (mean + half),
                sd=100 * sd, n=n, L_lo=floor, L_hi=floor + band_width)


def final_bleu(df, cond, variant):
    """{seed: final BLEU (0-100)} for one variant in one condition."""
    x = df[df["exp_id"].isin(CONDITIONS[cond]) & (df["variant"] == variant)]
    x = x.sort_values("pairs_seen").groupby("seed")["bleu"].last()
    return {int(k): float(v) for k, v in x.items()}
