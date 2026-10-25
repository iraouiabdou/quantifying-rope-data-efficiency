"""
Regenerate every table of the paper from data/curves.csv into results/.

    python analysis/tables.py

  results/table1_data_saved_vs_bandwidth.csv   Table 1 (and Figure 2 numbers at w=0.25)
  results/table4_paired_bleu.csv               Table 4
  results/table5_final_bleu_d512.csv           Table 5
"""
import numpy as np
import pandas as pd
from scipy import stats

import data_efficiency as de

OUT = de.ROOT / "results"


def table1(df):
    rows = []
    for group, label, cond in de.FOREST_ROWS:
        rope = de.runs_by_seed(df, de.CONDITIONS[cond], "rope")
        sin = de.runs_by_seed(df, de.CONDITIONS[cond], "sinusoidal")
        row = dict(group=group, configuration=label)
        for w in (0.15, 0.25, 0.35):
            r = de.forest_row(rope, sin, band_width=w)
            row[f"w{w}"] = f'{r["estimate"]:.1f} [{r["lo"]:.1f}, {r["hi"]:.1f}]'
            if w == 0.25:
                row.update(n=r["n"], band=f'[{r["L_lo"]:.3f}, {r["L_hi"]:.3f}]', sd=round(r["sd"], 2))
        rows.append(row)
    return pd.DataFrame(rows)[["group", "configuration", "n", "band", "w0.15", "w0.25", "w0.35", "sd"]]


def table4(df):
    rows, pooled = [], []
    for _, label, cond in de.FOREST_ROWS:
        r, s = de.final_bleu(df, cond, "rope"), de.final_bleu(df, cond, "sinusoidal")
        d = np.array([r[k] - s[k] for k in sorted(set(r) & set(s))])
        pooled += list(d)
        lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
        rows.append(dict(configuration=label, n=len(d), delta_bleu=f"{d.mean():+.2f}",
                         ci95=f"[{lo:+.2f}, {hi:+.2f}]"))
    p = np.array(pooled)
    rows.append(dict(configuration="Pooled", n=len(p),
                     delta_bleu=f"{p.mean():+.2f} (sd {p.std(ddof=1):.2f})",
                     ci95=f"p={stats.ttest_1samp(p, 0).pvalue:.1e}; RoPE ahead in {(p > 0).sum()}/{len(p)} pairs"))
    return pd.DataFrame(rows)


def table5(df):
    sources = {"t5": "d512 mechanism", "learned": "d512 mechanism", "rope": "d512 EN-FR",
               "sinusoidal": "d512 EN-FR", "nope": "d512 mechanism"}
    names = {"t5": "T5 relative", "learned": "Learned APE", "rope": "RoPE",
             "sinusoidal": "Sinusoidal", "nope": "NoPE"}
    vals = {v: np.array([de.final_bleu(df, c, v)[s] for s in (0, 1, 2)]) for v, c in sources.items()}
    base = vals["sinusoidal"].mean()
    return pd.DataFrame([dict(variant=names[v], bleu_mean=round(x.mean(), 2), bleu_sd=round(x.std(ddof=1), 2),
                              vs_sinusoidal="—" if v == "sinusoidal" else f"{x.mean() - base:+.2f}")
                         for v, x in vals.items()])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = de.load_curves()
    for name, fn, title in [("table1_data_saved_vs_bandwidth", table1, "Table 1: % data saved by RoPE vs sinusoidal [95% CI]"),
                            ("table4_paired_bleu", table4, "Table 4: paired BLEU difference, RoPE - sinusoidal"),
                            ("table5_final_bleu_d512", table5, "Table 5: final BLEU at d=512, seeds 0-2")]:
        t = fn(df)
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n{title}\n" + t.to_string(index=False))
