"""
Regenerate every figure from data/curves.csv into figures/.

    python analysis/figures.py

"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data_efficiency as de

OUT = de.ROOT / "figures"
DPI = 200

STYLE = {
    "rope":       ("RoPE",        "#1f77b4"),
    "sinusoidal": ("Sinusoidal",  "#d62728"),
    "learned":    ("Learned",     "#2ca02c"),
    "t5":         ("T5 relative", "#9467bd"),
    "nope":       ("NoPE",        "#7f7f7f"),
}
Y_CROP_D512 = (2.28, 3.20)
X_CAP = 4_000_000
N_GRID_CURVE = 300
N_GRID_GAP = 100


# ---------------------------------------------------------------- primitives
def curve_band(runs):
    seeds = sorted(runs)
    lo = max(runs[s][0].min() for s in seeds)
    hi = min(runs[s][0].max() for s in seeds)
    grid = np.linspace(lo, hi, N_GRID_CURVE)
    stacked = np.array([np.interp(grid, runs[s][0], runs[s][1]) for s in seeds])
    return grid, stacked.mean(0), stacked.std(0, ddof=1), len(seeds)


def plot_curves(ax, series, ylim=Y_CROP_D512):
    for runs, key in series:
        label, colour = STYLE[key]
        grid, mean, sd, n = curve_band(runs)
        ax.plot(grid, mean, color=colour, lw=2, label=f"{label} (n={n})")
        ax.fill_between(grid, mean - sd, mean + sd, color=colour, alpha=0.25, lw=0)
    ax.set_ylim(*ylim)
    ax.set_xlim(500_000, X_CAP)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e6:g}M")
    ax.set_xlabel("Training examples seen")
    ax.set_ylabel("Validation loss")
    ax.legend(loc="upper right", frameon=False)


def plot_gaps(ax, baseline, series, L_lo, L_hi, ref=10.0):
    Ls = np.linspace(L_lo, L_hi, N_GRID_GAP)
    for runs, key in series:
        label, colour = STYLE[key]
        mean, sd, seeds = de.gap_curve(runs, baseline, Ls)
        ax.plot(Ls, 100 * mean, color=colour, lw=2, label=f"{label} (n={len(seeds)})")
        ax.fill_between(Ls, 100 * (mean - sd), 100 * (mean + sd), color=colour, alpha=0.25, lw=0)
    if ref is not None:
        ax.axhline(ref, color="grey", ls="--", lw=0.8, zorder=0)
    ax.axhline(0, color="black", lw=0.8, zorder=0)
    ax.invert_xaxis()                       # training progresses left -> right
    ax.set_xlabel("Target validation loss $L$")
    ax.set_ylabel("Data saved vs sinusoidal (%)")
    ax.legend(frameon=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote figures/" + name)


def rope_sin(df, cond):
    return (de.runs_by_seed(df, de.CONDITIONS[cond], "rope"),
            de.runs_by_seed(df, de.CONDITIONS[cond], "sinusoidal"))


# ---------------------------------------------------------------- figures
def fig1(df):
    rope, sin = rope_sin(df, "d512 EN-FR")
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_curves(ax, [(rope, "rope"), (sin, "sinusoidal")])
    ax.set_title("Validation loss, RoPE vs sinusoidal (d=512, En-Fr)")
    save(fig, "fig1a_loss_curves_d512.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_gaps(ax, sin, [(rope, "rope")], de.common_floor(rope, sin), 3.40)
    ax.set_ylabel("Data saved by RoPE vs sinusoidal (%)")
    ax.set_title("Data saved by RoPE across training (d=512, En-Fr)")
    save(fig, "fig1b_data_saved_d512.png")


def forest(df, band_width, name, title):
    results = []
    for group, label, cond in de.FOREST_ROWS:
        rope, sin = rope_sin(df, cond)
        results.append((group, label, de.forest_row(rope, sin, band_width=band_width)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y, ticks, labels, last = 0.0, [], [], None
    for group, label, r in results:
        if group != last:
            y += 0.9
            ax.text(-1.2, y, group, fontweight="bold", fontsize=9, va="center")
            y += 0.7
            last = group
        ax.plot([r["lo"], r["hi"]], [y, y], color="black", lw=1.5)
        ax.plot([r["estimate"]], [y], "o", color=STYLE["rope"][1], ms=8, zorder=5)
        ax.text(r["hi"] + 0.5, y, f'{r["estimate"]:.1f}%  [{r["lo"]:.1f}, {r["hi"]:.1f}]',
                va="center", fontsize=8)
        ticks.append(y)
        labels.append(f'{label}\n$n$={r["n"]}, $L\\in$[{r["L_lo"]:.2f}, {r["L_hi"]:.2f}]')
        y += 1.0
    ax.invert_yaxis()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(10, color="grey", ls="--", lw=0.8, zorder=0)
    ax.axvline(0, color="black", lw=0.8, zorder=0)
    ax.set_xlim(-1.5, 26)
    ax.set_xlabel("Data saved by RoPE (%)")
    ax.set_title(title)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, name)


def fig2(df):
    forest(df, 0.25, "fig2_forest.png", "Data saved by RoPE over sinusoidal, five configurations")


def fig3(df):
    base, mech = de.CONDITIONS["d512 EN-FR"], de.CONDITIONS["d512 mechanism"]
    series = [(de.runs_by_seed(df, base, "rope"), "rope"),
              (de.runs_by_seed(df, mech, "t5"), "t5"),
              (de.runs_by_seed(df, mech, "learned"), "learned"),
              (de.runs_by_seed(df, base, "sinusoidal"), "sinusoidal"),
              (de.runs_by_seed(df, mech, "nope"), "nope")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_curves(ax, series)
    ax.set_title("All five encodings, d=512 (y-axis cropped)")
    save(fig, "fig3_all_encodings_d512.png")


def fig4(df):
    base, mech = de.CONDITIONS["d512 EN-FR"], de.CONDITIONS["d512 mechanism"]
    sin, rope = de.runs_by_seed(df, base, "sinusoidal"), de.runs_by_seed(df, base, "rope")
    t5, learned = de.runs_by_seed(df, mech, "t5"), de.runs_by_seed(df, mech, "learned")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_gaps(ax, sin, [(rope, "rope"), (t5, "t5"), (learned, "learned")],
              de.common_floor(sin, rope, t5, learned), 3.40, ref=None)
    ax.set_title("RoPE vs T5 bias vs learned absolute (d=512)")
    save(fig, "fig4a_mechanism_d512.png")

    d256 = de.CONDITIONS["d256 different LR"]
    sin, rope = de.runs_by_seed(df, d256, "sinusoidal"), de.runs_by_seed(df, d256, "rope")
    t5 = de.runs_by_seed(df, de.CONDITIONS["d256 t5"], "t5")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_gaps(ax, sin, [(rope, "rope"), (t5, "t5")], de.common_floor(sin, rope, t5), 3.60, ref=None)
    ax.set_title("RoPE vs T5 bias (d=256)")
    save(fig, "fig4b_mechanism_d256.png")


def appendix_gap(df, cond, name, title):
    rope, sin = rope_sin(df, cond)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_gaps(ax, sin, [(rope, "rope")], de.common_floor(rope, sin), 3.40)
    ax.set_ylabel("Data saved by RoPE vs sinusoidal (%)")
    ax.set_title(title)
    save(fig, name)


def fig5_6(df):
    appendix_gap(df, "d256 different LR", "fig5a_data_saved_d256_higher_lr.png", "d=256, schedule's own (higher) LR")
    appendix_gap(df, "d256 matched LR", "fig5b_data_saved_d256_matched_lr.png", "d=256, LR matched to d=512")
    appendix_gap(df, "d512 modern arch", "fig6a_data_saved_modern_arch.png", "d=512, modern architecture")
    appendix_gap(df, "d512 DE-EN", "fig6b_data_saved_de_en.png", "d=512, De-En")


def fig7(df):
    forest(df, 0.15, "fig7a_forest_w015.png", "Forest plot, band width w=0.15")
    forest(df, 0.35, "fig7b_forest_w035.png", "Forest plot, band width w=0.35")


if __name__ == "__main__":
    plt.rcParams.update({"figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    OUT.mkdir(exist_ok=True)
    df = de.load_curves()
    for f in (fig1, fig2, fig3, fig4, fig5_6, fig7):
        f(df)
