"""
Rebuild data/curves.csv from the raw training logs in logs/.

    python analysis/parse_logs.py            # writes data/curves_from_logs.csv
    python analysis/parse_logs.py --check    # compare against data/curves.csv

Each log is one batch of runs (exp_id). A run starts at a line
"=== <variant> seed <s> ===" and prints "[N pairs] val loss X" at every
checkpoint, then "Epoch 00 | train ... | val loss X" and a final BLEU.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUN = re.compile(r"^=== (\w+) seed (\d+) ===", re.M)
CKPT = re.compile(r"\[([\d,]+) pairs\] val loss\s+([\d.]+)")
FINAL = re.compile(r"Epoch 00 \| train\s+[\d.]+ \| val loss\s+([\d.]+)")
PARAMS = re.compile(r"Parameters: ([\d,]+)")
BLEU = re.compile(r"BLEU ([\d.]+)")
TRAIN = re.compile(r"train (\d+) val (\d+)")


def parse(path):
    txt = path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    exp_id = path.stem
    starts = [(m.start(), m.group(1), int(m.group(2))) for m in RUN.finditer(txt)]
    rows = []
    for i, (start, variant, seed) in enumerate(starts):
        block = txt[start: starts[i + 1][0] if i + 1 < len(starts) else len(txt)]
        bleu = 100 * float(BLEU.search(block).group(1))
        n_train = int(TRAIN.search(block).group(1))
        for k, (pairs, loss) in enumerate(CKPT.findall(block), 1):
            rows.append(dict(exp_id=exp_id, run_id=f"{exp_id}_{variant}_s{seed}", variant=variant,
                             seed=seed, checkpoint=k, pairs_seen=int(pairs.replace(",", "")),
                             val_loss=float(loss), bleu=round(bleu, 1),
                             params=int(PARAMS.search(block).group(1).replace(",", "")),
                             n_train=n_train))
        fm = FINAL.search(block)
        rows.append(dict(rows[-1], checkpoint=rows[-1]["checkpoint"] + 1,
                         pairs_seen=n_train, val_loss=float(fm.group(1))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    df = pd.DataFrame([r for p in sorted((ROOT / "logs").glob("exp*.txt")) for r in parse(p)])
    print(f"Parsed {len(df)} checkpoints from {df.run_id.nunique()} runs in {df.exp_id.nunique()} logs.")
    if not args.check:
        df.to_csv(ROOT / "data" / "curves_from_logs.csv", index=False)
    else:
        ref = pd.read_csv(ROOT / "data" / "curves.csv")
        key = ["run_id", "pairs_seen"]
        m = ref.merge(df, on=key, how="outer", suffixes=("_csv", "_log"), indicator=True)
        print(m["_merge"].value_counts().to_string())
        both = m[m["_merge"] == "both"]
        print("max |val_loss diff|:", (both.val_loss_csv - both.val_loss_log).abs().max())
        print("max |bleu diff|:    ", (both.bleu_csv - both.bleu_log).abs().max())
