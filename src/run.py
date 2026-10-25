"""
Train one or more (positional encoding x seed) runs.

Each run prints a validation loss every 100K training pairs and a final BLEU.
Those printed logs are what `logs/` contains and what `data/curves.csv` was
parsed from.

Examples (the five configurations):

  # 1. d=512, En->Fr                          (exp1, exp4, exp12)
  python run.py --pe rope sinusoidal --seeds 0 1 2 3 4

  # 2. d=256, schedule's own (higher) peak LR (exp2, exp3)
  python run.py --pe rope sinusoidal --seeds 0 1 2 3 4 --d_model 256 --heads 4

  # 3. d=256, peak LR matched to d=512        (exp7, exp10, exp11)
  python run.py --pe rope sinusoidal --seeds 0 1 2 3 4 --d_model 256 --heads 4 --lr_scale 0.7071

  # 4. d=512, modern architecture             (exp8, exp9)
  #    -> not in this codebase yet, see README "Known gaps"

  # 5. d=512, En->De                          (exp6)
  python run.py --pe rope sinusoidal --seeds 0 1 2 --pair de-en --tgt de

  # Mechanism comparison at d=512             (exp5, exp13)
  python run.py --pe learned nope t5 --seeds 0 1 2
  # T5 at d=256                               (exp14)
  python run.py --pe t5 --seeds 0 1 2 --d_model 256 --heads 4

Requires a CUDA GPU. One d=512 run is ~15K steps (~50 min on the Colab GPU used).
The first call builds and caches the dataset + tokenizer (~15 min).
"""
import argparse

import torch

from config import get_config
from train import train_model


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--pe", nargs="+", default=["sinusoidal", "rope"],
                 choices=["sinusoidal", "rope", "learned", "t5", "nope"])
  p.add_argument("--seeds", nargs="+", type=int, default=[0])
  p.add_argument("--d_model", type=int, default=512)
  p.add_argument("--heads", type=int, default=8)
  p.add_argument("--lr_scale", type=float, default=1.0)
  p.add_argument("--pair", default="fr-en", help="WMT14 config: fr-en or de-en")
  p.add_argument("--tgt", default="fr", help="target language (source is always en)")
  p.add_argument("--cache_dir", default="./mt_cache")
  p.add_argument("--compile", action="store_true", help="torch.compile (used for the runs)")
  a = p.parse_args()

  for pe_kind in a.pe:
    for seed in a.seeds:
      cfg = get_config()
      cfg.update(pos_encoding=pe_kind, seed=seed, d_model=a.d_model, h=a.heads,
                 lr_scale=a.lr_scale, dataset_config=a.pair, lang_tgt=a.tgt,
                 cache_dir=a.cache_dir, use_compile=a.compile)
      print(f"\n=== {pe_kind} seed {seed} === GPU before: {torch.cuda.memory_allocated()/1e9:.2f} GB")
      train_model(cfg)
      print(f"GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
  main()
