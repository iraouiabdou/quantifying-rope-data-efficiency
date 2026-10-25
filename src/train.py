# train.py
import random
import torch
import torch.nn as nn

from config import tokenizer_path, build_tokenizer, clean_output
from dataset import TranslationDataset, make_collate_fn, LengthBatchSampler, _ds_from_blob
from model import Transformer, greedy_decode

from tqdm import tqdm
from datasets import load_dataset
from itertools import islice
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
import math

import pickle, hashlib, gc, time
from pathlib import Path

_DS_MEMO = {}

def _ds_key(cfg):
  return "|".join(str(cfg[k]) for k in
    ["datasource","dataset_config","lang_src","lang_tgt",
     "shuffle_seed","num_pairs","vocab_size","max_seq"])

def _ds_cache_file(cfg):
  h = hashlib.md5(_ds_key(cfg).encode()).hexdigest()[:12]
  d = Path(cfg.get("cache_dir", "."))
  d.mkdir(parents=True, exist_ok=True)
  return d / f"ds_{h}.pkl"

def load_pairs(cfg):
  stream = load_dataset(cfg["datasource"], cfg["dataset_config"], split = "train", streaming = True)
  stream = stream.shuffle(seed = cfg["shuffle_seed"], buffer_size = 50000)
  train_rows = list(tqdm(islice(stream, cfg["num_pairs"]),
                           total=cfg["num_pairs"], desc="loading train", unit="pair"))
  # Hold out the first 1% of the shuffled stream as validation (40,400 pairs).
  # NOTE: this is what produced every result. Do NOT switch to the
  # official WMT14 "validation" split (newstest2013): absolute losses would change.
  k = len(train_rows) // 100
  val_rows, train_rows = train_rows[:k], train_rows[k:]
  return train_rows, val_rows


def get_or_build_tokenizer(cfg, rows, langs: list[str], name):
  path = tokenizer_path(cfg, name)
  if path.exists():
    return Tokenizer.from_file(str(path))
  tok, trainer = build_tokenizer(cfg["vocab_size"], ["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
  tok.train_from_iterator((row["translation"][l] for row in rows for l in langs), trainer)
  tok.save(str(path))
  return tok


def get_ds(cfg):
  src, tgt = cfg["lang_src"], cfg["lang_tgt"]
  key, path, tok_path = _ds_key(cfg), _ds_cache_file(cfg), tokenizer_path(cfg, "shared")

  if key in _DS_MEMO:
    train_ds, val_ds, tok = _DS_MEMO[key]
    print("dataset: memory cache hit")

  elif path.exists() and tok_path.exists():
    print(f"dataset: loading {path}")
    t0 = time.time()
    with open(path, "rb") as f:
      blob = pickle.load(f)
    tok = Tokenizer.from_file(str(tok_path))
    train_ds = _ds_from_blob(tok, *blob["train"])
    val_ds   = _ds_from_blob(tok, *blob["val"])
    _DS_MEMO[key] = (train_ds, val_ds, tok)
    print(f"dataset: loaded in {time.time()-t0:.0f}s")

  else:
    print("dataset: cold build (~15 min, happens once)")
    train_rows, val_rows = load_pairs(cfg)
    tok = get_or_build_tokenizer(cfg, train_rows, [src, tgt], "shared")
    train_ds = TranslationDataset(tok, train_rows, src, tgt, cfg["max_seq"])
    val_ds   = TranslationDataset(tok, val_rows,   src, tgt, cfg["max_seq"])
    del train_rows, val_rows; gc.collect()

    train_ds.samples = [("", "", s, t) for _, _, s, t in train_ds.samples]

    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
      pickle.dump({"train": (train_ds.samples, train_ds.lengths),
                   "val":   (val_ds.samples,   val_ds.lengths)}, f, protocol=5)
    tmp.rename(path)
    _DS_MEMO[key] = (train_ds, val_ds, tok)
    print(f"dataset: cached to {path}")

  print(f"Vocab {tok.get_vocab_size()} | train {len(train_ds)} val {len(val_ds)}")
  args = dict(collate_fn=make_collate_fn(tok.token_to_id("[PAD]")),
              num_workers=cfg['num_workers'], pin_memory=True,
              persistent_workers=True, prefetch_factor=cfg["prefetch_factor"])
  train_dl = DataLoader(train_ds, batch_sampler=LengthBatchSampler(train_ds.lengths, cfg["batch_size"]), **args)
  val_dl   = DataLoader(val_ds,   batch_sampler=LengthBatchSampler(val_ds.lengths, cfg["batch_size"], shuffle=False), **args)
  return train_dl, val_dl, tok


def ids_to_text(row, tok, sos, eos):
  ids = row.tolist()
  ids = ids[1:] if ids[0] == sos else ids
  if eos in ids:
    ids = ids[:ids.index(eos)]
  return clean_output(tok.decode(ids))

@torch.no_grad()
def run_validation_loss(model, val_dl, eval_loss_fn, pad, vocab, device):
  model.eval()
  total_loss, total_tokens = 0.0, 0
  for batch in val_dl:
    enc = batch["enc_input"].to(device, non_blocking=True)
    dec = batch["dec_input"].to(device, non_blocking=True)
    enc_mask = batch["enc_mask"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
      logits = model(enc, enc_mask, dec)
      loss = eval_loss_fn(logits.view(-1, vocab), label.view(-1))

    total_loss += loss.item()
    total_tokens += (label != pad).sum().item()

  return total_loss / max(total_tokens, 1)

@torch.no_grad()
def run_validation(model, val_dl, tok, cfg, device):
  model.eval()
  sos, eos, pad = (tok.token_to_id(t) for t in ("[SOS]", "[EOS]", "[PAD]"))
  expected = []
  predicted = []
  sources = []

  for batch in val_dl:
    enc = batch["enc_input"].to(device)
    enc_mask = batch["enc_mask"].to(device)

    rows = greedy_decode(model, enc, enc_mask, sos, eos, pad, cfg["max_seq"])
    for row, src_t, tgt_t in zip(rows, batch["src_txt"], batch["tgt_txt"]):
      predicted.append(ids_to_text(row, tok, sos, eos))
      expected.append(clean_output(tgt_t))
      sources.append(src_t)

    if cfg["validation_size"] and len(predicted) >= cfg["validation_size"]:
      break

  for i in random.sample(range(len(predicted)),
                         min(cfg["num_validation_examples"], len(predicted))):
    print(f"\nSOURCE:    {sources[i]}\n"
          f"TARGET:    {expected[i]}\n"
          f"PREDICTED: {predicted[i]}")

  bleu = torchmetrics.text.BLEUScore()(predicted, [[e] for e in expected]).item()
  print(f"\nVALIDATION (greedy) over {len(predicted)} sentences | BLEU {bleu:.3f}")
  return bleu


def set_seed(seed):
  import numpy as np
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def train_model(cfg):
  assert torch.cuda.is_available(), "this codebase is CUDA-only"
  torch.backends.cudnn.benchmark = True
  device = torch.device("cuda")
  torch.set_float32_matmul_precision("high")


  train_dl, val_dl, tok = get_ds(cfg)
  total_steps = len(train_dl)
  vocab, accum = tok.get_vocab_size(), max(1, cfg["grad_accum_steps"])

  cfg["warmup_steps"] = max(1, round(0.1 * total_steps/accum))
  set_seed(cfg["seed"])

  model = Transformer(cfg["d_model"], cfg["h"], cfg["N"],
                      tok.get_vocab_size(), cfg["max_seq"],
                      dropout=cfg["dropout"],
                      pos_encoding=cfg["pos_encoding"]).to(device)
  raw_model = model
  print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

  opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=cfg["betas"], eps=cfg["eps"], fused=True)
  d, w, k = cfg["d_model"], cfg["warmup_steps"], cfg["lr_scale"]
  sched = LambdaLR(opt, lambda s: k * d ** -0.5 * min(max(s, 1) ** -0.5, max(s, 1) * w ** -1.5))

  if cfg["use_compile"]:
    model = torch.compile(model, dynamic=True)

  pad_id = tok.token_to_id("[PAD]")
  loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id,
                                label_smoothing=cfg["label_smoothing"])
  eval_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")


  for epoch in range(cfg["num_epochs"]):
    model.train()
    opt.zero_grad(set_to_none=True)
    it = tqdm(train_dl, desc=f"Epoch {epoch:02d}")
    ema, gn = None, 0.0
    seen = 0
    next_val, next_bleu = cfg["val_loss_every_pairs"], cfg["bleu_every_pairs"]
    for i, batch in enumerate(it):
      enc = batch["enc_input"].to(device, non_blocking=True)
      dec = batch["dec_input"].to(device, non_blocking=True)
      enc_mask = batch["enc_mask"].to(device, non_blocking=True)
      label = batch["label"].to(device, non_blocking=True)

      with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(enc, enc_mask, dec)
        loss = loss_fn(logits.view(-1, vocab), label.view(-1)) / accum

      loss.backward()
      if (i + 1) % accum == 0:
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()

      cur = loss.item() * accum
      ema = cur if ema is None else 0.98 * ema + 0.02 * cur
      it.set_postfix(avg=f"{ema:6.3f}", gn=f"{gn:5.2f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
      seen += enc.size(0)
      if seen >= next_val or seen >= next_bleu:
        if seen >= next_val:
          next_val += cfg["val_loss_every_pairs"]
          vl = run_validation_loss(raw_model, val_dl, eval_loss_fn, pad_id, vocab, device)
          it.write(f"[{seen:,} pairs] val loss {vl:6.3f} | val ppl {math.exp(vl):8.2f}")
        if seen >= next_bleu:
          next_bleu += cfg["bleu_every_pairs"]
          run_validation(raw_model, val_dl, tok, cfg, device)
        model.train()
    val_loss = run_validation_loss(raw_model, val_dl, eval_loss_fn, pad_id, vocab, device)
    print(f"Epoch {epoch:02d} | train {ema:6.3f} | val loss {val_loss:6.3f}"
          f"| val ppl {math.exp(val_loss):8.2f}")


    run_validation(raw_model, val_dl, tok, cfg, device)
    model.train()

  del model, raw_model, opt, sched, loss_fn, eval_loss_fn
  del train_dl, val_dl
  if cfg["use_compile"]:
    torch._dynamo.reset()
  gc.collect()
  torch.cuda.empty_cache()
