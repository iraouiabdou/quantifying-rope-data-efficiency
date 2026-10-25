# config.py

from pathlib import Path
import re as _re

def get_config():
  return {
      "datasource": "wmt/wmt14",
      "dataset_config": "fr-en",
      "lang_src": "en",
      "lang_tgt": "fr",
      "shuffle_seed": 0,
      "vocab_size": 32000,
      "max_seq": 300,
      "num_workers": 8,
      "prefetch_factor": 4,
      "batch_size": 256,
      "num_validation_examples": 3,
      "validation_size": 3000,    # BLEU is computed on the first 3,072 held-out sentences (see README)
      "d_model": 512,
      "h": 8,
      "N": 6,
      "dropout": 0.0,
      "betas": (0.9, 0.98),
      "eps": 1e-9,
      "lr_scale": 1.0,           # multiplies the Noam schedule; 0.7071 gives d=256 the d=512 peak LR
      "use_compile": False,
      "label_smoothing": 0.1,
      "grad_accum_steps": 1,
      "num_epochs": 1,
      "num_pairs": 4_040_000,
      "pos_encoding": "sinusoidal",
      "val_loss_every_pairs": 100_000,
      "bleu_every_pairs": 5_000_000,
      "cache_dir": "./mt_cache",
      "seed": 0,
  }



def tokenizer_path(config, name: str) -> Path:
  slug = config["datasource"].split("/")[-1]
  d = Path(config.get("cache_dir", "."))
  d.mkdir(parents=True, exist_ok=True)
  return d / f"tokenizer_{slug}_{config['dataset_config']}_v{config['vocab_size']}_s{config['shuffle_seed']}_{name}.json"

def build_tokenizer(vocab_size, special_tokens):
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer

    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Split(pattern="'", behavior="merged_with_previous"),
    ])
    tok.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=2,
                               special_tokens=special_tokens, show_progress=False)
    return tok, trainer


_APOS = _re.compile(r"\s*'\s*")
_HYPH = _re.compile(r"\s+-\s+")
_PUNC = _re.compile(r"\s*([.,;:!?])")
_MULTI = _re.compile(r"\s+")

def clean_output(text: str) -> str:
  text = _APOS.sub("'", text)
  text = _HYPH.sub("-", text)
  text = _PUNC.sub(r"\1", text)
  return _MULTI.sub(" ", text).strip()
