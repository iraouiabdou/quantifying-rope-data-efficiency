# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class RelativePositionBias(nn.Module):
  def __init__(self, h, num_buckets=32, max_distance=128, bidirectional=True):
    super().__init__()
    self.num_buckets, self.max_distance = num_buckets, max_distance
    self.bidirectional = bidirectional
    self.emb = nn.Embedding(num_buckets, h)
    nn.init.normal_(self.emb.weight, std=0.02)

  @staticmethod
  def _bucket(rel, bidirectional, num_buckets, max_distance):
    # rel = k_pos - q_pos; rel > 0 means the key is in the future
    n = -rel
    ret = torch.zeros_like(n)
    if bidirectional:
      num_buckets //= 2
      ret = ret + (n < 0).long() * num_buckets   # separate halves for past/future
      n = n.abs()
    else:
      n = n.clamp(min=0)                          # future collapses to bucket 0
    max_exact = num_buckets // 2
    is_small = n < max_exact                      # exact bucket per distance
    large = max_exact + (torch.log(n.float().clamp(min=1) / max_exact)
                         / math.log(max_distance / max_exact)
                         * (num_buckets - max_exact)).long()
    return ret + torch.where(is_small, n, large.clamp(max=num_buckets - 1))

  def forward(self, q_len, k_len, device):
    q_pos = torch.arange(q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    b = self._bucket(k_pos - q_pos, self.bidirectional, self.num_buckets, self.max_distance)
    return self.emb(b).permute(2, 0, 1).unsqueeze(0)   # (1, h, q_len, k_len)


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float):
    super().__init__()
    self.w_q = nn.Linear(d_model, d_model, bias = False)
    self.w_k = nn.Linear(d_model, d_model, bias = False)
    self.w_v = nn.Linear(d_model, d_model, bias = False)
    self.w_o = nn.Linear(d_model, d_model, bias = False)
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model should be divisble by the number of heads"
    self.d_k = d_model // h
    self.dropout = dropout

  def split_heads(self, x):
    return x.reshape(x.shape[0], x.shape[1], self.h, self.d_k).swapaxes(1,2)

  def forward(self, q, k, v, mask=None, is_causal=False, rope=None, attn_bias=None):
    q, k, v = self.split_heads(self.w_q(q)), self.split_heads(self.w_k(k)), self.split_heads(self.w_v(v))
    if rope is not None:
      cos, sin = rope
      q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    Sq, Sk = q.shape[-2], k.shape[-2]

    if attn_bias is not None:
      bias = attn_bias.to(q.dtype)
      if is_causal:
        causal = torch.ones(Sq, Sk, device=q.device, dtype=torch.bool).tril(Sk - Sq)
        mask = causal if mask is None else (mask & causal)
      if mask is not None:
        bias = torch.where(mask, bias, torch.finfo(q.dtype).min)  # broadcasts (B,1,1,Sk) with (1,h,Sq,Sk)
      mask, is_causal = bias, False

    x = F.scaled_dot_product_attention(q, k, v, mask, self.dropout if self.training else 0.0, is_causal)

    x = x.swapaxes(1, 2).reshape(x.shape[0], x.shape[2], self.d_model)
    return self.w_o(x)


class EncoderBlock(nn.Module):
  def __init__(self, d_model, h, dropout):
    super().__init__()
    self.self_mha = MultiHeadAttention(d_model, h, dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
    self.ln2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask, rope = None, attn_bias=None):
    # mask shape should be (B, 1, 1, S)
    h = self.ln1(x)
    x = self.dropout(self.self_mha(h, h, h, mask, rope = rope, attn_bias=attn_bias)) + x
    return self.dropout(self.ffn(self.ln2(x))) + x

class DecoderBlock(nn.Module):
  def __init__(self, d_model, h, dropout):
    super().__init__()
    self.self_causal_mha = MultiHeadAttention(d_model, h, dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.cross_mha = MultiHeadAttention(d_model, h, dropout)
    self.ln2 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
    self.ln3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, tgt, enc_out, src_mask, rope = None, attn_bias=None):
    h = self.ln1(tgt)
    x = self.dropout(self.self_causal_mha(h, h, h, None, True, rope = rope, attn_bias=attn_bias)) + tgt
    x = self.dropout(self.cross_mha(self.ln2(x), enc_out, enc_out, src_mask)) + x
    return self.dropout(self.ffn(self.ln3(x))) + x


def pe(d_model, max_seq):
  pos = torch.arange(0, max_seq).unsqueeze(1)
  i2 = torch.arange(0, d_model, 2)
  ang = pos * torch.exp(-i2 / d_model * math.log(10000.0))
  out = torch.zeros(max_seq, d_model)
  out[:, 0::2] = torch.sin(ang)
  out[:, 1::2] = torch.cos(ang)

  return out


def rope_cache(d_k, max_seq, base = 10000.0):
  # cos/sin tables, shape (max_seq, d_k // 2)
  inv_freq = base ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
  freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
  return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
  S, dt = x.shape[-2], x.dtype
  x = x.float()
  c = cos[:S].view(1, 1, S, -1)
  s = sin[:S].view(1, 1, S, -1)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2).to(dt)


class Transformer(nn.Module):
  def __init__(self, d_model, h, N, vocab_sz, max_seq, dropout, pos_encoding = "sinusoidal"):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_seq = max_seq
    self.d_model = d_model
    self.pos_encoding = pos_encoding
    self.embed = nn.Embedding(vocab_sz, d_model)
    self.encoder = nn.ModuleList([EncoderBlock(d_model, h, dropout) for _ in range(N)])
    self.decoder = nn.ModuleList([DecoderBlock(d_model, h, dropout) for _ in range(N)])
    self.enc_final_ln = nn.LayerNorm(d_model)
    self.dec_final_ln = nn.LayerNorm(d_model)
    self.lm_head = nn.Linear(d_model, vocab_sz, False)
    self.lm_head.weight = self.embed.weight
    self._init_weights()
    if pos_encoding == "rope":
      cos, sin = rope_cache(d_model // h, max_seq)
      self.register_buffer('rope_cos', cos)
      self.register_buffer('rope_sin', sin)
    elif pos_encoding == "sinusoidal":
      self.register_buffer('pe', pe(d_model, max_seq))
    elif pos_encoding == "nope":
      pass
    elif pos_encoding == "learned":
      self.pos_embed = nn.Embedding(max_seq, d_model)
      nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.7)
    elif pos_encoding == "t5":
      self.enc_rel_bias = RelativePositionBias(h, bidirectional=True)
      self.dec_rel_bias = RelativePositionBias(h, bidirectional=False)
    else:
      raise ValueError(f"Unknown positional encoding: {pos_encoding}")


  def _init_weights(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)

  @property
  def rope(self):
    return (self.rope_cos, self.rope_sin) if self.pos_encoding == "rope" else None

  def _embed(self, ids):
    x = self.embed(ids) * math.sqrt(self.d_model)
    if self.pos_encoding == "sinusoidal":
      x = x + self.pe[:ids.shape[-1], :]
    elif self.pos_encoding == "learned":
      pos = torch.arange(ids.shape[-1], device=ids.device)
      x = x + self.pos_embed(pos)
    return self.dropout(x)


  def encode(self, src, src_mask):
    x = self._embed(src)
    S = src.shape[-1]
    bias = self.enc_rel_bias(S, S, src.device) if self.pos_encoding=="t5" else None
    for encoder_block in self.encoder:
      x = encoder_block(x, src_mask, self.rope, bias)
    return self.enc_final_ln(x)

  def decode(self, memory, src_mask, tgt):
    x = self._embed(tgt)
    S = tgt.shape[-1]
    bias = self.dec_rel_bias(S, S, tgt.device) if self.pos_encoding=="t5" else None
    for decoder_block in self.decoder:
      x = decoder_block(x, memory, src_mask, self.rope, bias)
    return self.dec_final_ln(x)

  def proj(self, x):
    return self.lm_head(x)

  def forward(self, src, src_mask, tgt):
    assert tgt.shape[-1] <= self.max_seq and src.shape[-1] <= self.max_seq
    return self.proj(self.decode(self.encode(src, src_mask), src_mask, tgt))


@torch.inference_mode()
def greedy_decode(model, src, src_mask, sos, eos, pad, max_len):
  B, device = src.size(0), src.device
  memory = model.encode(src, src_mask)
  ys = torch.full((B, 1), sos, dtype=torch.long, device=device)
  finished = torch.zeros(B, dtype=torch.bool, device=device)
  for _ in range(max_len - 1):
    logits = model.proj(model.decode(memory, src_mask, ys)[:, -1])
    nxt = logits.argmax(-1).masked_fill(finished, pad)
    ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
    finished |= nxt == eos
    if bool(finished.all()):
      break
  return ys
