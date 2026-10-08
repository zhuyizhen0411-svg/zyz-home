# -*- coding: utf-8 -*-
"""本地向量化：Qwen3-Embedding-0.6B（1024 维），直接用 transformers 前向。

不依赖 sentence-transformers —— 实测在本机型上 ST 的 encode 会长时间无响应，
而 transformers 直接前向单次 <0.1s，更可控。

用法：
    embedder = Embedder()
    vec = embedder.encode(["文本1", "文本2"])   # shape (N, 1024)，已做 L2 归一化
"""

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
MAX_LEN = 1024


def pick_device():
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class Embedder:
    def __init__(self, model_name=MODEL, device=None, max_len=MAX_LEN):
        self.device = device or pick_device()
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts, batch_size=32, normalize=True, log_every=0):
        """返回 float32 ndarray (N, dim)。按长度排序再还原顺序，减少 padding 开销。"""
        import time

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        t0 = time.time()
        for bi, s in enumerate(range(0, len(order), batch_size)):
            if log_every and bi % log_every == 0:
                el = time.time() - t0
                rate = (s / el) if el > 0 else 0
                eta = (len(order) - s) / rate if rate > 0 else 0
                print(
                    f"  [embed] {s}/{len(order)}  {rate:.1f} chunks/s  ETA {eta/60:.1f} min",
                    flush=True,
                )
            idx = order[s : s + batch_size]
            batch = [texts[i] for i in idx]
            enc = self.tok(
                batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            arr = pooled.float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = arr[j]
        return out
