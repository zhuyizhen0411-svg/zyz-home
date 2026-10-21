# -*- coding: utf-8 -*-
"""建索引：BM25（jieba）+ 向量（Qwen3-Embedding-0.6B）。

分两步，BM25 先建（快），向量后建（慢，支持断点续跑）：
    python 代码/build_index.py --only bm25
    python 代码/build_index.py --only vectors

产物（index/）：
    元数据.json       每块元数据：公司/年份/报告类型/章节/页码
    文本.json        每块参与检索与展示的文本
    词频索引.pkl      BM25 倒排索引
    向量.npy          (N, 1024) float32，已 L2 归一化
    向量-断点临时.npy 向量化过程中的断点文件（每 2000 块落盘一次）

向量化在本机（Apple GPU / MPS）上的实测：
    fp32 + max_len 512 + bs 32  -> 2.8 块/s（太慢，弃用）
    fp16 + max_len 320 + bs 128 -> 12 块/s
因此默认用 fp16。按文本长度排序后再分批，减少 padding 开销。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 词频检索 import BM25, get_tokenize  # noqa: E402

CHUNKS = os.path.join(ROOT, "数据", "文本块.jsonl")
IDX = os.path.join(ROOT, "检索索引")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
PARTIAL = os.path.join(IDX, "向量-断点临时.npy")
ORDER = os.path.join(IDX, "向量顺序.npy")


def load_chunks():
    meta, texts = [], []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            meta.append({
                "id": c["id"], "doc_id": c["doc_id"], "code": c["code"],
                "company": c["company"], "year": c["year"], "rtype": c["rtype"],
                "section": c["section"], "page": c["page"],
                "block_type": c["block_type"], "n_chars": c["n_chars"],
            })
            texts.append(c["embed_text"])
    return meta, texts


def build_bm25(texts):
    t = time.time()
    tok = get_tokenize()
    bm = BM25().fit([tok(t) for t in texts])
    bm.save(os.path.join(IDX, "词频索引.pkl"))
    print(f"[bm25] 完成，词表 {len(bm.df)} 词，用时 {(time.time()-t)/60:.1f} min")


def build_vectors(texts, batch_size=128, max_len=320, device="mps", limit=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    n = len(texts) if limit is None else min(limit, len(texts))
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float16).to(device).eval()
    dim = model.config.hidden_size

    # 按长度排序分批（确定性顺序，便于断点续跑）
    if os.path.exists(ORDER):
        order = np.load(ORDER).tolist()
        assert len(order) >= n, "已存在的排序文件与当前语料不一致"
        order = order[:n]
    else:
        order = sorted(range(n), key=lambda i: len(texts[i]))
        np.save(ORDER, np.array(order))

    done, parts = 0, []
    if os.path.exists(PARTIAL):
        arr = np.load(PARTIAL)
        done = arr.shape[0]
        parts = [arr]
        print(f"[vec] 发现断点文件，已完成 {done}/{n}，继续", flush=True)
    elif os.path.exists(os.path.join(IDX, "向量.npy")):
        print("[vec] 向量.npy 已存在，跳过（如需重建请先删除）")
        return

    t0 = time.time()
    while done < n:
        idxs = order[done:done + batch_size]
        batch = [texts[i] for i in idxs]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        parts.append(pooled.float().cpu().numpy())
        done += len(idxs)
        if done % 2000 < batch_size:
            np.save(PARTIAL, np.concatenate(parts, axis=0))
            el = time.time() - t0
            rate = done / el
            print(f"[vec] {done}/{n}  {rate:.1f} 块/s  ETA {(n-done)/rate/60:.1f} min", flush=True)

    arr = np.concatenate(parts, axis=0)
    out = np.zeros((n, dim), dtype=np.float32)
    out[np.array(order[:n])] = arr
    np.save(os.path.join(IDX, "向量.npy"), out)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)
    print(f"[vec] 完成 {out.shape}，用时 {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "bm25", "vectors"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 块，用于快速验证")
    a = ap.parse_args()

    os.makedirs(IDX, exist_ok=True)
    meta, texts = load_chunks()
    print(f"语料：{len(texts)} 块 / {len({m['doc_id'] for m in meta})} 份报告")

    if a.only in ("all", "bm25"):
        with open(os.path.join(IDX, "元数据.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        with open(os.path.join(IDX, "文本.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        build_bm25(texts)

    if a.only in ("all", "vectors"):
        build_vectors(texts, batch_size=a.batch_size, max_len=a.max_len,
                      device=a.device, limit=a.limit)


if __name__ == "__main__":
    main()
