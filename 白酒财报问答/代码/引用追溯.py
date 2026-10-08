# -*- coding: utf-8 -*-
"""出处追溯：让回答里的每个数字都能落回原始财报的具体文本块。

做三件事：
  1. 把回答按句切开，找出每句引用的 [n]，映射成「公司 / 年份 / 报告类型 / 页码 / 文本块编号」
  2. 逐句校验：句子里出现的每个数字，是否真能在被引用的文本块原文里找到
     —— 找不到就标记为 unverified，明确报「材料不足」，不允许蒙
  3. 按文本块编号取回原文全文 + 对应的原始 PDF 文件路径，供人工复核

用法:
    python 代码/citations.py --chunk 12345        # 查看某个文本块的原文与出处
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "数据")
IDX = os.path.join(ROOT, "检索索引")

# 引用标记：[1] / [1][3] / [1,3] / [1、3]
CITE_RE = re.compile(r"\[(\d[\d\s、,，]*)\]")
# 句子切分：句号、分号、问号、换行
SENT_SPLIT_RE = re.compile(r"(?<=[。；;！!？?\n])")
# 数字：千分位金额 / 小数 / 百分数，可带 亿元 万元 千升 吨 % 等单位
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千升|吨|%|个百分点|元|股|家|人)?")

_ZH_DIGITS = str.maketrans("０１２３４５６７８９．％　", "0123456789.% ")


def _norm(s):
    """归一化：全角转半角、去千分位逗号与空格，便于子串比对。"""
    return s.translate(_ZH_DIGITS).replace(",", "").replace(" ", "").replace(" ", "")


def expand_number(raw):
    """把一个数字串展开成若干等价写法，覆盖「亿元 ↔ 元」这类单位换算。

    原文常写 16,883,810.25（万元），模型可能输出 1688.38（亿元）或 168838102500（元），
    直接子串比对会漏，所以这里把常见单位换算的等价串都生成出来。
    """
    s = _norm(raw)
    m = re.match(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return {s}
    try:
        val = float(m.group(1))
    except ValueError:
        return {s}
    out = {s, m.group(1)}
    if "亿" in raw:
        for mul in (1e8, 1e4):  # 亿元 → 元 / 万元
            v = val * mul
            out.add(_strip_zero(f"{v:.6f}"))
            out.add(_strip_zero(f"{v:.0f}"))
    elif "万" in raw and "亿元" not in raw:
        for mul in (1e4, 1e-4):  # 万元 → 元 / 亿元
            v = val * mul
            out.add(_strip_zero(f"{v:.6f}"))
            out.add(_strip_zero(f"{v:.0f}"))
    return {x for x in out if x}


def _strip_zero(s):
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


YEAR_ONLY_RE = re.compile(r"^(19|20)\d{2}$")


def extract_numbers(text):
    """从一句话里抽出需要追溯的数字（去重、保序）。

    两处要剔除，否则会误报「材料不足」：
      - 引用标记里的编号（[2] 的 2 不是事实数字）
      - 单独的年份（2025 年只是时间状语，不需要回原文核对）
    """
    stripped = CITE_RE.sub("　", text)  # 先剥掉 [n]，避免编号被当成数字
    out = []
    for m in NUM_RE.finditer(stripped):
        raw = m.group(0).strip()
        if not raw:
            continue
        core = re.match(r"(-?\d[\d,]*(?:\.\d+)?)", raw)
        core = core.group(1).replace(",", "") if core else raw
        if YEAR_ONLY_RE.match(core) and not re.search(r"[.%]|亿元|万元|千升|吨|元|股|家|人", raw):
            continue
        if raw not in out:
            out.append(raw)
    return out


def parse_cites(text):
    """抽出一句里的全部引用编号。"""
    ns = []
    for m in CITE_RE.finditer(text):
        for part in re.split(r"[\s、,，]+", m.group(1)):
            if part.isdigit():
                n = int(part)
                if n not in ns:
                    ns.append(n)
    return ns


# ---------------- 块与源文件 ----------------
_CHUNKS = None
_DOC_MAP = None


def _load():
    global _CHUNKS, _DOC_MAP
    if _CHUNKS is None:
        _CHUNKS = {}
        with open(os.path.join(DATA, "文本块.jsonl"), encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                _CHUNKS[int(c["id"])] = c
    if _DOC_MAP is None:
        _DOC_MAP = {}
        mp = os.path.join(DATA, "reports", "manifest.json")
        if os.path.exists(mp):
            for r in json.load(open(mp, encoding="utf-8")):
                _DOC_MAP[f"{r['code']}_{r['year']}_{r['rtype']}"] = {
                    "pdf_path": r.get("path"),
                    "pages": r.get("pages"),
                    "source_url": r.get("url"),
                    "source": r.get("source"),
                    "publish_date": r.get("publish_date"),
                }
    return _CHUNKS, _DOC_MAP


def get_chunk(chunk_id):
    """按文本块编号取回原文 + 完整出处（含原始 PDF 路径）。"""
    chunks, doc_map = _load()
    c = chunks.get(int(chunk_id))
    if not c:
        return None
    doc = doc_map.get(c["doc_id"], {})
    return {
        "chunk_id": int(c["id"]),
        "doc_id": c["doc_id"],
        "company": c["company"],
        "code": c["code"],
        "year": c["year"],
        "rtype": c["rtype"],
        "section": c["section"],
        "page": c["page"],
        "block_type": c["block_type"],
        "n_chars": c.get("n_chars"),
        "text": c["text"],
        "pdf_path": doc.get("pdf_path"),
        "pdf_pages": doc.get("pages"),
        "source_url": doc.get("source_url"),
        "source": doc.get("source"),
        "publish_date": doc.get("publish_date"),
    }


# ---------------- 回答校验 ----------------
def verify_answer(answer, hits):
    """校验回答：每句引用的块是谁、句子里的数字能不能在原文里找到。

    返回 dict：
      sentences   逐句结果（文本 / 引用 / 数字及其支撑情况 / 状态）
      citations   去重后的引用清单（含块编号与原文摘要）
      summary     统计 + 是否存在「材料不足」
    status 取值：
      ok            有引用且数字全部可追溯
      no_citation   句子含数字但没有任何引用标记
      bad_citation  引用编号超出召回范围
      unverified    有引用，但句内数字在被引块原文里找不到
      none          该句没有数字（纯叙述），不校验
    """
    by_rank = {h["rank"]: h for h in hits}
    sentences, used = [], []
    for raw in SENT_SPLIT_RE.split(answer or ""):
        s = raw.strip()
        if not s:
            continue
        cites = parse_cites(s)
        nums = extract_numbers(s)

        cite_objs = []
        for n in cites:
            h = by_rank.get(n)
            if not h:
                cite_objs.append({"n": n, "ok": False, "reason": "引用编号不在召回范围内"})
                continue
            c = {
                "n": n,
                "ok": True,
                "chunk_id": h["id"],
                "company": h["company"],
                "code": h["code"],
                "year": h["year"],
                "rtype": h["rtype"],
                "page": h["page"],
                "section": h["section"],
                "block_type": h["block_type"],
            }
            cite_objs.append(c)
            used.append(c)

        num_objs = []
        corpus = "\n".join(_norm(by_rank[n]["text"]) for n in cites if n in by_rank)
        for raw_num in nums:
            cands = expand_number(raw_num)
            hit = any(cand and cand in corpus for cand in cands) if corpus else False
            num_objs.append({
                "raw": raw_num,
                "matched": bool(hit),
                "candidates": sorted(cands)[:6],
            })

        if not nums:
            status = "none"
        elif not cites:
            status = "no_citation"
        elif any(not c["ok"] for c in cite_objs):
            status = "bad_citation"
        elif any(not x["matched"] for x in num_objs):
            status = "unverified"
        else:
            status = "ok"

        sentences.append({
            "text": s,
            "citations": cite_objs,
            "numbers": num_objs,
            "status": status,
        })

    # 去重后的引用清单
    seen, cite_list = set(), []
    for c in used:
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        h = by_rank[c["n"]]
        cite_list.append({
            "n": c["n"],
            "chunk_id": c["chunk_id"],
            "company": c["company"],
            "code": c["code"],
            "year": c["year"],
            "rtype": c["rtype"],
            "page": c["page"],
            "section": c["section"],
            "block_type": c["block_type"],
            "pdf_path": (get_chunk(c["chunk_id"]) or {}).get("pdf_path"),
            "excerpt": h["text"][:180],
        })

    bad = [s for s in sentences if s["status"] in ("no_citation", "bad_citation", "unverified")]
    return {
        "sentences": sentences,
        "citations": cite_list,
        "summary": {
            "n_sentences": len(sentences),
            "n_cited": len([s for s in sentences if s["status"] == "ok"]),
            "n_bad": len(bad),
            "n_chunks_cited": len(cite_list),
            "insufficient": bool(bad) or not cite_list,
        },
    }


def insufficient_note(ver):
    """把无法追溯的内容拼成一句明确的「材料不足」提示；没问题就返回空串。"""
    if not ver["summary"]["insufficient"]:
        return ""
    lines = []
    for s in ver["sentences"]:
        if s["status"] == "no_citation":
            lines.append(f"· 「{s['text'][:60]}」含数字但未标注出处")
        elif s["status"] == "bad_citation":
            lines.append(f"· 「{s['text'][:60]}」引用编号无效")
        elif s["status"] == "unverified":
            miss = [x["raw"] for x in s["numbers"] if not x["matched"]]
            lines.append(f"· 「{s['text'][:60]}」中的 {'、'.join(miss)} 未能在所引原文中定位")
    if not ver["citations"]:
        lines.append("· 回答未引用任何文本块")
    return "材料不足：以下数字无法追溯到原文，不予采信：\n" + "\n".join(lines)


def format_report(ver, with_excerpt=True):
    """把校验结果渲染成可读文本（CLI / 日志用）。"""
    out = []
    for i, s in enumerate(ver["sentences"], 1):
        mark = {"ok": "✓", "none": "·", "no_citation": "✗", "bad_citation": "✗", "unverified": "✗"}[s["status"]]
        cites = " ".join(
            f"[{c['n']}]#{c['chunk_id']} {c['company']}{c['year']}年{c['rtype']} p{c['page']}"
            for c in s["citations"] if c.get("ok")
        )
        nums = " ".join(
            f"{x['raw']}{'' if x['matched'] else '(未定位)'}" for x in s["numbers"]
        )
        out.append(f"{mark} {s['text']}")
        if cites:
            out.append(f"    出处: {cites}")
        if nums:
            out.append(f"    数字: {nums}")
    out.append("")
    out.append("引用块清单：")
    for c in ver["citations"]:
        out.append(
            f"  [{c['n']}] 块#{c['chunk_id']} {c['company']}（{c['code']}）｜{c['year']}年{c['rtype']}｜"
            f"{c['section']}｜第{c['page']}页｜{c['block_type']}"
        )
        if c.get("pdf_path"):
            out.append(f"        原文: {c['pdf_path']}  第 {c['page']} 页")
        if with_excerpt:
            out.append(f"        摘录: {c['excerpt'][:120]}")
    s = ver["summary"]
    out.append(f"\n统计：{s['n_sentences']} 句 / 可追溯 {s['n_cited']} 句 / 问题 {s['n_bad']} 句 / 引用 {s['n_chunks_cited']} 块")
    return "\n".join(out)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, help="查看某个文本块编号的原文与出处")
    a = ap.parse_args()
    if a.chunk is None:
        ap.error("请给出 --chunk")
    c = get_chunk(a.chunk)
    if not c:
        print(f"未找到块 #{a.chunk}")
        return
    print(f"块 #{c['chunk_id']}（doc={c['doc_id']}）")
    print(f"{c['company']}（{c['code']}）｜{c['year']}年{c['rtype']}｜{c['section']}｜第{c['page']}页｜{c['block_type']}")
    print(f"原始 PDF: {c['pdf_path']}（共 {c['pdf_pages']} 页，第 {c['page']} 页）")
    print(f"来源: {c['source']}  {c['source_url']}")
    print("-" * 60)
    print(c["text"])


if __name__ == "__main__":
    main()
