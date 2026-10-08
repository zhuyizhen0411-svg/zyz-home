# -*- coding: utf-8 -*-
"""把 84 份年报/半年报 PDF 抽成「页级结构化文本」。

- 正文：pdfplumber 抽取，并剔除被识别为表格的区域，避免与表格内容重复；
- 表格：find_tables() 拿几何位置 + 二维内容，还原成「单元格 | 单元格」文本行；
        单元格内部空格一律去掉（中文财报无词间空格，否则数字会被切碎）；
        第一列被漏识别时（只剩数字）用几何位置把行标签捞回来；
- 章节：每页识别「第X节 XX」，取标题后剩余文字最多的那个（页尾过渡标题不算）。

输出: 数据/页级文本.jsonl（每行一页）
"""

import json
import os
import re
import sys
from multiprocessing import Pool

import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_OUT = os.path.join(ROOT, "数据", "页级文本.jsonl")

SECTION_HEAD_RE = re.compile(r"^\s*(第[一二三四五六七八九十]+节)\s*([^\n]{0,40})", re.M)
MIN_SECTION_TAIL = 80   # 「第X节」标题后至少还有这么多字，才算这一页真正属于该节
NUM_RE = re.compile(r"[\d,，.]+")


def pick_section(text, fallback):
    if not text:
        return fallback
    best, best_tail = None, -1
    for m in SECTION_HEAD_RE.finditer(text):
        tail = len(text) - m.end()
        if tail > best_tail:
            best, best_tail = m, tail
    if best is None or best_tail < MIN_SECTION_TAIL:
        return fallback
    title = best.group(1) + best.group(2)
    # 目录页里「第一节 释义 ...... 50」带一长串点线，会把章节名污染掉
    title = re.split(r"\.{3,}|·{3,}|．{3,}|\s{3,}", title)[0]
    return title.strip()[:60]


def clean_cell(c):
    """中文财报表格没有词间空格，单元格内换行/空格都是 PDF 排版造成的，必须去掉，
    否则数字会被切成「4,303,837,98 0.45」，检索匹配不到、模型也会读错。"""
    if c is None:
        return ""
    c = str(c).replace("\n", " ").replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"\s+", "", c).strip()


def render_table(table):
    rows = []
    for r in table:
        cells = [clean_cell(c) for c in r]
        if not any(cells):
            continue
        rows.append(cells)
    if not rows:
        return None
    return "\n".join(" | ".join(c) for c in rows)


def inside_any(obj, boxes, pad=1.5):
    x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
    for (bx0, btop, bx1, bbot) in boxes:
        if x0 >= bx0 - pad and x1 <= bx1 + pad and top >= btop - pad and bottom <= bbot + pad:
            return True
    return False


def norm_cell(c):
    return re.sub(r"\s+", "", str(c or ""))


def repair_missing_labels(table, page_words):
    """部分 PDF 的表格第一列（项目名）会被漏掉，只剩右边数字：用几何位置把标签捞回来。"""
    rows = table.extract()
    if len(rows) < 2:
        return rows
    body = rows[1:]
    empty = sum(1 for r in body if not str(r[0] or "").strip())
    if empty < max(2, len(body) * 0.6):
        return rows
    try:
        row_boxes = [r.bbox for r in table.rows]
    except Exception:  # noqa: BLE001
        return rows
    if len(row_boxes) != len(rows):
        return rows
    for i in range(1, len(rows)):
        if str(rows[i][0] or "").strip():
            continue
        x0, top, x1, bottom = row_boxes[i]
        ws = [w for w in page_words if top <= (w["top"] + w["bottom"]) / 2 <= bottom and w["x0"] < x1]
        nums = [w for w in ws if NUM_RE.fullmatch(w["text"].replace("%", ""))]
        if not nums:
            continue
        split = min(w["x0"] for w in nums)
        labels = [w for w in ws if w["x1"] < split and not NUM_RE.fullmatch(w["text"].replace("%", ""))]
        if labels:
            lab = " ".join(w["text"] for w in sorted(labels, key=lambda w: w["x0"])).strip()
            if lab:
                lab_n = norm_cell(lab)
                rest = ["" if norm_cell(c) == lab_n else c for c in rows[i][1:]]
                rows[i] = [lab] + rest
    return rows


def extract_pdf(task):
    """task: manifest 里的一条记录（含 path/name/code/year/rtype）"""
    out = []
    with pdfplumber.open(task["path"]) as pdf:
        total = len(pdf.pages)
        section = "封面"
        for pno, page in enumerate(pdf.pages, start=1):
            boxes, rendered = [], []
            try:
                found = page.find_tables()
                boxes = [tuple(t.bbox) for t in found]
                words = page.extract_words()
                for t in found:
                    txt = render_table(repair_missing_labels(t, words))
                    if txt:
                        rendered.append(txt)
            except Exception:  # noqa: BLE001
                boxes, rendered = [], []

            try:
                if boxes:
                    keep = lambda o, _b=boxes: not inside_any(o, _b)  # noqa: E731
                    text = (page.filter(keep).extract_text() or "")
                else:
                    text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                text = page.extract_text() or ""

            section = pick_section(text, section)
            out.append(
                {
                    "doc_id": f"{task['code']}_{task['year']}_{task['rtype']}",
                    "code": task["code"],
                    "company": task["name"],
                    "year": task["year"],
                    "rtype": task["rtype"],
                    "page": pno,
                    "total_pages": total,
                    "section": section,
                    "text": text.strip(),
                    "tables": rendered,
                }
            )
    return out


def main():
    manifest = json.load(open(os.path.join(ROOT, "数据", "财报原文", "manifest.json"), encoding="utf-8"))
    tasks = [m for m in manifest if m.get("status") in ("ok", "skip", "warning")]
    os.makedirs(os.path.dirname(PAGES_OUT), exist_ok=True)

    print(f"开始抽取 {len(tasks)} 份 PDF（多进程）...")
    all_pages = []
    with Pool(processes=8) as pool:
        for i, pages in enumerate(pool.imap_unordered(extract_pdf, tasks), 1):
            all_pages.extend(pages)
            tag = f"{pages[0]['company']}{pages[0]['year']}{pages[0]['rtype']}" if pages else "?"
            print(f"  ({i}/{len(tasks)}) {tag} 抽取 {len(pages)} 页", flush=True)

    all_pages.sort(key=lambda p: (p["doc_id"], p["page"]))
    with open(PAGES_OUT, "w", encoding="utf-8") as f:
        for p in all_pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_tab = sum(len(p["tables"]) for p in all_pages)
    n_char = sum(len(p["text"]) for p in all_pages)
    print(f"\n共 {len(all_pages)} 页，{n_tab} 张表，正文字数 {n_char:,}")
    print(f"写入 {PAGES_OUT}")


if __name__ == "__main__":
    main()
