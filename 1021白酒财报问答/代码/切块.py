# -*- coding: utf-8 -*-
"""页级文本 → 文本块（chunk）。

规则：
- 正文按段落合并，长段再按句号切，目标 ~700 字，段间重叠 100 字；
- 表格整块保留，超过 900 字按行拆分，行内用「列 | 列」还原行列关系；
- 每个块都带上「公司｜年份+报告类型｜章节｜页码」前缀，既写进元数据也写进向量化文本。

输出: 数据/文本块.jsonl
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(ROOT, "数据", "页级文本.jsonl")
OUT = os.path.join(ROOT, "数据", "文本块.jsonl")

TARGET = 700          # 正文块目标长度
OVERLAP = 100         # 正文块重叠
MAX_TABLE = 900       # 表格块上限
MIN_KEEP = 40         # 过短的块丢弃

SENT_SPLIT = re.compile(r"(?<=[。；！？])")
DOT_PAGE = re.compile(r"^[\s.·．\-—]*$")
NOISE_SECTIONS = ("目录", "备查文件", "释义")
# 章节标签可能落在「第一节 释义」上，但页面其实带着核心财务数据表，这类页必须保留
KEEP_MARKERS = ("主要会计数据", "主要财务指标", "主要财务数据", "主要会计数据及财务指标")
MIN_PAGE_CHARS = 120
# 真正的财务数据页一定有大额数字/两位小数；目录页虽然也写着「主要财务指标」，但没有数字
FIG_RE = re.compile(r"\d{1,3}(?:,\d{3}){2,}|\d+\.\d{2}|\d+\.\d{1,2}%")


def page_blob(page):
    return (page.get("text") or "") + "".join(page.get("tables") or [])


def is_noise(page):
    blob = page_blob(page)
    if len(blob.strip()) < MIN_PAGE_CHARS:
        return True
    sec = page.get("section") or ""
    if any(s in sec for s in NOISE_SECTIONS):
        # 目录/释义/备查文件页默认丢弃；只有真的带财务数字（如「主要会计数据表」落在
        # 被标成释义的页上）才保留
        return not (any(k in blob for k in KEEP_MARKERS) and FIG_RE.search(blob))
    # 目录页：整页都是「xxx ...... 50」这类点线页码，没有实质内容
    if blob.count("...") + blob.count("···") + blob.count("．．") >= 5 and not FIG_RE.search(blob):
        return True
    return False


def head(page):
    return f"【{page['company']}｜{page['year']}年{page['rtype']}｜{page['section']}｜第{page['page']}页】"


def split_long(text, size, overlap):
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            tail = text[start:end]
            m = list(SENT_SPLIT.finditer(tail))
            if m and m[-1].end() > size * 0.5:
                end = start + m[-1].end()
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if len(p.strip()) >= MIN_KEEP]


def paragraph_blocks(text):
    paras = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n")]
    paras = [p for p in paras if p and not DOT_PAGE.match(p)]
    blocks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) <= TARGET:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                blocks.append(buf)
            buf = p
    if buf:
        blocks.append(buf)
    return blocks


def base_meta(page, cid):
    return {
        "id": cid,
        "doc_id": page["doc_id"],
        "code": page["code"],
        "company": page["company"],
        "year": page["year"],
        "rtype": page["rtype"],
        "section": page["section"],
        "page": page["page"],
    }


def main():
    chunks = []
    cid = 0
    with open(PAGES, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            if is_noise(page):
                continue
            prefix = head(page)

            for blk in paragraph_blocks(page.get("text") or ""):
                for piece in split_long(blk, TARGET, OVERLAP):
                    if len(piece) < MIN_KEEP:
                        continue
                    cid += 1
                    chunks.append({
                        **base_meta(page, cid),
                        "block_type": "text",
                        "embed_text": f"{prefix}\n{piece}",
                        "text": piece,
                        "n_chars": len(piece),
                    })

            for t in page.get("tables") or []:
                t = t.strip()
                if len(t) < MIN_KEEP:
                    continue
                rows = [r for r in t.split("\n") if r.strip()]
                if len(t) <= MAX_TABLE:
                    groups = [t]
                else:
                    groups, cur = [], ""
                    for r in rows:
                        if cur and len(cur) + len(r) + 1 > MAX_TABLE:
                            groups.append(cur)
                            cur = r
                        else:
                            cur = (cur + "\n" + r).strip()
                    if cur:
                        groups.append(cur)
                for g in groups:
                    if len(g) < MIN_KEEP:
                        continue
                    cid += 1
                    chunks.append({
                        **base_meta(page, cid),
                        "block_type": "table",
                        "embed_text": f"{prefix}\n[表格]\n{g}",
                        "text": f"[表格]\n{g}",
                        "n_chars": len(g),
                    })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_tab = sum(1 for c in chunks if c["block_type"] == "table")
    docs = sorted({c["doc_id"] for c in chunks})
    print(f"切块完成：{len(chunks)} 块（正文 {len(chunks)-n_tab} / 表格 {n_tab}），覆盖 {len(docs)} 份报告")
    print(f"平均每块 {sum(c['n_chars'] for c in chunks)/max(len(chunks),1):.0f} 字")
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
