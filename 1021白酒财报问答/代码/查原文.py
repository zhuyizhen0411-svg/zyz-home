# -*- coding: utf-8 -*-
"""原文核对工具：按「公司 / 报告期 / 关键词」在解析后的原文块里查行，并打印出处。

用途（对应老师要求的「先核对口径与出处」）：
  * 出题时定金标：看清目标科目的那一行到底写在哪份报告、第几页、哪张表、单位是什么；
  * 交付时复核：回答里的每个数字，能不能落回这一行。

它只读 `数据/文本块.jsonl`，不改任何东西。

命令行：
    python 代码/查原文.py --company 泸州老窖 --year 2025 --rtype 年报 --kw 营业成本
    python 代码/查原文.py --company 五粮液 --year 2025 --rtype 年报 --kw 研发费用 --ctx 3
    python 代码/查原文.py --kw 资产负债率 --year 2025 --rtype 年报 --page
    python 代码/查原文.py --chunk 31342            # 直接看某个块的全文与出处
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = os.path.join(ROOT, "数据", "文本块.jsonl")


def load_blocks(company=None, year=None, rtype=None):
    with open(BLOCKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if company and c["company"] != company:
                continue
            if year and c["year"] != year:
                continue
            if rtype and c["rtype"] != rtype:
                continue
            yield c


def show_block(c, kw=None, ctx=2, prefix=""):
    head = f"{prefix}#{c['id']} {c['company']}｜{c['year']}年{c['rtype']}｜{c['section']}｜第{c['page']}页｜{c['block_type']}"
    print(head)
    lines = c["text"].split("\n")
    if not kw:
        for ln in lines:
            print("    " + ln)
        return
    hits = [i for i, ln in enumerate(lines) if kw in ln]
    shown = set()
    for i in hits:
        for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
            shown.add(j)
    for j in sorted(shown):
        mark = ">>" if j in hits else "  "
        print(f"    {mark} {lines[j]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--company")
    ap.add_argument("--year", type=int)
    ap.add_argument("--rtype", choices=["年报", "半年报"])
    ap.add_argument("--kw", help="关键词（在行内做子串匹配）")
    ap.add_argument("--ctx", type=int, default=2, help="命中行上下各带几行")
    ap.add_argument("--chunk", type=int, help="直接看某个块的全文")
    ap.add_argument("--limit", type=int, default=8, help="最多打印几个块")
    ap.add_argument("--page", action="store_true", help="只列出处，不打正文")
    a = ap.parse_args()

    if a.chunk is not None:
        for c in load_blocks():
            if c["id"] == a.chunk:
                show_block(c)
                return
        print(f"未找到块 #{a.chunk}")
        return

    if not a.kw:
        ap.error("请给出 --kw 或 --chunk")

    n_block = n_hit = 0
    for c in load_blocks(a.company, a.year, a.rtype):
        if a.kw not in c["text"]:
            continue
        n_block += 1
        n_hit += c["text"].count(a.kw)
        if a.page:
            print(f"#{c['id']} {c['company']}｜{c['year']}年{c['rtype']}｜{c['section']}｜第{c['page']}页｜"
                  f"{c['block_type']}｜命中 {c['text'].count(a.kw)} 次")
        else:
            show_block(c, a.kw, a.ctx)
            print()
        if n_block >= a.limit:
            print(f"...（已到上限 {a.limit} 块，共命中 {n_block}+ 块）")
            break
    print(f"\n合计：{n_block} 个块、{n_hit} 处命中（关键词「{a.kw}」）")


if __name__ == "__main__":
    main()
