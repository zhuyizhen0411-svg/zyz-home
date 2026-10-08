# -*- coding: utf-8 -*-
"""冒烟测试：跑几道典型问题，确认「检索 → 生成」整条链路可用。

用法:
    python 代码/smoke_test.py            # 含大模型生成
    python 代码/smoke_test.py --no-llm   # 只看召回，秒出
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 检索引擎 import Retriever  # noqa: E402

QUESTIONS = [
    ("贵州茅台2025年营业收入是多少？同比增速多少？", {}),
    ("五粮液2025年末合同负债余额是多少？", {}),
    ("泸州老窖2025年毛利率是多少？", {}),
    ("舍得酒业2025年白酒产量和销量分别是多少吨？", {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--top-k", type=int, default=6)
    a = ap.parse_args()

    r = Retriever(with_model=True)
    s = r.stats()
    print(f"索引：{s['n_chunks']} 块 / {s['n_docs']} 份报告 / {s['n_companies']} 家公司 / {s['dim']} 维\n")

    llm = None
    if not a.no_llm:
        from 生成回答 import generate_answer  # noqa: PLC0415

        llm = generate_answer

    for q, filt in QUESTIONS:
        t = time.time()
        hits, info = r.route(q, top_k=a.top_k, per_company_k=3, **filt)
        rt = time.time() - t
        print(f"Q: {q}")
        print(f"   路由 {info['route']}｜公司 {info['companies']}｜年 {info.get('year')}｜类型 {info.get('rtype')}")
        print(f"   召回 {len(hits)} 块（{rt:.2f}s）")
        for h in hits:
            mark = "表" if h["block_type"] == "table" else "文"
            print(f"     [{h['rank']}] {mark} {h['company']}｜{h['year']}年{h['rtype']}｜"
                  f"{h['section'][:18]}｜p{h['page']}｜vec={h['vector'] or 0:.2f} bm25={h['bm25'] or 0:.1f}")
            print(f"          {h['text'][:100].replace(chr(10), ' ')}")
        if llm:
            ans, gt = llm(q, hits)
            print(f"   A（{gt:.1f}s）：{ans[:400]}")
        print()

    print("冒烟测试结束")


if __name__ == "__main__":
    main()
