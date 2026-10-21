# -*- coding: utf-8 -*-
"""出处追溯能力测试：跑若干问题，检查每个数字能否落回原始文本块。

用法:
    python 代码/test_citations.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 生成回答 import answer  # noqa: E402
from 引用追溯 import format_report, get_chunk  # noqa: E402
from 检索引擎 import Retriever  # noqa: E402

QUESTIONS = [
    "贵州茅台2025年营业收入是多少？",
    "五粮液2025年末合同负债余额是多少？",
    "舍得酒业2025年白酒产量和销量分别是多少吨？",
    "贵州茅台和五粮液2025年的营业收入分别是多少？哪家更高？",
    "贵州茅台2025年门店数量是多少家？",  # 财报不披露门店数，应触发「材料不足」
]


def main():
    r = Retriever(with_model=True)
    for q in QUESTIONS:
        print("=" * 78)
        print(f"Q: {q}")
        hits, info = r.route(q, top_k=6, per_company_k=3)
        print(f"[路由] {info['route']}｜{info['companies']}｜{info.get('year')}年｜{info.get('rtype')}")
        ans, gt, ver = answer(q, hits)
        print(f"\n【回答】（{gt:.1f}s）\n{ans}")
        print(f"\n【出处追溯】\n{format_report(ver)}")

        # 抽查第一个引用块：确认能按编号取回原文并定位到 PDF
        if ver["citations"]:
            c0 = ver["citations"][0]
            full = get_chunk(c0["chunk_id"])
            print(f"\n【抽查块 #{c0['chunk_id']}】")
            print(f"  {full['company']}｜{full['year']}年{full['rtype']}｜{full['section']}｜第{full['page']}页｜{full['block_type']}")
            print(f"  原始 PDF: {full['pdf_path']}")
            print(f"  块原文前 150 字: {full['text'][:150]}")
        print()


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"总计 {time.time()-t:.1f}s")
