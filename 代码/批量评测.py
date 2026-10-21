# -*- coding: utf-8 -*-
"""第 9 步：按 数据/评测/questions_v1.json 逐题跑基线评测（只测不改）。

本脚本**不修改任何检索 / 路由 / 引用 / 提示词 / 页面逻辑**，只调用现有接口：
    Retriever.route()  →  生成回答.answer()  →  引用追溯.verify_answer()

每题记录：路由类型、识别到的公司/年份/报告类型、召回块清单、各公司召回分布、
最终回答、引用块、数字追溯状态、自动覆盖检查。

产物：
    数据/评测/results_v1.json   机器可读全量结果
    数据/评测/基线结果_v1.md     人读报告（由本脚本生成骨架，判分需人工复核）

命令行：
    python 代码/run_eval.py                    # 跑全部（默认 数据/评测/questions_v1.json）
    python 代码/run_eval.py --ids 1,2,3        # 只跑指定题
    python 代码/run_eval.py --no-llm           # 只检索不生成
    python 代码/run_eval.py --questions 数据/评测/questions_v2.json --out 数据/评测/results_v2.json

本脚本只做「调用 + 记录」，不参与任何检索 / 生成 / 引用决策。
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 检索引擎 import Retriever  # noqa: E402
import 生成回答  # noqa: E402

QUESTIONS = os.path.join(ROOT, "数据", "评测", "题库", "题库1-最初的10道题.json")
OUT_JSON = os.path.join(ROOT, "数据", "评测", "结果", "评测结果.json")

# 与页面默认值保持一致
TOP_K = 9
PER_COMPANY_K = 3
MODE = "hybrid"

ALL_COMPANIES = [
    "贵州茅台", "五粮液", "泸州老窖", "山西汾酒", "洋河股份", "古井贡酒",
    "今世缘", "迎驾贡酒", "口子窖", "水井坊", "舍得酒业", "酒鬼酒",
]

INSUFFICIENT_MARKERS = ("材料不足", "材料未覆盖", "未覆盖", "无法追溯", "不予采信")


def hit_brief(h):
    return {
        "rank": h["rank"],
        "id": h["id"],
        "company": h["company"],
        "year": h["year"],
        "rtype": h["rtype"],
        "section": h["section"],
        "page": h["page"],
        "block_type": h["block_type"],
        "vector": round(h["vector"], 4) if h.get("vector") is not None else None,
        "bm25": round(h["bm25"], 2) if h.get("bm25") is not None else None,
        "chars": len(h["text"]),
        "preview": h["text"][:150].replace("\n", " "),
    }


def run_one(r, q, no_llm=False):
    t0 = time.time()
    hits, info = r.route(
        q["question"], top_k=TOP_K, per_company_k=PER_COMPANY_K, mode=MODE, rerank=True,
    )
    rt = time.time() - t0

    dist = dict(Counter(h["company"] for h in hits))
    # 覆盖检查：预期公司（路由识别的，或全局题的全部12家）是否都被召回
    if info["route"] == "global" or not info.get("companies"):
        expect = ALL_COMPANIES
    else:
        expect = info["companies"]
    missing = [c for c in expect if c not in dist] if len(expect) <= 12 else []

    rec = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "capability": q["capability"],
        "difficulty": q.get("difficulty"),
        "expected_source": q.get("expected_source"),
        "baseline_note": q.get("baseline"),
        "route": info.get("route"),
        "detected": {
            "companies": info.get("companies"),
            "year": info.get("year"),
            "rtype": info.get("rtype"),
            "rtype_strict": info.get("rtype_strict"),
            "metric": info.get("metric"),
            "per_company_k": info.get("per_company_k"),
            "per_company_hits": info.get("per_company_hits"),
        },
        "retrieve_sec": round(rt, 3),
        "n_hits": len(hits),
        "hits": [hit_brief(h) for h in hits],
        "company_distribution": dist,
        "expected_companies": expect,
        "missing_companies": missing,
        "coverage_ok": len(missing) == 0,
    }

    if no_llm:
        rec["answer"] = None
        rec["generate_sec"] = None
        rec["verification"] = None
        return rec, hits

    ans, gt, ver = 生成回答.answer(q["question"], hits)
    rec["answer"] = ans
    rec["generate_sec"] = round(gt, 2)
    rec["verification"] = {
        "summary": ver.get("summary"),
        "sentences": [
            {
                "text": s.get("text", "")[:200],
                "status": s.get("status"),
                "cites": [
                    {"n": c.get("n"), "chunk_id": c.get("chunk_id"), "company": c.get("company"),
                     "year": c.get("year"), "rtype": c.get("rtype"), "page": c.get("page")}
                    for c in s.get("citations", [])
                ],
                "numbers": [
                    {"raw": n.get("raw"), "matched": n.get("matched")}
                    for n in s.get("numbers", [])
                ],
            }
            for s in ver.get("sentences", [])
        ],
        "citations": [
            {
                "n": c.get("n"), "chunk_id": c.get("chunk_id"), "company": c.get("company"),
                "year": c.get("year"), "rtype": c.get("rtype"), "page": c.get("page"),
                "block_type": c.get("block_type"),
            }
            for c in ver.get("citations", [])
        ],
    }
    rec["says_insufficient"] = any(k in (ans or "") for k in INSUFFICIENT_MARKERS)
    n_num = sum(len(s.get("numbers", [])) for s in ver.get("sentences", []))
    n_ok = sum(1 for s in ver.get("sentences", []) for n in s.get("numbers", []) if n.get("matched"))
    rec["n_numbers"] = n_num
    rec["n_numbers_matched"] = n_ok
    rec["auto_pass_hint"] = (not rec["says_insufficient"]) and n_num > 0 and n_ok == n_num and rec["coverage_ok"]
    return rec, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="只跑指定题号，如 1,2,3")
    ap.add_argument("--out", default=None, help="结果输出路径（默认 数据/评测/results_v1.json）")
    ap.add_argument("--questions", default=None,
                    help="题集文件路径（默认 数据/评测/questions_v1.json）")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()

    q_path = os.path.abspath(a.questions) if a.questions else QUESTIONS
    print(f"[题集] {os.path.relpath(q_path, ROOT)}", flush=True)

    qs = json.load(open(q_path, encoding="utf-8"))["questions"]
    if a.ids:
        want = {int(x) for x in a.ids.split(",")}
        qs = [q for q in qs if q["id"] in want]
    out_json = a.out or OUT_JSON

    t0 = time.time()
    r = Retriever(with_model=True)
    print(f"[索引加载 {time.time()-t0:.1f}s] {r.stats()['n_chunks']} 块 / {r.stats()['n_docs']} 份报告", flush=True)

    results = []
    for q in qs:
        rec, hits = run_one(r, q, no_llm=a.no_llm)
        results.append(rec)
        print(f"\n{'='*78}\nQ{q['id']} [{q['type']}] {q['question']}", flush=True)
        print(f"  路由={rec['route']} | 公司={rec['detected']['companies']} | "
              f"年={rec['detected']['year']} | 类型={rec['detected']['rtype']} | "
              f"检索 {rec['retrieve_sec']}s", flush=True)
        print(f"  召回 {rec['n_hits']} 块 | 分布 {rec['company_distribution']}", flush=True)
        print(f"  缺失公司 {rec['missing_companies'] or '无'} | 覆盖OK={rec['coverage_ok']}", flush=True)
        for h in rec["hits"]:
            mark = "表" if h["block_type"] == "table" else "文"
            print(f"    [{h['rank']}] 块#{h['id']} {mark} {h['company']} {h['year']}年{h['rtype']} "
                  f"p{h['page']} vec={h['vector']} bm25={h['bm25']} | {h['section'][:20]}", flush=True)
            print(f"         {h['preview'][:110]}", flush=True)
        if rec.get("answer") is not None:
            print(f"  --- 回答（生成 {rec['generate_sec']}s）---\n{rec['answer']}", flush=True)
            print(f"  --- 追溯：{rec['n_numbers_matched']}/{rec['n_numbers']} 数字可追溯 | "
                  f"summary={rec['verification']['summary']} | "
                  f"引用块={[c['chunk_id'] for c in rec['verification']['citations']]}", flush=True)
            for s in rec["verification"]["sentences"]:
                if s["status"] != "none":
                    print(f"    [{s['status']}] {s['text'][:90]} | "
                          f"{[(n['raw'], n['matched']) for n in s['numbers']]}", flush=True)

    json.dump(results, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[完成] 共 {len(results)} 题 -> {os.path.relpath(out_json, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
