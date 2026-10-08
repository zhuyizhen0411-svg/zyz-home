# -*- coding: utf-8 -*-
"""财报问答：检索（向量+BM25 混合）→ 本地大模型生成带出处的回答。

命令行：
    python 代码/生成回答.py "贵州茅台2025年的营业收入是多少"
    python 代码/生成回答.py "五粮液合同负债" --company 五粮液 --year 2025 --rtype 年报
    python 代码/生成回答.py "山西汾酒毛利率" --no-llm          # 只看召回，不生成
    python 代码/生成回答.py "山西汾酒毛利率" --mode bm25       # 只走词频
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 检索引擎 import Retriever  # noqa: E402
from 引用追溯 import verify_answer, insufficient_note, format_report  # noqa: E402
from 原文聚焦 import focus_view, metric_hint  # noqa: E402

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TEMPERATURE = float(os.environ.get("ANSWER_TEMP", "0.15"))
MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "1100"))

SYSTEM_PROMPT = (
    "你是上市公司财报分析助手。只能依据用户提供的编号材料回答，不得使用材料以外的知识。\n"
    "硬性要求：\n"
    "1. 先找后答：材料大量以“|”分隔的表格行呈现。请先在材料里逐行定位提问对应的科目/指标行——"
    "先看该表表头确定每列含义，再读目标行的数值。材料里存在该行时，不得回答“材料不足”，"
    "即使表格列很多、空单元格很多或表头没有完整出现。\n"
    "2. 引用：每给出一个数字，紧跟方括号标注它真正来源的编号，例如“营业收入16,883,810.25万元[4]”；"
    "一个数字只能标它真正来源的那个编号。\n"
    "3. 单位：数字照抄材料原文，禁止换算，也禁止跨行借用单位。"
    "以“|”分隔的表格行，若该表的标题、列名或单元格没有写明单位，按财报披露惯例即为「元」；"
    "表内写了“万元”“亿元”的从表内标注。叙述句里写的单位（如“为人民币16,883,810.25万元”）以该句为准。\n"
    "4. 口径：同一科目在材料里可能出现多行（合并数、母公司数、子公司数、分部数、小计、关联方明细）。"
    "提问没有明确指向某个局部时，取合并口径（通常在“主要会计数据”“主要财务指标”或合并报表中），"
    "避开子公司、分部、小计、关联方明细行。\n"
    "5. 跨公司比较：先对每家分别列出“公司 → 指标值 + 单位 + 同比”，确认各家口径与单位一致后再下结论；"
    "单位不一致时保留各自原单位并明确说明差异，不要急着换算。"
    "确有必要统一单位时记住方向：元→万元是 ÷10,000（数字变小），万元→元是 ×10,000（数字变大）；"
    "换算后位数没有按这个方向变化就说明算错了，应当退回按原单位说明，不得用算错的结果下结论。\n"
    "6. 材料中确实找不到该指标时才回答“材料不足”，不要推测、不要用常识补全、不要给近似值、"
    "不要自己把几个数加起来求合计；判断“增长/下滑”只看同比数字的正负符号，不要自行增设幅度门槛。\n"
    "回答用中文，简洁、直接，不要复述问题。"
)

_LLM = None


def load_llm():
    global _LLM
    if _LLM is None:
        from mlx_lm import load

        t = time.time()
        _LLM = load(ANSWER_MODEL)
        print(f"[模型加载 {time.time()-t:.1f}s]", flush=True)
    return _LLM


def build_material(hits, per_chunk=900, total=16000, question=None):
    """把召回块拼成带编号的材料；单块截断 + 总量上限，避免一块吃掉全部上下文。

    question 不为空时，末尾追加「聚焦原文行」视图 + 提问到科目的映射提示，
    帮小模型在宽表里定位行（纯啰嗦-friendly 的排版整理，不新增信息以外的数字）。
    """
    lines, used = [], 0
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h['company']}（{h['code']}）｜{h['year']}年{h['rtype']}｜{h['section']}｜第{h['page']}页"
        body = h["text"]
        if len(body) > per_chunk:
            body = body[:per_chunk] + "…"
        if used + len(body) > total:
            lines.append(f"{head}\n（略，上下文已满）")
            continue
        used += len(body)
        lines.append(f"{head}\n{body}")
    material = "\n\n".join(lines)
    if question:
        view, metric = focus_view(hits, question)
        if view:
            material = f"{material}\n\n{view}"
        hint = metric_hint(metric)
        if hint:
            material = f"{material}\n{hint}"
    return material


def generate_answer(question, hits):
    model, tokenizer = load_llm()
    from mlx_lm import generate

    material = build_material(hits, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"材料：\n{material}\n\n问题：{question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:  # mlx_lm >= 0.2x 要求温度包成 sampler 传入
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=TEMPERATURE)
    except Exception:  # noqa: BLE001
        sampler = None
    kwargs = {"sampler": sampler} if sampler is not None else {}
    t = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, **kwargs)
    return text.strip(), time.time() - t


def answer(question, hits):
    """生成回答 + 出处校验。返回 (最终回答, 生成耗时, 校验结果)。

    校验不通过时，在回答末尾追加明确的「材料不足」说明，让无法追溯的数字不被当成事实。
    """
    text, gen_t = generate_answer(question, hits)
    ver = verify_answer(text, hits)
    note = insufficient_note(ver)
    if note:
        text = f"{text}\n\n{note}"
    return text, gen_t, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--company", default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--rtype", default=None, choices=["年报", "半年报"])
    ap.add_argument("--no-llm", action="store_true", help="只检索不生成")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-route", action="store_true", help="关闭跨公司路由，退回全局 TopK")
    ap.add_argument("--per-company-k", type=int, default=3, help="路由模式下每家公司取几块")
    a = ap.parse_args()

    t0 = time.time()
    r = Retriever(with_model=True)
    print(f"[索引加载 {time.time()-t0:.1f}s] {r.stats()['n_chunks']} 块 / {r.stats()['n_docs']} 份报告", flush=True)

    t = time.time()
    if a.no_route:
        hits = r.search(
            a.question, mode=a.mode, top_k=a.top_k, company=a.company,
            year=a.year, rtype=a.rtype, rerank=not a.no_rerank,
        )
        info = {"route": "off"}
    else:
        hits, info = r.route(
            a.question, top_k=a.top_k, per_company_k=a.per_company_k, mode=a.mode,
            company=a.company, year=a.year, rtype=a.rtype, rerank=not a.no_rerank,
        )
    rt = time.time() - t
    rinfo = info["route"]
    if info.get("companies"):
        rinfo += f" · 涉及 {len(info['companies'])} 家：{'、'.join(info['companies'])}"
    if info.get("year"):
        rinfo += f" · {info['year']}年"
    if info.get("rtype"):
        rinfo += f" · {info['rtype']}"
    print(f"[路由] {rinfo}")
    print(f"[检索 {rt:.2f}s] 命中 {len(hits)} 块\n")

    for h in hits:
        mark = "表" if h["block_type"] == "table" else "文"
        print(f"  [{h['rank']}] {mark} {h['company']} {h['year']}年{h['rtype']} p{h['page']} "
              f"vec={h['vector'] or 0:.2f} bm25={h['bm25'] or 0:.1f} | {h['section'][:22]}")
        print(f"      {h['text'][:110].replace(chr(10), ' ')}")

    if not a.no_llm:
        ans, gt, ver = answer(a.question, hits)
        print(f"\n【回答】（生成 {gt:.1f}s）\n{ans}")
        print("\n【出处追溯校验】")
        print(format_report(ver))
    print("\n【出处】")
    for h in hits:
        print(f"  [{h['rank']}] 块#{h['id']} {h['company']}｜{h['year']}年{h['rtype']}｜{h['section']}｜第{h['page']}页")


if __name__ == "__main__":
    main()
