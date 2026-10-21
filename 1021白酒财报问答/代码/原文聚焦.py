# -*- coding: utf-8 -*-
"""生成侧「读表辅助」。

职责边界：只服务生成环节 —— 把召回块里与提问相关的**原文行**整理成易读视图，
并给出一个「提问 → 财报科目」的通用映射提示，帮助小模型越过同义表述这道坎。

明确不做的事：
  * 不改检索召回结果、不改路由、不改指标定向权重、不改引用校验；
  * 不写死任何公司、年份、具体数值或某道题的答案；
  * 只做行的筛选与「空单元格压缩」，不新增数字、不计算、不换算单位。
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts = os.path.join(ROOT, "代码")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from 指标规则 import detect_metric  # noqa: E402

# ---------------------------------------------------------------- 词面工具

CN_RUN = re.compile(r"[\u4e00-\u9fa5]{2,6}")
UNIT_RE = re.compile(r"单位\s*[：:]\s*([^\s|，。；、]{1,6})")
DIGIT_RE = re.compile(r"\d[\d,，]*\.?\d*")
PREFIX_LINE = re.compile(r"【[^】]*】")

# 提问里常见的虚词/比较词，不做为聚焦词
QUESTION_NOISE = frozenset("""
多少 分别 哪家 哪些 是否 请问 相比 这一 这些 那些 那么 什么 怎样 如何 以及 之间 还有
公司 企业 年报 半年报 年度报告 年度 半年 上半年 全年 年末 年底 同期 报告 数据 金额
以下 上述 其中 情况 分别 大约 大概 左右 之间 存在 出现 发生 显示 披露
""".split())

# 同比类提问的加成词
YOY_WORDS = ("同比", "增减", "比上年", "较上年", "增长", "下滑", "增速", "下降")

# 通用会计口径说明（只讲科目含义与常见写法，不含任何公司、年份、数值）
SUBJECT_GLOSSARY = {
    "营业收入": (
        "「营业收入」在材料中也可能写作：营业总收入、营业收入合计、主营业务收入。"
        "公司合并口径的年度金额通常在“主要会计数据 / 主要财务指标”表或利润表首行；"
        "分行业、分产品、分地区、分渠道表里的金额是局部口径，不能当作公司总额。"
    ),
    "合同负债": (
        "「合同负债」核算的是客户（如经销商）已付款、公司尚未交付或未确认收入的部分，"
        "材料中也可能写作：合同负债、预收款项、预收账款、预收货款。"
        "资产负债表中该科目的期末（年末）余额，即提问所指的金额。"
    ),
    "净利润": (
        "提问通常指向合并口径的「归属于上市公司股东的净利润」（也写作“归属于母公司所有者的净利润”、"
        "“归母净利润”）。注意与同一表格里的“扣除非经常性损益的净利润”“少数股东损益”区分开。"
    ),
    "毛利率": (
        "「毛利率」通常在“主营业务分行业 / 分产品 / 分销售模式”表的毛利率列，写作“毛利率”“毛利率（%）”或百分比数值。"
        "若材料只给出分口径（如分渠道、分产品）的毛利率，请连同该口径名称一起说明，不要当成公司整体毛利率。"
    ),
    "产销量": (
        "「产销量」通常在“产销量情况”表，列名多为生产量 / 销售量，单位随该表标注（常见千升、吨），"
        "请以该表标注的单位为准。"
    ),
}


def strip_surface(question, names=()):
    """去掉问句表面的公司名 / 年份 / 标点，留下实义部分用于抽词。"""
    q = question
    for n in names:
        if n:
            q = q.replace(n, " ")
    q = re.sub(r"20\d{2}\s*年?", " ", q)
    q = re.sub(r"[0-9０-９\s年月日度第章节％%．.,，。？?！!、；;：:（）()\[\]]", " ", q)
    return q


def question_tokens(question, names=()):
    """从问句里抽实义中文片段（2~6 字），按长度降序，便于长词优先命中。"""
    toks = []
    for t in CN_RUN.findall(strip_surface(question, names)):
        t = t.strip()
        if len(t) < 2 or t in QUESTION_NOISE:
            continue
        toks.append(t)
    # 长词在前，避免「营业收入」被「营业」先占用额度
    return sorted(set(toks), key=lambda x: (-len(x), x))


# ---------------------------------------------------------------- 行视图

def _clean_lines(text):
    """取正文行：跳过块首标记（[表格]）与块前缀行（【公司｜年份｜章节｜第N页】）。"""
    lines = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if re.fullmatch(r"\[[^\]]{0,8}\]", s):          # [表格] / [图] 等标记
            continue
        if PREFIX_LINE.fullmatch(s) and ("页】" in s or s.count("｜") >= 3):
            continue                                     # 块前缀，信息已在标题行里
        lines.append(s)
    return lines


def _split_lines(text):
    lines = _clean_lines(text)
    return (lines[0] if lines else ""), lines


def _render(cells):
    return " | ".join(c for c in cells if c)


def merge_wrapped(lines):
    """把被切块/折行截断的表格行按列拼回一行。

    年报表格里一个单元格太长时会另起一行，剩余值落在后面的列上，
    不拼回来的话模型就会读到「净利润有金额、没有同比」这种残缺行。
    """
    rows = []
    for raw in lines:
        cells = [p.strip() for p in raw.split("|")]
        lead = 0
        while lead < len(cells) and not cells[lead]:
            lead += 1
        nonempty = [c for c in cells if c]
        tail_has = any(c for c in cells[lead:])

        if rows and lead >= 2 and tail_has:
            prev = rows[-1]                       # 续行：把非空值按列填回上一行
            for i, c in enumerate(cells):
                if c:
                    prev.extend([""] * (i + 1 - len(prev)))
                    prev[i] = c
        elif rows and nonempty and not has_number(" ".join(nonempty)) and len("".join(nonempty)) <= 4:
            rows[-1][0] = (rows[-1][0] + "".join(nonempty)).strip()   # 如散落的「（元）」
        else:
            rows.append(cells)
    return [_render(r) for r in rows if _render(r)]


def compact(line):
    """压缩表格行的空单元格：`| a |  |  | b |` -> `a | b`，数字与单位一个字符不动。"""
    if "|" not in line:
        return line.strip()
    return _render([p.strip() for p in line.split("|")])


def has_number(line):
    return bool(DIGIT_RE.search(line))


def _is_header(line):
    """表头/说明行：没有数字，且列数≥2（或直接口语叙述句）。"""
    return (not has_number(line)) and (line.count("|") >= 2 or len(line) < 40)


def pick_rows(text, strong=(), tokens=(), yoy=False, max_rows=8, max_span=12):
    """在一个块里挑出与提问最相关的原文行（只压缩空单元格，不改数字与单位）。

    命中落在表头/说明行时，连带给出它下面的数据行 —— 否则「营业收入」只在表头出现、
    数据行里没有该词，模型仍然读不到数值。
    """
    lines = merge_wrapped(_clean_lines(text))
    if not lines:
        return "", []
    unit = ""
    for line in lines:
        m = UNIT_RE.search(line)
        if m:
            unit = m.group(1)
            break

    scored = []
    for i, line in enumerate(lines):
        score = 0
        if strong and any(t in line for t in strong):
            score += 10
        score += sum(1 for t in tokens if t in line)
        if yoy and any(w in line for w in YOY_WORDS):
            score += 2
        if not score:
            continue
        if not _is_header(line):
            score += 1                      # 数据行优先于纯说明行
        scored.append((score, i, line))

    if not scored:
        return unit, []

    scored.sort(key=lambda x: (-x[0], x[1]))
    top_score, top_i, top_line = scored[0]

    if _is_header(top_line) or top_score >= 10 and _is_header(top_line):
        # 命中的是表头：整张表（含其数据行）都给出来，让模型按列对应着读
        idxs = list(range(top_i, min(len(lines), top_i + max_span)))
    else:
        idxs = [i for _, i, _ in scored]
    idxs = sorted(set(idxs))[:max_rows]
    return unit, [lines[i] for i in idxs]


# ---------------------------------------------------------------- 对外主函数

def _render_block(i, h, rows, unit, per_row):
    bits = [f"[{i}] {h['company']}｜{h['year']}年{h['rtype']}｜第{h['page']}页"]
    if unit:                       # 只在块里确有“单位：X”时才标注，避免模型拿常识补单位
        bits.append(f"    单位：{unit}")
    for r in rows:
        line = compact(r)
        if line:
            bits.append("    行：" + line[:per_row])
    return "\n".join(bits)


def focus_view(hits, question, max_rows=8, per_row=170, budget=11000):
    """为每个召回块生成「聚焦行」视图。返回 (字符串, 指标名)；无相关内容时字符串为空。

    输出里每一行都原样取自该块正文（只去掉空单元格），因此仍可被 citations 追溯。
    行数先按「每块少量」铺一遍，预算有余再放宽 —— 保证多公司题里每家公司至少有一块上榜。
    """
    names = tuple({h.get("company", "") for h in hits} | {h.get("code", "") for h in hits})
    metric, yoy = detect_metric(question)
    toks = question_tokens(question, names)
    strong = (metric,) if metric else ()

    picked = []
    for h in hits:
        unit, rows = pick_rows(h["text"], strong, toks, yoy, max_rows=max_rows)
        picked.append((unit, rows))

    for cap in (3, 5, max_rows):
        blocks = [
            _render_block(i, h, rows[:cap], unit, per_row)
            for i, (h, (unit, rows)) in enumerate(zip(hits, picked), 1) if rows
        ]
        if sum(len(b) for b in blocks) <= budget:
            break

    if not blocks:
        return "", metric
    while blocks and sum(len(b) for b in blocks) > budget:
        blocks.pop()                      # 命中顺序已是轮转的，末尾的是各家的次优块
    tip = ("（以下各行摘自上方对应编号的材料原文：只去掉了空单元格、并把被折行的单元格拼回同一行，"
           "数字与单位未做任何改动。某行原文没写单位时，回答只给数字、不要补单位。）")
    return "【聚焦原文行】\n" + tip + "\n" + "\n".join(blocks) + "\n", metric


def metric_hint(metric):
    """根据识别到的科目/指标输出通用 reading 提示（只讲会计口径，不含任何具体数值）。"""
    if not metric:
        return ""
    line = SUBJECT_GLOSSARY.get(metric)
    head = (f"【提问对应的财报科目】本提问指向的科目/指标是「{metric}」。"
            f"材料中凡是出现该科目（或其全称、简称、别名）的行，就是候选答案所在行；"
            f"请优先在这些行里取值。")
    return head + (f"\n【科目口径】{line}" if line else "")
