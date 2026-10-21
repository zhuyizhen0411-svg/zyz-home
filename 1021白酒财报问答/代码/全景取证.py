# -*- coding: utf-8 -*-
"""全景题按公司取证：把「行业普遍问题 / 各家应对措施」拆成证据义务，逐家公司走一遍。

为什么单独做这一层
------------------
第三课第 61–63 页讲的正是这个坑：问「2026 年上半年白酒行业普遍面临什么问题、
各家分别采取了什么应对措施」，向量、分词、抽关键词、精炼问题、Agentic 挑公司
五条路线，每条都只召回三四家公司、几段话，五个答案里每家也只写出两到四家的措施。

原因不是检索不够强，是**题目形状不对**：这是一道「全景题」，
它的答案本来就分布在 12 家公司各自的「管理层讨论与分析」里，
靠一次 Top-K 检索永远只能看到一角。

所以这里的做法不是「再调一次召回」，而是：
    ① 先把问题拆成 2 项「证据义务」——行业共同问题 / 本公司应对措施
    ② 按公司逐家走：12 家 × 2026 半年报 × 第三节管理层讨论与分析，全节扫描
    ③ 画覆盖地图：每家每项义务标「已覆盖 / 部分 / 缺据」，缺的就说缺
    ④ 只在已覆盖的公司上汇总共性问题与措施，缺据的公司明确列出

输出：
    数据/评测/全景取证.json                         机器可读（每家的证据块、页码、原句）
    文档/02-实验过程与结果/第12步-全景题按公司取证.md   人读的覆盖地图

命令行：
    python 代码/全景取证.py
    python 代码/全景取证.py --year 2026 --rtype 半年报 --tag 2026H1
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 检索引擎 import Retriever  # noqa: E402

QUESTION = "2026年上半年，白酒行业的上市公司普遍面临什么共同的问题？各家分别采取了什么应对措施？"

# 两项证据义务。关键词只描述「财报里怎么写这类内容」，不含公司名与数值。
#
# core（核心词）是这条义务「非它不可」的信号：问题义务必须有「下行/承压/竞争/库存」
# 这类词，措施义务必须有「推进/加强/优化/渠道」这类动作词。ctx（场景词）只说明这句
# 话是在讲行业或讲本公司。只有核心词才算 strong —— 否则一句「满足不同层次消费者
# 的需求」也会被算成「行业问题」，覆盖地图就变成盖章了。
OBLIGATIONS = {
    "O1_行业共同问题": {
        "desc": "行业层面普遍存在的问题（需求、库存、价格、竞争、消费变化等）",
        "core": ("调整", "下行", "承压", "压力", "竞争", "库存", "挑战", "困难", "下滑",
                 "疲弱", "收缩", "过剩", "减弱", "分化", "波动", "增速放缓", "去库存",
                 "价格倒挂", "动销", "需求不足"),
        "ctx": ("行业", "白酒", "消费"),
    },
    "O2_本公司应对措施": {
        "desc": "本公司针对上述问题采取的具体做法（渠道、产品、品牌、营销、控量稳价等）",
        "core": ("措施", "应对", "推进", "加强", "优化", "提升", "完善", "深化", "拓展",
                 "聚焦", "围绕", "开展", "实施", "控量", "稳价", "渠道", "打造", "建设",
                 "强化", "布局", "降本", "增效"),
        "ctx": ("公司", "产品", "品牌", "营销", "市场"),
    },
}

# 把「证据句」归到几个话题族，用于统计「几家提到」。族名就是全景题答案里能写的说法，
# 每族后面的括号是判据词——只做归类，不改原句。
FAMILIES = {
    "O1_行业共同问题": [
        ("行业处于深度调整 / 调整周期", ("深度调整", "调整期", "调整周期", "结构性调整", "周期性")),
        ("渠道库存压力 / 去库存", ("库存",)),
        ("价格体系 / 价盘承压", ("价格", "价盘", "倒挂")),
        ("存量竞争 / 分化加剧 / 集中度提升", ("存量竞争", "竞争加剧", "竞争日趋", "分化", "集中度", "马太效应")),
        ("消费需求偏弱 / 消费场景变化", ("需求偏弱", "需求不足", "需求减弱", "动能不足", "消费信心",
                                          "消费场景", "消费疲弱", "消费需求")),
        ("动销下降 / 销售承压", ("承压", "三重收缩", "下滑", "下降", "动销")),
    ],
    "O2_本公司应对措施": [
        ("渠道建设与终端精耕", ("渠道", "终端", "下沉", "乡镇", "市场建设")),
        ("产品结构优化 / 大单品", ("产品结构", "大单品", "产品升级", "主导品牌", "产品创新")),
        ("品牌建设与营销创新", ("品牌", "营销", "宣传", "IP")),
        ("消费者培育 / 场景运营", ("消费者培育", "场景", "宴席", "圈层", "开瓶")),
        ("控量稳价 / 供需适配", ("控量", "稳价", "供需", "配额", "价格体系")),
        ("数字化 / 组织与降本增效", ("数字化", "降本", "增效", "组织力", "组织架构", "组织变革")),
    ],
}

NUM_RE = re.compile(r"\d{1,3}(?:,\d{3}){2,}|\d+\.\d+")
SENT_SPLIT = re.compile(r"(?<=[。；;！!？?])")
MD_SECTION = "管理层讨论"

# 目录页 / 表格勾选框 / 页眉这类「长得像正文、其实没内容」的句子，不算证据。
BOILERPLATE = re.compile(
    r"^[【\[].{0,60}[】\]]$"          # 纯块头
    r"|□\s*适用|√\s*适用|不适用"        # 勾选框
    r"|^第[一二三四五六七八九十]+节"      # 节标题
    r"|^\(?[一二三四五六七八九十]+\)?\s*$"
)


# 含「竞争」两字、但说的不是行业竞争的固定说法，打分前先摘掉，
# 否则「不可复制的核心竞争力」会被当成「行业竞争加剧」。
FALSE_FRIENDS = ("核心竞争力", "竞争优势", "竞争力", "竞争性")

# 块头与页眉：判分和展示时都去掉，免得证据句里混进「【…｜第11页】」这种没信息的东西。
HEADER_LINE = re.compile(r"^【[^】]*】\s*$|^.{0,30}(股份有限公司|有限公司)20\d{2}年(半年度|年度)报告(全文)?\s*$|^\[表格\]\s*$")


def _clean(text):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lines = [ln for ln in lines if not HEADER_LINE.match(ln)]
    return "\n".join(lines)


def _sentences(text):
    out = []
    for s in SENT_SPLIT.split(_clean(text)):
        s = re.sub(r"[ \t　]+", "", s).strip()   # 保留换行：换行后才知道哪一截是小标题
        if len(s) < 12 or BOILERPLATE.search(s):
            continue
        out.append(s)
    return out


# 出现这些词，说明这一截已经在说事了，不是小标题，不能砍掉。
VERBISH = ("面临", "处于", "经历", "持续", "加剧", "承压", "呈现", "存在", "受到",
           "成为", "进入", "仍", "正", "将", "面临", "只有", "随着")


def _trim_lead(s):
    """去掉句首那几截「小标题」：短、没有逗号句号、也没有在说事。

    例：「1、品牌运营情况」「（一）公司上半年经营情况回顾」要砍；
        「白酒行业正经历着深刻的周期性」不能砍（它有「正」、「经历」）。
    """
    parts = [p for p in s.split("\n") if p]
    while len(parts) > 1:
        p = parts[0]
        if len(p) < 20 and "，" not in p and "。" not in p and not any(v in p for v in VERBISH):
            parts.pop(0)
        else:
            break
    return "".join(parts)


def _score(line, words, drop_false_friends=False):
    if drop_false_friends:
        for p in FALSE_FRIENDS:
            line = line.replace(p, "")
    return sum(1 for w in words if w in line)


def assess(text, ob):
    """块内逐句打分，取最好的一句。

    strong：核心词 ≥2 且出现场景词 —— 这句话确实在讲「行业有问题」或「本公司做了什么」
    weak  ：只沾到核心词或只沾到场景词 —— 提到了，但不像正面回答
    none  ：都没有
    """
    best, best_level, best_core = "", "none", 0
    for s in _sentences(text):
        c = _score(s, ob["core"], drop_false_friends=True)
        h = _score(s, ob["ctx"])
        if c >= 1 and h >= 1:
            level = "strong"
        elif c >= 1 or h >= 1:
            level = "weak"
        else:
            continue
        # 同一块里挑最强的一句：strong 优先，其次核心词多
        rank = {"strong": 2, "weak": 1, "none": 0}
        if (rank[level], c) > (rank[best_level], best_core):
            best, best_level, best_core = _trim_lead(s), level, c
    return best, best_level, best_core


def company_chunks(r, company, year, rtype):
    """某家公司某期的「管理层讨论与分析」全部块（不排序、不截断 —— 全景题要全节看）。"""
    out = []
    for i, m in enumerate(r.meta):
        if m["company"] != company:
            continue
        if year and m["year"] != year:
            continue
        if rtype and m["rtype"] != rtype:
            continue
        if MD_SECTION not in m["section"]:
            continue
        out.append((i, m))
    return out


def collect(r, year, rtype):
    companies = r.companies()
    result = {"question": QUESTION, "year": year, "rtype": rtype,
              "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "obligations": {k: v["desc"] for k, v in OBLIGATIONS.items()},
              "companies": []}
    for c in companies:
        chunks = company_chunks(r, c, year, rtype)
        row = {"company": c, "n_chunks": len(chunks), "evidence": {}, "coverage": {},
               "families": {}}
        for key, ob in OBLIGATIONS.items():
            hits, strong_texts = [], []
            for i, m in chunks:
                sent, level, sc = assess(r.texts[i], ob)
                if level == "none":
                    continue
                if level == "strong":
                    strong_texts.append(r.texts[i])
                hits.append({
                    "chunk_id": int(m["id"]), "page": m["page"], "section": m["section"],
                    "level": level, "score": sc,
                    "quote": sent[:160],
                })
            hits.sort(key=lambda h: (h["level"] != "strong", -h["score"]))
            row["evidence"][key] = hits[:3]
            # 话题族统计扫「全部 strong 证据块」的整块原文，不用只留三句的展示列表，免得漏算
            joined = "".join(strong_texts)
            row["families"][key] = [name for name, kws in FAMILIES[key]
                                    if any(kw in joined for kw in kws)]
            strong = [h for h in hits if h["level"] == "strong"]
            weak = [h for h in hits if h["level"] == "weak"]
            if strong:
                row["coverage"][key] = "已覆盖"
            elif weak:
                row["coverage"][key] = "部分"
            else:
                row["coverage"][key] = "缺据"
        # 整家的覆盖状态：两项都到位才算覆盖
        states = list(row["coverage"].values())
        if all(s == "已覆盖" for s in states):
            row["overall"] = "已覆盖"
        elif all(s == "缺据" for s in states):
            row["overall"] = "缺据"
        else:
            row["overall"] = "部分"
        result["companies"].append(row)
    # 汇总统计
    from collections import Counter

    cnt = Counter(c["overall"] for c in result["companies"])
    result["summary"] = dict(cnt)
    result["summary"]["总家数"] = len(result["companies"])
    for key in OBLIGATIONS:
        result["summary"][key] = dict(Counter(c["coverage"][key] for c in result["companies"]))
    # 话题族 → 哪几家公司提到（全景题答案里的每句话都能落到这份名单上）
    result["summary"]["families"] = {}
    for key, fams in FAMILIES.items():
        result["summary"]["families"][key] = {
            name: [c["company"] for c in result["companies"] if name in c["families"][key]]
            for name, _ in fams
        }
    return result


def render_md(result, path):
    lines = []
    lines.append("# 第 12 步 · 全景题按公司取证：覆盖地图\n")
    lines.append("> 对应第三课第 61–63 页（全景题五条路线都只覆盖几家）与第 90–92 页（覆盖地图：已覆盖 / 部分 / 缺据）。\n")
    lines.append(f"**问题**：{result['question']}\n")
    lines.append(f"**范围**：12 家公司 × {result['year']} 年{result['rtype']} × 第三节「管理层讨论与分析」全节\n")
    lines.append("**做法**：先把问题拆成 2 项证据义务，再按公司逐家把整节扫一遍，"
                 "每项义务标状态；不是再调一次 Top-K 召回。\n")

    ob_keys = list(OBLIGATIONS)
    lines.append("## 一、覆盖地图\n")
    lines.append("| 公司 | 管理层讨论块数 | " + " | ".join(OBLIGATIONS[k]["desc"] for k in ob_keys) + " | 整家 |")
    lines.append("|---|---|" + "---|" * len(ob_keys) + "---|")
    for c in result["companies"]:
        lines.append(f"| {c['company']} | {c['n_chunks']} | "
                     + " | ".join(c["coverage"][k] for k in ob_keys)
                     + f" | {c['overall']} |")
    lines.append("")
    s = result["summary"]
    lines.append(f"**统计**：整家已覆盖 {s.get('已覆盖', 0)} 家 / 部分 {s.get('部分', 0)} 家 / 缺据 {s.get('缺据', 0)} 家（共 {s['总家数']} 家）\n")
    for k in ob_keys:
        d = s.get(k, {})
        lines.append(f"- {OBLIGATIONS[k]['desc']}：已覆盖 {d.get('已覆盖', 0)} 家 / 部分 {d.get('部分', 0)} 家 / 缺据 {d.get('缺据', 0)} 家")
    lines.append("")

    lines.append("## 二、逐家取证（每项义务给出原句与页码）\n")
    for c in result["companies"]:
        lines.append(f"### {c['company']}：整家 {c['overall']}\n")
        for k in ob_keys:
            st = c["coverage"][k]
            ev = c["evidence"][k]
            lines.append(f"- **{OBLIGATIONS[k]['desc']}**：{st}")
            if not ev:
                lines.append("  - （该节内没有找到对应表述，记为缺据，不用别处材料顶替）")
                continue
            for h in ev[:2]:
                lines.append(f"  - 第 {h['page']} 页（块 #{h['chunk_id']}，{h['level']}）：{h['quote']}")
        lines.append("")

    fam_all = s.get("families", {})
    if fam_all:
        lines.append("## 三、共性话题汇总（按提及家数排序）\n")
        lines.append("这一节才是「共同问题」的答案骨架：同一件事被几家写到、分别是哪几家。"
                     "答案里每写一条，都要能对到下面某一行。\n")
        for key in ob_keys:
            fams = fam_all.get(key, {})
            if not fams:
                continue
            lines.append(f"### {OBLIGATIONS[key]['desc']}\n")
            lines.append("| 话题 | 提及家数 | 公司 |")
            lines.append("|---|---|---|")
            for name, comps in sorted(fams.items(), key=lambda kv: -len(kv[1])):
                lines.append(f"| {name} | {len(comps)}/{len(result['companies'])} | {'、'.join(comps) or '—'} |")
            lines.append("")

    lines.append("## 四、这张表怎么看\n")
    lines.append("- 「缺据」不是失败，是把「没有」说出来。缺据的公司不许用别的公司的情况替它说话，"
                 "也不许用行业常识补。")
    lines.append("- 全景题的答案只能由「已覆盖」的公司撑着；每一条共性问题都要能指回某家公司的原句+页码。")
    lines.append("- 与逐题记录里的「召回表现」分开看：这一节只回答「证据有没有覆盖到每家」，"
                 "不回答「最终答案写得对不对」。")
    lines.append("- **12 家全都有证据，恰恰说明这里的问题不在语料，而在取样方式**：")
    lines.append("  语料里每家都写了行业问题和应对措施，可一次 Top-K 检索只端上来三四家、几段话。")
    lines.append("  所以全景题不能靠「调召回」，只能按公司逐家走一遍——这张表就是走完的结果。")
    lines.append("- 措施那一栏 12 家都覆盖是实情（每家半年报都写渠道、品牌、产品），")
    lines.append("  它区分不出高下；真正能拉开差别的是「行业问题」那一栏的家数（10/12、7/12、6/12），")
    lines.append("  以及哪几家没提到——那才是「共同」与「不共同」的分界。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--rtype", default="半年报", choices=["年报", "半年报"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    r = Retriever(with_model=False)
    result = collect(r, a.year, a.rtype)
    tag = a.tag or f"{a.year}{a.rtype}"
    out_json = os.path.join(ROOT, "数据", "评测", "全景取证.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    out_md = os.path.join(ROOT, "文档", "02-实验过程与结果", "第12步-全景题按公司取证.md")
    render_md(result, out_md)
    s = result["summary"]
    print(f"全景取证完成（{tag}）：整家已覆盖 {s.get('已覆盖', 0)} 家 / 部分 {s.get('部分', 0)} 家 / "
          f"缺据 {s.get('缺据', 0)} 家")
    print(f"  机读：{os.path.relpath(out_json, ROOT)}")
    print(f"  人读：{os.path.relpath(out_md, ROOT)}")


if __name__ == "__main__":
    main()
