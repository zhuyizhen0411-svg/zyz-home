# -*- coding: utf-8 -*-
"""完整逐题记录：把三批题、每一题、修复前后，汇到一张表里。

老师这次要的两件事，这个脚本各解决一半：

  ① 「两批题和修复前后记录能区分开发题与新题」
     → 每题都带「批次 / 角色」两个字段；角色只有两种取值：
       开发题（调参时看过它的错，不能再当成绩）／新题（金标先冻结，系统跑之前没见过）
  ② 「补齐完整逐题记录」
     → 每题的：金标（数值+单位+口径）、出处（报告+页码+块号）、
       召回表现（公司有没有覆盖到、金标块排第几）、答案表现（答没答对、为什么）、
       修复前 / 修复后（同一道题，改代码之前和之后的回答与判定）

评价纪律（第三课第 90–92 页）：
    **召回表现与答案表现分开记**。召回只回答「证据找没找到」，
    答案回答「找到了之后有没有读对」。两者混在一起，就分不清该修检索还是该修生成。

产出：
    数据/评测/逐题记录.json
    文档/02-实验过程与结果/第11步-完整逐题记录.md

命令行：
    python 代码/逐题记录.py
"""

import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QB = os.path.join(ROOT, "数据", "评测", "题库")
RES = os.path.join(ROOT, "数据", "评测", "结果")
OUT_JSON = os.path.join(ROOT, "数据", "评测", "逐题记录.json")
OUT_MD = os.path.join(ROOT, "文档", "02-实验过程与结果", "第11步-完整逐题记录.md")

BANKS = [
    {
        "file": "题库1-最初的10道题.json",
        "batch": "第一批",
        "role": "开发题",
        "role_note": "基线（第01步）和前五步优化（第02–06步）都照着它的错题改，"
                     "所以它上面的分数只能说明「改过的地方有没有退步」，不能当新题成绩",
        "before": ["结果5b-v1题库回归.json"],
        "after": ["结果08-第12步-修复后回归-题库1.json"],
    },
    {
        "file": "题库2-全新10道题.json",
        "batch": "第二批",
        "role": "开发题（原检验题，已污染）",
        "role_note": "第07步当作检验题测得 4/10；随后第08–10步反向调参，再用它复测到 7/10。"
                     "看过错题再改代码，这批题就不再是「没见过的新题」",
        "before": ["结果6-引用校验继承修复.json"],
        "after": ["结果09-第12步-修复后回归-题库2.json"],
    },
    {
        "file": "题库3-冻结金标-10道题.json",
        "batch": "第三批",
        "role": "新题（检验题）",
        "role_note": "金标逐条抄自原文并哈希冻结（金标冻结记录.json），出题和写金标时都没跑系统。"
                     "这批题的分数才是可对外的成绩",
        "before": [],
        "after": ["结果07-第11步-题库3冻结金标检验.json"],
    },
]

NUMU = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|千元|元/股|元|个百分点|%|千升|吨|股|家)?")
CUR = {"元": 1.0, "千元": 1e3, "万元": 1e4, "亿元": 1e8}
UP_WORDS = ("增长", "增加", "上升", "提高", "上涨", "正增长")
DOWN_WORDS = ("下滑", "下降", "减少", "降低", "回落", "负增长")
INSUFFICIENT = ("材料不足", "材料未覆盖", "无法追溯", "不予采信")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_nums(text):
    out = []
    for m in NUMU.finditer(text or ""):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append((v, m.group(2)))
    return out


def _rel_ok(a, b, tol=1e-4):
    return abs(a - b) <= max(0.01, abs(b) * tol)


RATIO_UNITS = ("%", "个百分点")


def _parse_target(target, default_unit):
    """金标里的数值可能写成「32.52%」「10.0438」或「10.0438 元/股」，一律拆成（数值, 单位）。"""
    m = NUMU.fullmatch(str(target).strip())
    if not m:
        return None, default_unit
    return float(m.group(1).replace(",", "")), (m.group(2) or default_unit)


def match_value(nums, target, unit):
    """回答里有没有这个数（允许 亿元 ↔ 元 的正确换算）。

    金额只在同族内换算；百分比与百分点视为同一类（财报里这两种写法都指同一个变化量）；
    元/股、千升、吨按原值比。回答没写单位时按目标单位直比。
    """
    t, t_unit = _parse_target(target, unit)
    if t is None:
        return False
    for v, u in nums:
        if u in CUR and t_unit in CUR:
            if _rel_ok(v * CUR[u], t * CUR[t_unit]):
                return True
        elif u is None:
            if _rel_ok(v, t):
                return True
        elif u == t_unit:
            if _rel_ok(v, t):
                return True
        elif u in RATIO_UNITS and t_unit in RATIO_UNITS:
            if _rel_ok(v, t):
                return True
    return False


def _val_unit(gold):
    """「32.52%」不要再补一个「%」；「38,197,033,001.78」要补上「元」。"""
    v = str(gold.get("value") or "")
    u = str(gold.get("unit") or "")
    return v if (not u or u in v) else v + u


def parse_change(chg):
    """「-5.74%」「+0.06%」「减少3.51个百分点」都拆成（带符号的数值, 方向）。"""
    if not chg:
        return None, None
    s = str(chg)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None, None
    v = float(m.group(0))
    if any(w in s for w in DOWN_WORDS) or s.lstrip().startswith("-"):
        return -abs(v), "down"
    if any(w in s for w in UP_WORDS) or s.lstrip().startswith("+"):
        return abs(v), "up"
    return v, None


NOTE_MARK = ("材料不足：以下数字无法追溯到原文", "口径核对：以下数字需要人工确认")


def answer_body(ans):
    """去掉系统追加的「材料不足 / 口径核对」提示段，只留回答本身。

    否则「18 个数字里有 3 个没追溯上」的提醒会被误读成「模型自己说材料不足」。
    """
    ans = ans or ""
    for mk in NOTE_MARK:
        i = ans.find(mk)
        if i > 0:
            ans = ans[:i]
    return ans.strip()


def judge_gold(q, answer):
    """按冻结的金标判一道题：数值 / 比较值 / 变化幅度 / 方向。

    只判「金标里写明了的东西」，不自创标准；判不了的（比如问的是开放性描述）
    返回「需人工」，不假装有结论。
    """
    gold = q.get("gold") or {}
    ans = answer_body(answer)
    detail = []
    if any(k in ans for k in INSUFFICIENT):
        return "答错（说材料不足，但材料里有）", "回答里出现「材料不足」，而金标出处明确存在"
    nums = parse_nums(ans)

    ok_val = None
    if gold.get("value"):
        ok_val = match_value(nums, gold["value"], gold.get("unit"))
        detail.append(f"金标值 {_val_unit(gold)}：" + ("命中" if ok_val else "未命中"))

    ok_cmp = None
    cmp_ = gold.get("compare") or {}
    qs = q.get("question") or ""
    # 只有问句明确要「上年同期是多少 / 分别是多少」时，才把比较值当成必答项；
    # 只问「是增加还是减少」的，写不写上年数字都不扣分。
    needs_cmp = any(k in qs for k in ("上年同期是多少", "分别是多少", "上期是多少",
                                      "去年同期是多少", "比上一年度是多少"))
    if cmp_.get("value"):
        hit_cmp = match_value(nums, cmp_["value"], gold.get("unit"))
        ok_cmp = True if hit_cmp else (False if needs_cmp else None)
        detail.append(f"比较值 {cmp_['value']}：{'命中' if hit_cmp else '未写（问句未强制要求）' if not needs_cmp else '未命中'}")

    ok_chg = None
    chg = cmp_.get("change")
    if chg:
        c, _dir = parse_change(chg)
        if c is not None:
            # 变化量可能是「%」也可能是「个百分点」，两种写法都认
            hit = match_value(nums, abs(c), "%") or match_value(nums, abs(c), "个百分点")
            has_up = any(w in ans for w in UP_WORDS)
            has_down = any(w in ans for w in DOWN_WORDS)
            wrong_dir = (c < 0 and has_up and not has_down) or (c > 0 and has_down and not has_up)
            if wrong_dir:
                detail.append(f"变化幅度 {chg}：方向说反了")
                return "答错（方向反了）", "；".join(detail)
            # 幅度写对了算命中；只写了方向、没写幅度，算「方向对」但不加分
            ok_chg = True if hit else (True if (has_up or has_down) else False)
            detail.append(f"变化幅度 {chg}：{'命中' if hit else ('仅方向正确' if ok_chg else '未命中')}")

    if ok_val is None:
        return "需人工", "该题金标没有可比的数值（开放性题），见下方原回答"
    if ok_val and (ok_cmp is not False) and (ok_chg is not False):
        return "答对", "；".join(detail)
    if ok_val:
        return "部分正确", "；".join(detail)
    return "答错", "；".join(detail)


def auto_verdict(rec):
    """没有金标的题（第一、二批）只能做「自动核对」，不能算人工判定。

    自动核对的口径要写清楚：数字全部能在召回块里找到、公司覆盖齐全、没有说材料不足。
    它只能证明「没有编数字」，不能证明「答的就是问的那件事」——所以表里单列一栏。
    """
    if not rec or rec.get("answer") is None:
        return "未生成回答"
    j = rec.get("judge")
    if j:
        return ("答对" if j.get("pass") else "答错") + "（历史人工判定）"
    return "自动核对通过" if rec.get("auto_pass_hint") else "自动核对未通过"


def find_result(files):
    """按文件名前缀找结果文件（允许带数字后缀的旧命名）。"""
    found = []
    for name in files:
        p = os.path.join(RES, name)
        if os.path.exists(p):
            found.append((name, load(p)))
    return found


def by_question(result_list):
    return {q["question"]: q for q in result_list}


def recall_view(rec, q):
    """召回表现：只讲「证据有没有找到」，不讲答案对错。"""
    if not rec:
        return {"状态": "无记录"}
    gold_id = (q.get("source") or {}).get("chunk_id")
    ids = [h["id"] for h in rec.get("hits", [])]
    rank = ids.index(gold_id) + 1 if gold_id in ids else None
    return {
        "识别的公司": (rec.get("detected") or {}).get("companies"),
        "识别年度": (rec.get("detected") or {}).get("year"),
        "识别报告类型": (rec.get("detected") or {}).get("rtype"),
        "命中块数": rec.get("n_hits"),
        "每家公司分布": rec.get("company_distribution"),
        "缺哪家": rec.get("missing_companies") or [],
        "公司覆盖": "齐全" if rec.get("coverage_ok") else "不全",
        "金标块块号": gold_id,
        "金标块位次": f"第 {rank} 位" if rank else ("未进召回" if gold_id else "该题未标块号"),
    }


CITE_RE = re.compile(r"\[(\d{1,2})\]")


def cited_blocks(rec):
    """回答里的 [n] 分别指向哪一块。

    答错时这一行最有用：它直接告诉你「这个数字是从哪一页、哪张表搬来的」，
    拿去和金标的页码一比，是母公司报表还是合并报表、是分部数还是合计，一眼就看出来。
    """
    if not rec:
        return []
    hits = {h.get("rank"): h for h in (rec.get("hits") or [])}
    out, seen = [], set()
    for m in CITE_RE.finditer(rec.get("answer") or ""):
        n = int(m.group(1))
        h = hits.get(n)
        if h is None or n in seen:
            continue
        seen.add(n)
        out.append(f"[{n}] {h.get('company')} {h.get('year')}{h.get('rtype')}"
                   f" 第{h.get('page')}页 块#{h.get('id')}")
    return out


def answer_view(rec, q, judged=None):
    if not rec or rec.get("answer") is None:
        return {"判定": "未生成回答"}
    return {
        "判定": judged or ("自动：数字全部可追溯" if rec.get("auto_pass_hint") else "自动：有数字对不上原文"),
        "数字可追溯": f"{rec.get('n_numbers_matched')} / {rec.get('n_numbers')}",
        "说材料不足": bool(rec.get("says_insufficient")),
        "回答": rec.get("answer"),
    }


def build():
    records = []
    for b in BANKS:
        qs = load(os.path.join(QB, b["file"]))["questions"]
        before = find_result(b["before"])
        after = find_result(b["after"])
        before_map = by_question(before[-1][1]) if before else {}
        after_map = by_question(after[-1][1]) if after else {}

        for q in qs:
            question = q["question"]
            brec = before_map.get(question)
            arec = after_map.get(question)
            gold = q.get("gold") or {}
            src = q.get("source") or {}

            if gold:
                judged_after = judge_gold(q, arec.get("answer"))[0] if arec else None
                judged_before = judge_gold(q, brec.get("answer"))[0] if brec else None
            else:
                judged_after = auto_verdict(arec)
                judged_before = auto_verdict(brec) if brec else None

            records.append({
                "批次": b["batch"],
                "角色": b["role"],
                "角色说明": b["role_note"],
                "题号": q["id"],
                "问题": question,
                "题型": q.get("type"),
                "难度": q.get("difficulty"),
                "金标": gold or {"说明": "这批题未逐条写金标，" + (q.get("expected_source") or "")},
                "出处": src or {"说明": q.get("expected_source")},
                "召回表现": recall_view(arec, q),
                "答案表现": answer_view(arec, q, judged_after),
                "答案引用块": cited_blocks(arec),
                "修复前": {
                    "来源": b["before"][-1] if before else "—（新题，冻结后首跑即修复后）",
                    "判定": judged_before or ("见原文档" if brec else "—"),
                    "说材料不足": bool(brec.get("says_insufficient")) if brec else None,
                    "回答": (brec or {}).get("answer"),
                },
                "修复后": {
                    "来源": b["after"][-1] if after else "（结果文件还没生成）",
                    "判定": judged_after or ("自动：数字可追溯" if (arec and arec.get("auto_pass_hint")) else "—"),
                    "回答": (arec or {}).get("answer"),
                },
            })
    return records


def _clip(s, n=60):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def render_md(records, path):
    L = []
    L.append("# 第 11 步 · 完整逐题记录（三批题 · 修复前后）\n")
    L.append("> 对应老师的两条要求：**能区分开发题与新题**、**补完整逐题记录**。")
    L.append("> 评价纪律按第三课第 90–92 页：召回表现与答案表现分开记，不混成一个分数。\n")

    L.append("## 一、三批题的身份\n")
    L.append("| 批次 | 角色 | 题数 | 为什么这么定 |")
    L.append("|---|---|---|---|")
    for b in BANKS:
        n = sum(1 for r in records if r["批次"] == b["batch"])
        L.append(f"| {b['batch']} | **{b['role']}** | {n} | {b['role_note']} |")
    L.append("")
    L.append("判分只认**第三批（新题）**。第一批是调参时反复看过的题，它的分数只说明有没有退步；")
    L.append("第二批原本是检验题，但第 08–10 步拿它反向调过参，已经不能算「没见过的新题」。\n")

    # ---- 总表 ----
    L.append("## 二、逐题总表\n")
    L.append("| 批次 | 角色 | 题号 | 问题 | 公司覆盖 | 金标块位次 | 修复前 | 修复后 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        rv = r["召回表现"]
        L.append(f"| {r['批次']} | {r['角色']} | {r['题号']} | {_clip(r['问题'], 38)} | "
                 f"{rv.get('公司覆盖', '-')} | {rv.get('金标块位次', '-')} | "
                 f"{_clip(str(r['修复前']['判定']), 12)} | {_clip(str(r['修复后']['判定']), 12)} |")
    L.append("")

    # ---- 成绩 ----
    L.append("## 三、分数（分母是题数，不是检索块数）\n")
    L.append("第三批有冻结金标，可以按「数值 + 单位 + 方向」严格判对错；")
    L.append("第一、二批没有逐条金标，只能用「自动核对」（数字都能在原文找到、公司覆盖齐全）——")
    L.append("它只能证明没有编数，不能证明答的就是问的那件事，所以单列一栏，不混进对外成绩。\n")
    L.append("| 批次 | 角色 | 修复前 | 修复后 | 判分口径 |")
    L.append("|---|---|---|---|---|")
    for b in BANKS:
        rs = [r for r in records if r["批次"] == b["batch"]]
        def cnt(key):
            return sum(1 for r in rs if str(r[key]["判定"]).startswith(("答对", "自动核对通过")))
        n = len(rs)
        before_txt = f"{cnt('修复前')} / {n}" if b["before"] else "—（新题，无修复前）"
        note = "**按冻结金标判，可对外**" if b["role"].startswith("新题") else "自动核对口径，只作回归参考"
        L.append(f"| {b['batch']} | {b['role']} | {before_txt} | {cnt('修复后')} / {n} | {note} |")
    L.append("")

    L.append("**和开发阶段的历史人工判定对照**（那些数字来自各步报告逐题看过的结论）：\n")
    L.append("| 阶段 | 题集 | 历史人工判定 | 出处 |")
    L.append("|---|---|---|---|")
    L.append("| 第01步 基线 | 第一批 | 3 / 10 | 第01步-基线评测-答对3题.md |")
    L.append("| 第06步 优化后 | 第一批 | 7 / 10 | 第06步-优化生成侧-回到7题.md |")
    L.append("| 第07步 换新题 | 第二批 | 4 / 10 | 第07步-换全新题检验-掉到4题.md |")
    L.append("| 第08步 通用兜底 | 第二批 | 7 / 10 | 第08步-陌生科目通用兜底-回到7题.md |")
    L.append("| 第11步 新题检验 | 第三批 | 见上表「修复后」 | 本文件第四节 |")
    L.append("")
    L.append("自动核对比历史人工判定松 1–2 题：数值对、但口径或单位有瑕疵的题，"
             "自动核对会放过（它只看数字在不在原文），人工会算不完整。两者不矛盾，口径不同。\n")

    # ---- 明细 ----
    L.append("## 四、逐题明细\n")
    for r in records:
        rv, av = r["召回表现"], r["答案表现"]
        L.append(f"### {r['批次']} · 第 {r['题号']} 题 · {r['角色']}\n")
        L.append(f"**问题**：{r['问题']}\n")
        g = r["金标"]
        if isinstance(g, dict) and g.get("value"):
            cmp_ = g.get("compare") or {}
            L.append(f"**金标**：{g.get('value')} {g.get('unit') or ''}"
                     + (f"；比较期 {cmp_.get('value')}；变化 {cmp_.get('change')}" if cmp_.get("value") else "")
                     + (f"｜口径：{g.get('caliber_note')}" if g.get("caliber_note") else ""))
        else:
            L.append(f"**金标**：{g.get('说明', '—')}")
        s = r["出处"]
        if isinstance(s, dict) and s.get("report"):
            L.append(f"**出处**：{s.get('report')} 第 {s.get('page')} 页"
                     f"（块 #{s.get('chunk_id')}，{s.get('table') or s.get('section') or ''}）\n")
        else:
            L.append(f"**出处**：{s.get('说明', '—')}\n")
        L.append("**召回表现**（证据找没找到）："
                 f"公司覆盖 {rv.get('公司覆盖', '-')}"
                 + (f"，缺 {'、'.join(rv['缺哪家'])}" if rv.get("缺哪家") else "")
                 + f"；{rv.get('金标块位次', '-')}；共召回 {rv.get('命中块数', '-')} 块\n")
        L.append(f"**答案表现**（找到之后读没读对）：{av.get('判定', '-')}"
                 + (f"｜数字可追溯 {av['数字可追溯']}" if av.get("数字可追溯") else "") + "\n")
        if r.get("答案引用块"):
            L.append("**答案用到哪几块**（答错时先看这一行）：" + "；".join(r["答案引用块"]) + "\n")
        if r["修复前"].get("回答"):
            L.append(f"**修复前**（{r['修复前']['来源']}）：{r['修复前']['判定']}\n")
            L.append(f"> {_clip(r['修复前']['回答'], 300)}\n")
        if r["修复后"].get("回答"):
            L.append(f"**修复后**（{r['修复后']['来源']}）：{r['修复后']['判定']}\n")
            L.append(f"> {_clip(r['修复后']['回答'], 300)}\n")

    L.append("## 五、怎么核对这张表\n")
    L.append("1. 随便挑一题，按「出处」的页码去 PDF 里看那一行，数值应当和金标一字不差；")
    L.append("2. 「金标块位次」是系统有没有把那一块检索出来，和答案对错无关——")
    L.append("   位次写了「未进召回」而答案却对，说明答案是从别处推的，这种要警惕；")
    L.append("3. 「修复前」和「修复后」是同一道题、同一个问题，只差代码；")
    L.append("   两栏判定不一样，才说明那几处修改真的起作用了。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    records = build()
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "n_records": len(records), "records": records}, f, ensure_ascii=False, indent=2)
    render_md(records, OUT_MD)
    for b in BANKS:
        rs = [r for r in records if r["批次"] == b["batch"]]
        done_b = sum(1 for r in rs if str(r["修复前"]["判定"]).startswith(("答对", "自动核对通过")))
        done_a = sum(1 for r in rs if str(r["修复后"]["判定"]).startswith(("答对", "自动核对通过")))
        # 新题是冻结之后才第一次跑，本来就没有「修复前」，别印成 0/10 误导人
        before_txt = f"{done_b}/{len(rs)}" if b["before"] else "—（新题，无修复前）"
        print(f"{b['batch']}｜{b['role']}｜修复前 {before_txt} → 修复后 {done_a}/{len(rs)}")
    print(f"\n机读：{os.path.relpath(OUT_JSON, ROOT)}")
    print(f"人读：{os.path.relpath(OUT_MD, ROOT)}")


if __name__ == "__main__":
    main()
