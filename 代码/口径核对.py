# -*- coding: utf-8 -*-
"""口径核对：单位 / 负号 / 指标口径 三项检查，能做确定性修复的就直接修。

老师反馈里点名的三类优先错误，正好对应三个函数：

    单位      原文「单位：元」，回答写成「万元」        → check_units()
    负号      原文 +7.07%，回答写成「同比下滑」          → check_signs()
    指标别名   问扣非、答归母；问稀释、答基本            → check_caliber()

三个都遵守同一条纪律：
  * **只在证据无歧义时才动手**。数字在原文里精确对上、且只对应一行，才修；
    对上多行 / 对不上，只报「存疑」，不改答案 —— 宁可留着让人看，也不猜。
  * 每一处改动都留「改前 / 改后 / 依据（块编号+页码+原文行）」，写进核对报告，
    这样「修复前后记录」是可核查的，不是我说改了就算改了。
  * 只做「单位词替换 / 方向词替换 / 补单位」这三种同义改写，
    **不换算数值、不跨行取数、不改结论**。

对外入口：
    check(question, answer, hits) -> dict     # 报告 + 修复后的回答
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts = os.path.join(ROOT, "代码")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from 口径字典 import (  # noqa: E402
    caliber, detect, distinct_pattern, distinct_terms, row_pattern, same_terms, unit_of,
)
from 引用追溯 import SENT_SPLIT_RE, _norm, extract_numbers, parse_cites  # noqa: E402

# 表格里的「单位：元 / 单位：人民币元 / 单位：万元」
UNIT_DECL_RE = re.compile(r"单位\s*[：:]\s*(人民币)?\s*([^\s|，。；、]{1,8})")
# 回答里的「数字 + 单位」（单位可有可无）
NUM_UNIT_RE = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|千元|元/股|元|个百分点|%|千升|吨|股|家)?"
)

UNITS = ("亿元", "万元", "千元", "元/股", "元", "个百分点", "%", "千升", "吨", "股", "家")

# 方向词
UP_WORDS = ("增长", "增加", "上升", "提高", "上涨", "正增长")
DOWN_WORDS = ("下滑", "下降", "减少", "降低", "回落", "负增长")

# 单位词替换时，同一个逻辑单位的等价写法（原文可能写「人民币元」）
UNIT_ALIAS = {"人民币元": "元", "元/股": "元/股"}

# 单位分族：只有同族之间才允许替换单位词。
# 否则「同比 7.07%」会被表格表头的「单位：元」改成「7.07 元」——这是我在自测里
# 真实踩到的坑，所以这条守卫必须有。
UNIT_FAMILY = {
    "千元": "currency", "万元": "currency", "亿元": "currency", "元": "currency",
    "%": "ratio", "个百分点": "ratio",
    "元/股": "per_share",
    "千升": "volume", "吨": "volume",
    "股": "shares", "家": "count",
}


def _family(u):
    return UNIT_FAMILY.get(u, "?")


def _same_family(a, b):
    fa, fb = _family(a), _family(b)
    return fa != "?" and fa == fb


def _digits(raw):
    m = re.match(r"-?(\d[\d,]*(?:\.\d+)?)", raw or "")
    return m.group(1).replace(",", "") if m else ""


def _num_unit(raw):
    """从一个数字串里拆出（数值部分，单位词）。"""
    m = NUM_UNIT_RE.fullmatch((raw or "").strip())
    if not m:
        return _digits(raw), None
    return m.group(1).replace(",", ""), m.group(2)


def _declared_unit(lines, limit=25):
    """块里声明的单位。表格一般在块首几行写「单位：元」。"""
    for ln in lines[:limit]:
        m = UNIT_DECL_RE.search(ln)
        if m:
            u = (m.group(1) or "") + m.group(2)
            u = u.strip()
            return UNIT_ALIAS.get(u, u)
    return None


def _lines(hit):
    return [ln.strip() for ln in hit["text"].split("\n") if ln.strip()]


def _sibling_unit(h, hits):
    """本块没写单位时，看同一份报告同一页（±1 页）的其它召回块写没写。

    表格被切块时，「单位：元」常常落在上一块表头上。同页相邻块的声明是原文里
    真实存在的标注，比「按惯例一律是元」更硬；但只在这一页上只有一种说法时才采用，
    出现两种单位仍然不动。
    """
    us = set()
    for x in hits:
        if (x.get("company"), x.get("year"), x.get("rtype")) != (h.get("company"), h.get("year"), h.get("rtype")):
            continue
        if abs((x.get("page") or 0) - (h.get("page") or 0)) > 1:
            continue
        u = _declared_unit(_lines(x))
        if u:
            us.add(u)
    return us.pop() if len(us) == 1 else None


def _find_lines(num_digits, hit_lines, context=0):
    """在块里找出现这个数字的行（按数字边界匹配，避免 7.07 命中 17.07）。"""
    if not num_digits:
        return []
    core = re.escape(num_digits)
    pat = re.compile(r"(?<![\d.])" + core + r"(?![\d])")
    out = []
    for i, ln in enumerate(hit_lines):
        if pat.search(ln.replace(",", "")):
            if context:
                lo, hi = max(0, i - context), min(len(hit_lines), i + context + 1)
                out.append((i, "\n".join(hit_lines[lo:hi]), ln))
            else:
                out.append((i, ln, ln))
    return out


def _longest_hit(terms, line):
    """这一行里出现了哪个词、多长。用来按「最长匹配」判口径。"""
    best = ""
    for t in terms:
        if t and t in line and len(t) > len(best):
            best = t
    return best


def _label_status(key, line):
    """这一行是不是目标口径。返回 same / distinct / unknown。

    按**最长匹配**判，而不是「先看 same 再看 distinct」：
      · 「营业总成本 | 8,283,201,155.62」里同时含「营业成本」，
        但含的是更长的「营业总成本」→ 判 distinct；
      · 「扣除非经常性损益的净利润」里也含「净利润」，
        但含的是更长的「扣除非经常性损益的净利润」→ 判 same。
    单靠先后顺序，这两条必有一条判错——题库3 第 5 题就是这样漏掉的。
    """
    same = _longest_hit(same_terms(key), line)
    dist = _longest_hit(distinct_terms(key), line)
    if same and len(same) >= len(dist):
        return "same"
    if dist:
        return "distinct"
    if same:
        return "same"
    return "unknown"


def _line_sign(line, num_digits):
    """从原文行判断这个数字的正负号。返回 -1 / +1 / 0（无符号信息）。"""
    core = re.escape(num_digits)
    m = re.search(r"([-－+＋])?\s*" + core, line.replace(",", ""))
    if m:
        sign = m.group(1)
        if sign in ("-", "－"):
            return -1
        if sign in ("+", "＋"):
            return 1
    # 没有显式符号：看紧邻的文字
    window = line.replace(",", "")
    pos = window.find(num_digits)
    if pos >= 0:
        left = window[max(0, pos - 10):pos]
        if any(w in left for w in ("减少", "下降", "下滑", "降低", "回落")):
            return -1
        if any(w in left for w in ("增加", "增长", "上升", "提高")):
            return 1
    return 0


def _replace_once(text, old, new):
    i = text.find(old)
    if i < 0:
        return text, False
    return text[:i] + new + text[i + len(old):], True


def _iter_numbers(answer):
    """按句返回（句子原文，数字串列表）。"""
    for raw in SENT_SPLIT_RE.split(answer or ""):
        s = raw.strip()
        if s:
            yield s, extract_numbers(s)


def _cited_blocks(sentence, hits):
    ns = parse_cites(sentence)
    out = [h for h in hits if h["rank"] in ns]
    return out or list(hits)


def check_units(question, answer, hits, repairs):
    """单位核对：数字精确对上、且原文声明了单位时，把回答里的单位词改对。"""
    issues = []
    for s, nums in _iter_numbers(answer):
        blocks = _cited_blocks(s, hits)
        for raw_num in nums:
            digits, ans_unit = _num_unit(raw_num)
            if not digits:
                continue
            cands = []
            for h in blocks:
                lines = _lines(h)
                units = _declared_unit(lines) or _sibling_unit(h, hits)
                for _, line, _ in _find_lines(digits, lines):
                    cands.append((h, units, line))
            if not cands:
                # 回答把自己标的引用块写错了：这个数不在它引的块里。
                # 引用错要单独记，但单位仍可依据「真正含这个数的召回块」来核对 ——
                # 不回退的话，单位错误就会因为「引错块」而一起漏过去。
                for h in hits:
                    if h in blocks:
                        continue
                    lines = _lines(h)
                    units = _declared_unit(lines) or _sibling_unit(h, hits)
                    for _, line, _ in _find_lines(digits, lines):
                        cands.append((h, units, line))
            if not cands:
                continue
            declared = {u for _, u, _ in cands if u}
            if not declared:
                # 召回块里根本没写「单位：元」——表格被切块时表头常常落在上一块。
                # 这种情况下不改数字：回答若写的是万元/亿元这种「放大过的单位」，
                # 就把风险说出来让人看；写「元」的按惯例放行。
                if ans_unit in ("万元", "千元", "亿元") and len(digits) >= 6:
                    issues.append({
                        "num": raw_num, "status": "unlabeled",
                        "detail": f"含这个数的表格块没写单位，回答却写「{ans_unit}」；"
                                  f"财报表格未注明单位时按惯例是「元」，需人工确认",
                    })
                continue
            if len(declared) != 1:
                continue                      # 同一条数字对应两种单位 → 不猜
            src_unit = declared.pop()
            if ans_unit is None:
                # 回答漏了单位，而原文明确是金额单位 → 补上（只补金额，不补 % 和 家）
                if src_unit in ("元", "万元", "亿元") and len(digits) >= 6:
                    old = raw_num
                    new = f"{raw_num}{src_unit}"
                    answer, ok = _replace_once(answer, old, new)
                    if ok:
                        h, _, line = cands[0]
                        repairs.append({
                            "type": "unit", "action": "补单位",
                            "before": old, "after": new,
                            "reason": f"原文声明单位「{src_unit}」",
                            "chunk": h["id"], "page": h["page"], "line": line[:120],
                        })
                        issues.append({"num": raw_num, "status": "fixed", "detail": f"补单位 {src_unit}"})
                continue
            if ans_unit == src_unit:
                issues.append({"num": raw_num, "status": "ok", "detail": f"单位一致（{src_unit}）"})
                continue
            if not _same_family(ans_unit, src_unit):
                # 百分比 vs 金额、元/股 vs 元：不是同一个东西，绝不替换
                issues.append({"num": raw_num, "status": "skip",
                               "detail": f"回答「{ans_unit}」与原文「{src_unit}」不同类，不替换"})
                continue
            # 同族单位词不同、且数字完全一样 → 以原文单位为准，只换单位词
            old = raw_num
            new = raw_num.replace(ans_unit, src_unit)
            if new == old:
                new = f"{digits}{src_unit}"
            answer, ok = _replace_once(answer, old, new)
            if ok:
                h, _, line = cands[0]
                repairs.append({
                    "type": "unit", "action": "改单位词",
                    "before": old, "after": new,
                    "reason": f"原文声明单位「{src_unit}」，回答写成「{ans_unit}」",
                    "chunk": h["id"], "page": h["page"], "line": line[:120],
                })
                issues.append({"num": raw_num, "status": "fixed",
                               "detail": f"{ans_unit} → {src_unit}"})
            else:
                issues.append({"num": raw_num, "status": "suspect",
                               "detail": f"原文单位「{src_unit}」，回答「{ans_unit}」"})
    return answer, issues


def check_signs(question, answer, hits, repairs):
    """负号核对：方向词与原文同比值的正负号打架时，改方向词（只改词，不改数字）。"""
    issues = []
    for s, nums in _iter_numbers(answer):
        has_up = any(w in s for w in UP_WORDS)
        has_down = any(w in s for w in DOWN_WORDS)
        if has_up == has_down:                # 两种都有 / 都没有 → 不碰
            continue
        # 只对「百分比」类数字做符号判断，避免把「净利润 12.3 亿元」当成变化率
        pct = [n for n in nums if ("%" in n or "个百分点" in n)]
        if len(pct) != 1:
            continue
        raw_num = pct[0]
        digits, _ = _num_unit(raw_num)
        blocks = _cited_blocks(s, hits)
        signs = set()
        evidence = None
        for h in blocks:
            for _, line, full in _find_lines(digits, _lines(h)):
                sg = _line_sign(full, digits)
                if sg:
                    signs.add(sg)
                    evidence = (h, full)
        if len(signs) != 1:
            continue
        src_sign = signs.pop()
        said = "up" if has_up else "down"
        wrong = (src_sign > 0 and said == "down") or (src_sign < 0 and said == "up")
        if not wrong:
            issues.append({"num": raw_num, "status": "ok", "detail": "方向与原文符号一致"})
            continue
        words = DOWN_WORDS if said == "down" else UP_WORDS
        repl = {"下滑": "增长", "下降": "增长", "减少": "增加", "降低": "提高", "回落": "回升",
                "负增长": "正增长", "增长": "下滑", "增加": "减少", "上升": "下降",
                "提高": "降低", "上涨": "下跌", "正增长": "负增长"}
        done = False
        for w in words:
            if w in s:
                old, new = w, repl[w]
                answer, ok = _replace_once(answer, old, new)
                if ok:
                    h, line = evidence
                    repairs.append({
                        "type": "sign", "action": "改方向词",
                        "before": old, "after": new,
                        "reason": f"原文 {raw_num} 的符号为{'正' if src_sign > 0 else '负'}",
                        "chunk": h["id"], "page": h["page"], "line": line[:120],
                    })
                    issues.append({"num": raw_num, "status": "fixed", "detail": f"{old} → {new}"})
                    done = True
                break
        if not done:
            issues.append({"num": raw_num, "status": "suspect",
                           "detail": f"方向词与原文符号（{'正' if src_sign > 0 else '负'}）矛盾"})
    return answer, issues


def check_caliber(question, answer, hits):
    """指标口径核对：回答用的是不是问句要的那一行。只报不改（换行等于换数，必须人工确认）。"""
    key, alias = detect(question)
    if not key:
        return {"key": None, "issues": []}, None
    issues = []
    for s, nums in _iter_numbers(answer):
        blocks = _cited_blocks(s, hits)
        for raw_num in nums:
            digits, _ = _num_unit(raw_num)
            if not digits or len(digits) < 4:
                continue
            statuses = []
            evidence = None
            for h in blocks:
                for _, line, full in _find_lines(digits, _lines(h)):
                    st = _label_status(key, full)
                    statuses.append(st)
                    if evidence is None or st == "distinct":
                        evidence = (h, full, st)
            if not statuses:
                continue
            if "same" in statuses:
                issues.append({"num": raw_num, "status": "ok",
                               "detail": f"取自「{key}」对应行"})
            elif "distinct" in statuses:
                h, line, _ = evidence
                issues.append({
                    "num": raw_num, "status": "wrong_caliber",
                    "detail": f"问的是「{key}」，但这个数取自另一口径的行：{line[:80]}",
                    "chunk": h["id"], "page": h["page"],
                })
            else:
                issues.append({"num": raw_num, "status": "unlabeled",
                               "detail": "所在行看不出科目名（表格被切块断开），无法判定口径"})
    return {"key": key, "alias": alias, "unit": unit_of(key), "issues": issues}, key


def note(report):
    """把需要人看的问题拼成一句提示（不需要时返回空串）。

    同一个数在回答里出现两次（前后句各提一次），只需提醒一次，所以按行去重。
    """
    lines = []
    for it in report["units"]:
        if it["status"] in ("suspect", "unlabeled"):
            lines.append(f"· 数字 {it['num']}：{it['detail']}")
    for it in report["signs"]:
        if it["status"] == "suspect":
            lines.append(f"· 数字 {it['num']}：{it['detail']}")
    for it in report["caliber"]["issues"]:
        if it["status"] in ("wrong_caliber", "unlabeled"):
            lines.append(f"· 数字 {it['num']}：{it['detail']}")
    if not lines:
        return ""
    seen, uniq = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)
    return "口径核对：以下数字需要人工确认（未自动修改）：\n" + "\n".join(uniq)


def check(question, answer, hits):
    """三项一起做。返回 (修复后的回答, 报告)。"""
    repairs = []
    answer, unit_issues = check_units(question, answer, hits, repairs)
    answer, sign_issues = check_signs(question, answer, hits, repairs)
    cal, key = check_caliber(question, answer, hits)
    report = {
        "caliber": cal,
        "units": unit_issues,
        "signs": sign_issues,
        "repairs": repairs,
        "n_repaired": len(repairs),
    }
    return answer, report


if __name__ == "__main__":
    # 自测：给一段「单位写错 + 方向写反」的回答，看能不能被改回来
    import json

    hits = [{
        "rank": 1, "id": 1, "company": "某公司", "code": "000000", "year": 2025,
        "rtype": "年报", "section": "第三节", "page": 10, "block_type": "table",
        "text": "【某公司｜2025年年报｜第三节｜第10页】\n[表格]\n单位：元\n"
                "主要财务指标 | 2025年 | 2024年 | 本期比上年同期增减(%)\n"
                "营业收入 | 158.94 | 147.00 | 7.07\n"
                "归属于上市公司股东的净利润 | 20.00 | 18.00 | 11.11\n"
                "扣除非经常性损益的净利润 | 19.00 | 17.00 | 11.76\n",
        "vector": 0.9, "bm25": 30.0,
    }]
    for q, a in [
        ("某公司2025年营业收入是多少？同比如何？", "营业收入158.94万元[1]。同比下滑7.07%[1]。"),
        ("某公司2025年扣非归母净利润是多少？", "扣非归母净利润20.00元[1]。"),
    ]:
        fixed, rep = check(q, a, hits)
        print("问题:", q)
        print("原文:", a)
        print("修复:", fixed)
        print("报告:", json.dumps(rep, ensure_ascii=False, indent=1))
        print("提示:", note(rep) or "（无）")
        print("-" * 60)
