# -*- coding: utf-8 -*-
"""冻结金标：在跑系统之前，把「标准答案」和「题目」一起钉死。

为什么必须冻结
--------------
如果先跑系统、看了答案，再回头改标准答案，那正确率永远可以做到 100%，
但这个数什么也证明不了。所以顺序只能是：

    出题 → 写金标（数值 + 单位 + 口径 + 出处）→ **冻结（哈希 + 时间戳）** → 跑系统 → 对答案

冻结之后，任何对题目或金标的改动都会被 --verify 查出来，并且能指出是第几题被改了。
这样「新题」才是新题，不是事后补的题。

同时给三批题标明身份（这才是老师说的「能区分开发题与新题」）：
    题库1  开发题：基线 → 前五步优化全程看着它改
    题库2  开发题：本来是检验题，但第08–10步用它反向调参，已经污染
    题库3  检验题：金标先冻结，系统在跑之前没见过

命令行：
    python 代码/冻结金标.py --freeze    # 生成/更新冻结记录
    python 代码/冻结金标.py --verify    # 校验题库有没有被改过
    python 代码/冻结金标.py             # 只看当前状态
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QB = os.path.join(ROOT, "数据", "评测", "题库")
RECORD = os.path.join(QB, "金标冻结记录.json")

BANKS = [
    {
        "file": "题库1-最初的10道题.json",
        "batch": "第一批",
        "role": "开发题",
        "status": "连续用于基线测（第01步）和前五步优化（第02–06步），系统是照着它的错题改的",
        "used_for": "第01–06步",
    },
    {
        "file": "题库2-全新10道题.json",
        "batch": "第二批",
        "role": "开发题（原检验题，已污染）",
        "status": "第07步作为检验题测得 4/10；随后第08–10步用它反向调参（科目兜底、查询改写、每家补召），"
                  "再用它复测得到 7/10。这批题因此不再是「没见过的新题」，按开发题看待",
        "used_for": "第07–10步",
    },
    {
        "file": "题库3-冻结金标-10道题.json",
        "batch": "第三批",
        "role": "检验题",
        "status": "金标逐条抄自原文并冻结；出题、写金标时未运行系统，系统未见过这批题",
        "used_for": "第11步（修复后检验）",
    },
]


def _canon(obj):
    """规范化序列化：键排序、不转义中文，保证哈希可复现。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def question_hash(q):
    """单题哈希只覆盖「题目 + 金标 + 出处」，其余说明性字段不参与。"""
    core = {
        "id": q.get("id"),
        "question": q.get("question"),
        "caliber": q.get("caliber"),
        "gold": q.get("gold"),
        "source": q.get("source"),
    }
    return _sha(_canon(core))


def build_records():
    out = []
    for b in BANKS:
        p = os.path.join(QB, b["file"])
        if not os.path.exists(p):
            out.append({**b, "missing": True})
            continue
        d = load(p)
        qs = d.get("questions", [])
        rec = {
            **b,
            "path": os.path.relpath(p, ROOT),
            "n_questions": len(qs),
            "sha256": _sha(_canon({"questions": [
                {k: q.get(k) for k in ("id", "question", "caliber", "gold", "source")} for q in qs
            ]})),
            "questions": {str(q["id"]): question_hash(q) for q in qs},
        }
        out.append(rec)
    return out


def freeze():
    recs = build_records()
    rec = {
        "freeze_version": 1,
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "先写金标并冻结，再跑系统；冻结后看到的系统表现只报告，不回头改题。"
                "若某题确实有错，只能连同勘误说明一起重新冻结，并在逐题记录里注明改过什么。",
        "hash_scheme": "sha256(json.dumps(按 id/question/caliber/gold/source 排序序列化))",
        "banks": recs,
    }
    with open(RECORD, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"已冻结 → {os.path.relpath(RECORD, ROOT)}")
    for r in recs:
        if r.get("missing"):
            print(f"  ✗ {r['file']}：文件不存在")
        else:
            print(f"  ✓ {r['file']}｜{r['role']}｜{r['n_questions']} 题｜{r['sha256'][:16]}…")
    print(f"  冻结时间：{rec['frozen_at']}")


def verify():
    if not os.path.exists(RECORD):
        print("还没有冻结记录，先运行 --freeze")
        return 2
    old = load(RECORD)
    new = {b["file"]: b for b in build_records()}
    bad = 0
    for ob in old["banks"]:
        nb = new.get(ob["file"])
        if not nb or nb.get("missing"):
            print(f"✗ {ob['file']}：题库文件不见了")
            bad += 1
            continue
        if nb["sha256"] == ob["sha256"]:
            print(f"✓ {ob['file']}：与冻结时一致（{ob['n_questions']} 题）")
            continue
        bad += 1
        print(f"✗ {ob['file']}：题库与冻结时不一致")
        for qid, h in ob["questions"].items():
            nh = nb["questions"].get(qid)
            if nh is None:
                print(f"    · 第 {qid} 题被删除")
            elif nh != h:
                print(f"    · 第 {qid} 题被改动（题目/金标/出处）")
        for qid in nb["questions"]:
            if qid not in ob["questions"]:
                print(f"    · 第 {qid} 题是冻结后新增的")
    print(f"\n冻结时间：{old['frozen_at']}")
    print("结论：" + ("全部一致，金标未被事后修改" if not bad else f"{bad} 个题库与冻结记录不一致"))
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.verify:
        sys.exit(verify())
    else:
        for b in build_records():
            if b.get("missing"):
                print(f"✗ {b['file']}：不存在")
            else:
                print(f"{b['batch']}｜{b['role']}｜{b['n_questions']} 题｜{b['file']}")
                print(f"    {b['status']}")
        print(f"\n冻结记录：{'已存在' if os.path.exists(RECORD) else '尚未生成'}（--freeze 生成，--verify 校验）")


if __name__ == "__main__":
    main()
