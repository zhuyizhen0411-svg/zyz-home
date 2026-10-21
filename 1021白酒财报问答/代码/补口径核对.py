# -*- coding: utf-8 -*-
"""补口径核对：把已经跑完的回答，用**修好的**口径规则再看一遍。

为什么不重跑大模型
------------------
口径核对是「生成完之后的一道检查」，它不改模型怎么想，只改「检查有没有看出来」。
所以修了规则之后，不必再让模型把 10 道题重答一遍（那既慢，又会引入新的随机性），
只要把当时召回的那几块原文重新取出来，让新规则对同一份回答再判一次即可。

步骤：
    ① 读结果 JSON（里面有每题召回块的 id、页码、公司）
    ② 按 id 把原文块取回来（注意 id 与下标差 1，见下方 POS 注释）
    ③ 去掉旧回答末尾「材料不足 / 口径核对」提示段
    ④ 用 check() 重判：单位、正负号、指标口径
    ⑤ 写出一份新的结果文件，每题留下「旧提示 → 新提示」的变化

输出默认与输入同名加 `-口径复核` 后缀。

命令行：
    python 代码/补口径核对.py --result 数据/评测/结果/结果07-第11步-题库3冻结金标检验.json
    python 代码/补口径核对.py --result ... --out ... --dry-run
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "代码"))

from 检索引擎 import Retriever  # noqa: E402
from 口径核对 import check, note  # noqa: E402

# 这里只复核「口径核对」这一道检查，所以只剥它自己的提示段。
# 「材料不足：…」是另一道检查（引用追溯）加的，不归这里管，剥了就会误报成「提示消失」。
CALIBER_MARK = "口径核对：以下数字需要人工确认"
SHORTAGE_MARK = "材料不足：以下数字无法追溯到原文"


def split_notes(ans):
    """把回答拆成（正文, 材料不足提示, 口径提示）。"""
    out = ans or ""
    shortage = ""
    i = out.find(SHORTAGE_MARK)
    if i >= 0:
        j = out.find(CALIBER_MARK)
        shortage = out[i:j].strip() if j > i else out[i:].strip()
    j = out.find(CALIBER_MARK)
    caliber = out[j:].strip() if j >= 0 else ""
    cut = min([k for k in (i if i >= 0 else len(out), j if j >= 0 else len(out))])
    return out[:cut].strip(), shortage, caliber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="要复核的结果 JSON")
    ap.add_argument("--out", default="", help="输出路径，默认在同目录加「-口径复核」")
    ap.add_argument("--dry-run", action="store_true", help="只打印变化，不写文件")
    a = ap.parse_args()

    src = a.result if os.path.isabs(a.result) else os.path.join(ROOT, a.result)
    out = a.out or os.path.join(
        os.path.dirname(src),
        os.path.splitext(os.path.basename(src))[0] + "-口径复核.json",
    )
    with open(src, encoding="utf-8") as f:
        records = json.load(f)

    r = Retriever(with_model=False)
    # id 与下标的坑：Retriever.texts 按下标排，meta[i]["id"] == i + 1。
    # 直接 texts[chunk_id] 会整体错一位，取到隔壁块。
    POS = {m["id"]: i for i, m in enumerate(r.meta)}

    changed = 0
    for rec in records:
        body, shortage, before = split_notes(rec.get("answer"))
        hits = []
        for h in rec.get("hits") or []:
            cid = h.get("id")
            if cid not in POS:
                continue
            hits.append({**h, "text": r.texts[POS[cid]]})
        answer, report = check(rec.get("question", ""), body, hits)
        after = note(report)
        # 回答正文 + 材料不足提示（原样保留）+ 新的口径提示
        rec["answer_recheck"] = "\n\n".join(x for x in (answer, shortage, after) if x)
        rec["verification_recheck"] = report
        rec["note_before"] = before
        rec["note_after"] = after
        if after != before:
            changed += 1
            print(f"\n第 {rec.get('id')} 题（{rec.get('question', '')[:34]}）")
            print(f"  旧提示：{before.replace(chr(10), ' / ') or '（无）'}")
            print(f"  新提示：{after.replace(chr(10), ' / ') or '（无）'}")

    print(f"\n复核完成：{len(records)} 题，提示有变化 {changed} 题")
    if a.dry_run:
        print("（--dry-run，未写文件）")
        return
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"已写出：{os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
