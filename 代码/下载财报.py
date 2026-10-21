# -*- coding: utf-8 -*-
"""按索引下载 12 家白酒公司的年报 / 半年报全文 PDF。

目录结构：数据/财报原文/<公司名>/<年份>/<年度报告|半年度报告>.pdf
校验：文件头必须为 %PDF，且年报 >= 500KB、半年报 >= 300KB（过小判为摘要/残缺，标记 warning）

用法:
    python 代码/download_reports.py
"""

import concurrent.futures as cf
import csv
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 公司清单 import ROOT  # noqa: E402

INDEX = os.path.join(ROOT, "数据", "公告索引", "selected.json")
OUT_DIR = os.path.join(ROOT, "数据", "财报原文")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MIN_BYTES = {"年报": 500 * 1024, "半年报": 300 * 1024}


def target_path(rec):
    folder = os.path.join(OUT_DIR, rec["name"], str(rec["year"]))
    fname = "年度报告.pdf" if rec["rtype"] == "年报" else "半年度报告.pdf"
    return os.path.join(folder, fname)


def fetch(rec, tries=3):
    path = target_path(rec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 100 * 1024:
        rec["status"] = "skip"
        rec["path"] = path
        rec["size_bytes"] = os.path.getsize(path)
        return rec

    last = None
    for i in range(tries):
        try:
            r = requests.get(
                rec["url"],
                headers={"User-Agent": UA, "Referer": rec["url"].split("/download")[0] + "/"},
                timeout=180,
                stream=True,
            )
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            os.replace(tmp, path)
            last = None
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))

    if last is not None:
        rec["status"] = "fail"
        rec["error"] = str(last)
        rec["path"] = path
        return rec

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(5)
    rec["path"] = path
    rec["size_bytes"] = size
    if head != b"%PDF-":
        rec["status"] = "fail"
        rec["error"] = "不是 PDF 文件"
    elif size < MIN_BYTES[rec["rtype"]]:
        rec["status"] = "warning"
        rec["error"] = f"文件偏小（{size/1024:.0f}KB），可能不是完整全文"
    else:
        rec["status"] = "ok"
    return rec


def write_manifest(recs):
    ok = [r for r in recs if r.get("status") in ("ok", "skip", "warning")]
    recs_sorted = sorted(recs, key=lambda r: (r["name"], r["year"], r["rtype"]))

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(recs_sorted, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "文件清单.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["公司", "代码", "市场", "报告期", "报告类型", "披露日期", "来源", "文件路径", "页数", "大小(MB)", "状态", "原文URL"])
        for r in recs_sorted:
            w.writerow([
                r["name"], r["code"], r["market"], f"{r['year']}年", r["rtype"], r.get("publish_date", ""),
                r["source"], os.path.relpath(r.get("path", ""), ROOT), r.get("pages", ""),
                f"{(r.get('size_bytes') or 0)/1024/1024:.2f}", r.get("status", ""), r["url"],
            ])

    lines = ["# 财报 PDF 文件清单", "",
             f"- 公司：12 家 A 股白酒上市公司",
             f"- 报告：年报 {sum(1 for r in recs_sorted if r['rtype']=='年报')} 份 + 半年报 {sum(1 for r in recs_sorted if r['rtype']=='半年报')} 份"
             f" = 共 {len(recs_sorted)} 份",
             f"- 下载成功：{sum(1 for r in recs_sorted if r.get('status') in ('ok','skip'))} 份；"
             f"异常：{sum(1 for r in recs_sorted if r.get('status') not in ('ok','skip'))} 份",
             f"- 总体积：{sum(r.get('size_bytes') or 0 for r in recs_sorted)/1024/1024:.0f} MB；"
             f"总页数：{sum(r.get('pages') or 0 for r in recs_sorted)} 页", "",
             "## 数据来源", "",
             "| 市场 | 来源 | 说明 |", "|---|---|---|",
             "| 深交所（6 家） | 深交所官网 www.szse.cn 公告 API | 交易所官网原始 PDF |",
             "| 上交所（6 家） | 巨潮资讯网 www.cninfo.com.cn | 证监会指定法定披露平台；上交所公告查询接口已停用 |", "",
             "全部为年报/半年报**全文**PDF，已排除摘要版、英文版、审计报告、更正/问询类公告。", ""]

    by = {}
    for r in recs_sorted:
        by.setdefault(r["name"], []).append(r)
    for name, items in by.items():
        head = items[0]
        lines += [f"## {name}（{head['code']}·{head['market']}）", "",
                  f"来源：{head['source']}", "",
                  "| 报告期 | 类型 | 披露日期 | 页数 | 文件 | 大小 | 状态 |", "|---|---|---|---|---|---|---|"]
        for r in items:
            lines.append(
                f"| {r['year']}年 | {r['rtype']} | {r.get('publish_date','')} | {r.get('pages','')} | "
                f"`{os.path.relpath(r.get('path',''), ROOT)}` | "
                f"{(r.get('size_bytes') or 0)/1024/1024:.2f} MB | {r.get('status','')} |"
            )
        lines.append("")

    with open(os.path.join(OUT_DIR, "文件清单.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n成功/跳过 {len(ok)}，失败 {len(recs_sorted)-len(ok)}")


def main():
    recs = json.load(open(INDEX, encoding="utf-8"))
    print(f"待下载 {len(recs)} 份 → {OUT_DIR}")
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(fetch, recs))
    for r in out:
        flag = {"ok": "[ ok ]", "skip": "[skip]", "warning": "[warn]", "fail": "[fail]"}[r.get("status", "fail")]
        print(f"  {flag} {r['name']:<6}{r['year']} {r['rtype']:<4} "
              f"{(r.get('size_bytes') or 0)/1024/1024:6.2f}MB  {r.get('error','')}")
    write_manifest(out)


if __name__ == "__main__":
    main()
