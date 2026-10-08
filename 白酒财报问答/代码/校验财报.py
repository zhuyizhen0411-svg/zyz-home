# -*- coding: utf-8 -*-
"""校验已下载 PDF：页数、是否可提取正文（防止下到扫描件/残缺件）。

用法:
    python 代码/verify_pdfs.py
"""

import json
import os
import sys

import pdfplumber

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 公司清单 import ROOT  # noqa: E402

MANIFEST = os.path.join(ROOT, "数据", "财报原文", "manifest.json")


def main():
    recs = json.load(open(MANIFEST, encoding="utf-8"))
    for r in recs:
        p = r.get("path")
        if not p or not os.path.exists(p):
            r["pages"] = None
            r["chars_per_page"] = None
            r["verify"] = "missing"
            continue
        try:
            with pdfplumber.open(p) as pdf:
                n = len(pdf.pages)
                # 等距抽 9 页：封面/财报表格页天然文字少，抽样太少会误判
                idxs = sorted({min(n - 1, round(i * (n - 1) / 8)) for i in range(9)})
                chars = 0
                for i in idxs:
                    chars += len((pdf.pages[i].extract_text() or "").strip())
                cpp = chars / len(idxs)
            r["pages"] = n
            r["chars_per_page"] = round(cpp)
            r["verify"] = "ok" if cpp >= 150 else "suspicious(可能是扫描件/正文极少)"
        except Exception as e:  # noqa: BLE001
            r["pages"] = None
            r["chars_per_page"] = None
            r["verify"] = f"error: {e}"
        print(f"  {r['name']:<6}{r['year']} {r['rtype']:<4} {str(r['pages']):>5}页 "
              f"{str(r['chars_per_page']):>6}字/页  {r['verify']}")

    json.dump(recs, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    from download_reports import write_manifest  # 页数回填后重刷清单
    write_manifest(recs)
    bad = [r for r in recs if r.get("verify") != "ok"]
    total_pages = sum(r.get("pages") or 0 for r in recs)
    print(f"\n共 {len(recs)} 份，总页数 {total_pages}，异常 {len(bad)} 份")
    for r in bad:
        print("  异常：", r["name"], r["year"], r["rtype"], r.get("verify"))


if __name__ == "__main__":
    main()
