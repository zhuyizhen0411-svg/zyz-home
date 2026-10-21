# -*- coding: utf-8 -*-
"""抓取 12 家白酒公司的年报 / 半年报公告索引（只抓索引，不下载 PDF）。

数据源：
    深交所 -> www.szse.cn 官方公告 API（交易所官网）
    上交所 -> www.cninfo.com.cn 巨潮资讯网（证监会指定法定披露平台）；
              上交所官网公告查询接口（query.sse.com.cn）改版后已停用，返回空结果。

用法:
    python 代码/fetch_index.py

输出:
    数据/公告索引/announcements.json  原始候选公告
    数据/公告索引/selected.json       每家公司每个报告期选定的 1 份全文
"""

import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 公司清单 import (  # noqa: E402
    COMPANIES,
    ANNUAL_YEARS,
    INTERIM_YEARS,
    SE_BEGIN,
    SE_END,
    ROOT,
)

INDEX_DIR = os.path.join(ROOT, "数据", "公告索引")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------- 标题判定 ----------
# 标题写法有「2025年年度报告」与「2022年度报告」两种，故「年」设为可选
KEEP_RE = re.compile(r"(20\d{2})\s*年?\s*(半)?年度报告")
BAD_WORDS = (
    "摘要", "英文", "取消", "问询", "业绩预告", "业绩快报", "提示性", "审核报告",
    "核查", "专项", "审计报告", "关于", "补充", "说明", "编制", "勘误", "差异",
)


def norm(s):
    return (s or "").replace(" ", "").replace("\u3000", "").replace("\xa0", "")


def parse_title(title):
    """从公告标题解析 (报告期年份, 年报/半年报)；非定期报告全文返回 None。"""
    t = norm(title)
    if any(w in t for w in BAD_WORDS):
        return None
    m = KEEP_RE.search(t)
    if not m:
        return None
    year = int(m.group(1))
    rtype = "半年报" if m.group(2) else "年报"
    return year, rtype


# ---------- 深交所 ----------
SZSE_API = "https://www.szse.cn/api/disc/announcement/annList"
SZSE_PDF = "http://disc.szse.cn/download"


def szse_list(code):
    """拉取深交所某公司窗口期内的全部公告（官方接口单页上限 50 条，需翻页）。"""
    sess = requests.Session()
    out, page = [], 1
    while page <= 20:
        body = {
            "seDate": [SE_BEGIN, SE_END],
            "stock": [code],
            "channelCode": ["listedNotice_disc"],
            "pageSize": 50,
            "pageNum": page,
        }
        r = sess.post(
            SZSE_API,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": "http://www.szse.cn/disclosure/listed/fixed_disc.html",
                "User-Agent": UA,
                "X-Request-Type": "ajax",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data") or []
        out.extend(data)
        if len(data) < 50:
            break
        page += 1
        time.sleep(0.4)
    recs = []
    for a in out:
        if (a.get("attachFormat") or "").upper() != "PDF":
            continue
        recs.append(
            {
                "title": norm(a.get("title")),
                "publish_date": (a.get("publishTime") or "")[:10],
                "url": SZSE_PDF + (a.get("attachPath") or ""),
                "source": "深交所官网 www.szse.cn",
                "size_kb": a.get("attachSize"),
            }
        )
    return recs


# ---------- 巨潮（沪市） ----------
CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_SEARCH = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
CNINFO_HEADERS = {
    "User-Agent": UA,
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
}
CATEGORIES = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
}


def cninfo_orgid(code):
    # 注意：该接口只接受 POST，GET 会返回 500「系统异常」
    r = requests.post(
        CNINFO_SEARCH,
        data={"keyWord": code, "maxNum": "10"},
        headers=CNINFO_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    for it in r.json() or []:
        if (it.get("code") or "").endswith(code):
            return it.get("orgId")
    return None


def cninfo_list(code, org_id, rtype):
    out = []
    for page in (1, 2, 3):
        payload = {
            "pageNum": str(page),
            "pageSize": "30",
            "column": "sse",
            "tabName": "fulltext",
            "plate": "",
            "stock": f"{code},{org_id}",
            "searchkey": "",
            "secid": "",
            "category": CATEGORIES[rtype],
            "trade": "",
            "seDate": f"{SE_BEGIN}~{SE_END}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        r = requests.post(CNINFO_QUERY, headers=CNINFO_HEADERS, data=payload, timeout=30)
        r.raise_for_status()
        anns = r.json().get("announcements") or []
        for a in anns:
            url = a.get("adjunctUrl") or ""
            if not url.lower().endswith(".pdf"):
                continue
            ts = a.get("announcementTime")
            out.append(
                {
                    "title": norm(a.get("announcementTitle")),
                    "publish_date": time.strftime("%Y-%m-%d", time.localtime(ts / 1000)) if ts else "",
                    "url": CNINFO_STATIC + url,
                    "source": "巨潮资讯网 www.cninfo.com.cn（证监会指定披露平台）",
                    "size_kb": None,
                }
            )
        if len(anns) < 30:
            break
        time.sleep(0.4)
    return out



def collect(company):
    code, name, market = company["code"], company["name"], company["market"]
    recs = []
    if market == "深交所":
        recs = szse_list(code)
        origin = "深交所官网公告 API"
    else:
        org = cninfo_orgid(code)
        for rtype in ("年报", "半年报"):
            recs.extend(cninfo_list(code, org, rtype))
        origin = "巨潮资讯网公告 API"
    cands = []
    for r in recs:
        pt = parse_title(r["title"])
        if not pt:
            continue
        cands.append({**r, "year": pt[0], "rtype": pt[1], "code": code, "name": name, "market": market})
    print(f"  {name}({code}) {origin}: 公告 {len(recs)} 条，定期报告全文候选 {len(cands)} 条")
    return cands


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    all_cands, selected = [], []
    for c in COMPANIES:
        try:
            cands = collect(c)
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {c['name']}: {e}")
            cands = []
        all_cands.extend(cands)

        for rtype, years in (("年报", ANNUAL_YEARS), ("半年报", INTERIM_YEARS)):
            for y in years:
                same = [x for x in cands if x["rtype"] == rtype and x["year"] == y]
                if not same:
                    print(f"    [miss] {c['name']} {y} {rtype}")
                    continue
                # 同一报告期若有多份（原文 / 修订版 / 更新后），取披露时间最新的一份
                same.sort(key=lambda x: x["publish_date"])
                pick = same[-1]
                pick["period"] = f"{y}年{'半' if rtype == '半年报' else ''}年度报告"
                selected.append(pick)

    with open(os.path.join(INDEX_DIR, "announcements.json"), "w", encoding="utf-8") as f:
        json.dump(all_cands, f, ensure_ascii=False, indent=2)
    with open(os.path.join(INDEX_DIR, "selected.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"\n候选 {len(all_cands)} 条，选定 {len(selected)} 份（目标 {len(COMPANIES) * (len(ANNUAL_YEARS) + len(INTERIM_YEARS))} 份）")


if __name__ == "__main__":
    main()
