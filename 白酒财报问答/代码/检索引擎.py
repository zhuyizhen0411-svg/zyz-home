# -*- coding: utf-8 -*-
"""检索层。

v1：向量 / BM25 / 1:1 权重的 RRF 混合 → 直接返回 top_k
v2：在上面基础上加了两件事（详见 README 的「两轮迭代」一节）
     ① 混合时 BM25 权重更高（财报里的精确术语和数字，词频匹配比语义更可靠）
     ② 两段式：先宽召回 pool_size 块，再按「同源收敛 + 数字密度」重排，最后取 top_k
         —— 解决 v1 里「第八节 财务报告」的大段无数字叙述占满上下文的问题
"""

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "代码")
IDX = os.path.join(ROOT, "检索索引")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from 词频检索 import BM25, get_tokenize  # noqa: E402
from 指标规则 import augment_query, detect_metric, metric_vector, quota  # noqa: E402
from 科目兜底 import augment_generic, generic_vector  # noqa: E402

# ---------------- 重排用的特征 ----------------
# 一个「财务数字」：千分位金额（至少到百万级）或带小数点的百分数
FIG_RE = re.compile(r"\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?|-?\d+\.\d{2}%|\d+\.\d{2}")
ASK_NUM_RE = re.compile(r"多少|占比|比重|增速|增长率|同比|百分之|几个|第几|余额|费用|收入|利润|毛利率|亿元|是多少")


def has_figure(text):
    return FIG_RE.search(text) is not None


def asks_number(query):
    return ASK_NUM_RE.search(query) is not None


# ---------------- 查询意图识别（跨公司路由用） ----------------
# 公司别名表：key 是全称（与语料 meta 里的 company 一致），value 是可能出现的简称
COMPANY_ALIASES = {
    "贵州茅台": ["贵州茅台", "茅台"],
    "五粮液": ["五粮液"],
    "泸州老窖": ["泸州老窖", "泸州", "老窖"],
    "山西汾酒": ["山西汾酒", "汾酒"],
    "洋河股份": ["洋河股份", "洋河"],
    "古井贡酒": ["古井贡酒", "古井贡", "古井"],
    "今世缘": ["今世缘"],
    "迎驾贡酒": ["迎驾贡酒", "迎驾"],
    "口子窖": ["口子窖", "口子"],
    "水井坊": ["水井坊"],
    "舍得酒业": ["舍得酒业", "舍得"],
    "酒鬼酒": ["酒鬼酒", "酒鬼"],
}

# 这类措辞说明问的是「所有公司」，而不是某几家。
# 用正则而不是纯字符串，才能容忍中间插入行业词：「哪些白酒公司」「哪些上市酒企」也算全局问法。
ALL_COMPANY_RE = re.compile(
    r"12\s*家|十二家|十家|全部公司|所有公司|各家|每家|各公司|这些公司|这几家|上述公司|全部上市|"
    r"哪些[^。？，、]{0,6}(公司|企业|酒企|厂商|白酒)|"
    r"谁的[^。？，、]{0,8}(最高|最低|最多|最少)"
)
# 没点名任何公司、但用了行业集合名词：说的就是这个行业的一整组公司
INDUSTRY_SET_RE = re.compile(
    r"白酒公司|白酒企业|白酒行业|白酒板块|白酒上市公司|上市酒企|酒类公司|酒企|上市公司(们|整体)"
)
# 没点名公司、但问的是跨公司枚举/比较（最高、哪些、各家…）——全局混合 TopK 必然覆盖不全，
# 这种情况也应该逐家检索
CROSS_COMPANY_RE = re.compile(
    r"哪些|谁的|哪家|各家|每家|分别|各自|最高|最低|最多|最少|最快|最慢|"
    r"排名|排序|前\s*\d|有多少家|有几家|所有|全部|普遍|均出现|都出现"
)

YEAR_RE = re.compile(r"(20\d{2})\s*年")
HALF_YEAR_RE = re.compile(r"上半年|半年度|半年报|1—6月|1-6月|半年度报")
FULL_YEAR_RE = re.compile(r"全年|年度报告|年报|年度业绩|年末|年底")
# 全年口径的指标/时点：问这些时用户在问「整个会计年度」，应优先年报而不是半年报。
# 只描述指标本身，不含公司、年份、数值，也不针对任何具体题目。
FULL_PERIOD_METRIC_RE = re.compile(
    r"营业收入|营业总收入|营收|销售收入|营业额|"
    r"归母净利润|归属于上市公司股东的净利润|净利润|利润总额|扣非|"
    r"毛利率|毛利|净利率|"
    r"产销量|生产量|销售量|产量|销量|"
    r"合同负债|预收款项|预收账款|"
    r"净资产收益率|每股收益|经营现金流|资产总额|同比增长|同比"
)
# 报告期软偏好：没明说报告类型时，非目标期报告的降权系数
PERIOD_SOFT_MULT = 0.35


def detect_companies(query, companies=None):
    """识别问题里提到的公司。返回全称列表（按命中别名长度降序，长名优先）。

    一个公司只算一次。若一家都没点名，按三级线索判断是不是在问「全部公司」：
      1. 明确的全体措辞（12家 / 所有公司 / 各家 / 哪些XX公司 …）
      2. 行业集合名词（白酒公司 / 酒企 / 白酒行业 …）——没点名具体某家，通常指这一整组
      3. 跨公司枚举/比较语义（哪些 / 谁的 / 各家 / 最高 / 排名 …）
    命中任一 → 视为全部公司，走逐家检索而不是全局混合 TopK
    （全局 TopK 在跨公司题上必然被一两家公司的块占满，覆盖不全）。
    """
    names = companies or list(COMPANY_ALIASES)
    found = []
    for name in names:
        for alias in COMPANY_ALIASES.get(name, [name]):
            if alias in query:
                found.append((name, len(alias)))
                break
    out = []
    for name, _ in sorted(found, key=lambda x: -x[1]):
        if name not in out:
            out.append(name)
    if out:
        return out
    if ALL_COMPANY_RE.search(query):
        return list(names)
    if INDUSTRY_SET_RE.search(query):
        return list(names)
    if CROSS_COMPANY_RE.search(query):
        return list(names)
    return []


def detect_year(query):
    """识别问题里的年份（取最后一个，通常是对比的基准年）。"""
    ys = YEAR_RE.findall(query)
    return int(ys[-1]) if ys else None


def detect_period(query):
    """识别报告期口径，返回 (rtype, strict)。

    strict=True   用户明确点名了报告类型（「半年报」「年报」「上半年」）→ 只取该类型
    strict=False  用户只说了年份、但问的是全年口径指标 → 优先年报；
                  该期材料不足时自动放宽为软加权（见 Retriever.search 的 rtype_pref）

    这样「2025年营业收入」不会再召到 2025 半年报，而「2025年上半年营业收入」仍然只取半年报。
    """
    if HALF_YEAR_RE.search(query):
        return "半年报", True
    if FULL_YEAR_RE.search(query):
        return "年报", True
    # 只说了「2025年」：默认年报口径（年报是完整会计年度，半年报只是其中一段）
    if YEAR_RE.search(query):
        return "年报", False
    # 连年份都没说，但问的是全年口径指标：同样优先年报
    if FULL_PERIOD_METRIC_RE.search(query):
        return "年报", False
    return None, False


def detect_rtype(query):
    """兼容旧接口：只返回识别到的报告类型（不含是否严格）。"""
    return detect_period(query)[0]


class Retriever:
    def __init__(self, with_model=True):
        self.emb = np.load(os.path.join(IDX, "向量.npy"))
        with open(os.path.join(IDX, "元数据.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(IDX, "文本.json"), encoding="utf-8") as f:
            self.texts = json.load(f)
        self.bm25 = BM25.load(os.path.join(IDX, "词频索引.pkl"))
        self._metric_cache = {}  # 指标定向权重向量的缓存（按 指标+是否同比 复用）
        self._generic_cache = {}  # 陌生科目兜底的权重缓存
        self.model = None
        if with_model:
            # 用自写前向（embed_util），sentence-transformers 在本机型上 encode 会卡死
            sys.path.insert(0, os.path.join(ROOT, "代码"))
            from embed_util import Embedder  # noqa: PLC0415

            self.model = Embedder(model_name=EMBED_MODEL, device=os.environ.get("EMBED_DEVICE") or None)

    # ---------- 单路打分 ----------
    def vector_scores(self, q):
        qv = self.model.encode([q])[0]
        return self.emb @ qv

    def bm25_scores(self, q):
        return self.bm25.scores(get_tokenize()(q))

    # ---------- 检索 ----------
    def search(
        self,
        query,
        mode="hybrid",
        top_k=6,
        company=None,
        year=None,
        rtype=None,
        section=None,
        rerank=True,
        pool_size=60,
        rtype_pref=None,
    ):
        """mode: vector | bm25 | hybrid。rerank=True 时启用两阶段（召回 pool_size → 重排）。

        company/year/rtype 是元数据硬过滤（先筛后召回之外再筛一次，保证结果干净）。
        rtype_pref 是「报告期偏好」：优先取该报告期，硬过滤后材料不足时自动退回软加权，
        用来实现「问 2025 年就默认年报，但别把没有年报的情形搞成零召回」。
        """
        vs = bs = bs2 = None
        if mode in ("vector", "hybrid"):
            vs = self.vector_scores(query)
        if mode in ("bm25", "hybrid"):
            bs = self.bm25_scores(query)
        # 定向检索：用「标准科目名+口径词」再走一路 BM25。
        # 解决问法里没有科目名（如「经销商已打款未确认收入」→ 合同负债）时的召回失败。
        aug = augment_query(query)
        if aug and mode in ("bm25", "hybrid"):
            bs2 = self.bm25_scores(aug)
        else:
            # 没命中 metrics.py 的规则（陌生科目）：用「标准科目名 + 通用报表表头词」再走一路
            gaug = augment_generic(query)
            if gaug and mode in ("bm25", "hybrid"):
                bs2 = self.bm25_scores(gaug)

        # 融合分：RRF 的分数形式（非负），便于再乘指标定向权重
        score = self._fused_scores(mode, vs, bs, bs2)

        # 指标/口径定向：问「营业收入」就偏向合并口径与主要会计数据，
        # 抑制子公司表、贸易业务表、分渠道小计这类局部口径（不改动公司/年份过滤）
        mlabel, mvec, mmask = metric_vector(self.texts, query, self._metric_cache)
        if mvec is not None:
            score = score * mvec
        else:
            # 陌生科目兜底：规则外的科目不再各自写规则，改用通用的「报表结构」信号加权，
            # 并共用同一套定向配额机制（详见 代码/科目兜底.py）。
            glabel, gvec, gmask = generic_vector(self.texts, query, self._generic_cache)
            if gvec is not None:
                score = score * gvec
                mmask = gmask
                mlabel = glabel
        self.last_metric = mlabel
        order = np.argsort(-score)
        self._last_mask = mmask

        # 报告期口径：用户明说报告类型时 rtype 已是硬过滤；没明说时走 rtype_pref
        # ——先试着只留目标报告期（保证同一口径，跨公司题不会一家年报一家半年报），
        # 若该期材料不够 top_k（例如问的年份根本没有年报），退回软加权而不是零召回。
        eff_rtype = rtype
        self._last_period = ""
        if rtype_pref:
            strict = self._mask(order, company, year, rtype_pref, section)
            if len(strict) >= top_k:
                order = strict
                eff_rtype = rtype_pref
                self._last_period = rtype_pref + "·硬"
            else:
                score = score * self._period_vec(rtype_pref)
                order = np.argsort(-score)
                self._last_period = rtype_pref + "·软"

        # 硬过滤前置：先把不符合元数据的块剔掉，再做重排。
        # 否则重排池（pool_size）会被无关公司/年份的块占满，真正相关的反而进不来。
        if company or year or eff_rtype or section:
            order = self._mask(order, company, year, eff_rtype, section)

        # 定向配额：命中目标口径的块至少占一半席位。
        # 加权只能改变相对次序，救不回排在几千名开外的块（RRF 名次分几乎为 0），
        # 所以再给目标口径单独的席位保障。
        if mmask is not None:
            order = self._quota_merge(order, mmask, quota(top_k))

        if rerank:
            order = self._rerank(order[:pool_size], query, need_figure=asks_number(query))

        out = []
        for i in order:
            m = self.meta[i]
            if company and m["company"] != company:
                continue
            if year and m["year"] != year:
                continue
            if eff_rtype and m["rtype"] != eff_rtype:
                continue
            if section and section not in m["section"]:
                continue
            out.append(
                {
                    "rank": len(out) + 1,
                    "id": int(m["id"]),
                    "company": m["company"],
                    "code": m["code"],
                    "year": m["year"],
                    "rtype": m["rtype"],
                    "section": m["section"],
                    "page": m["page"],
                    "block_type": m["block_type"],
                    "vector": float(vs[i]) if vs is not None else None,
                    "bm25": float(bs[i]) if bs is not None else None,
                    "text": self.texts[i],
                }
            )
            if len(out) >= top_k:
                break
        return out

    def _match(self, i, company, year, rtype, section):
        m = self.meta[i]
        if company and m["company"] != company:
            return False
        if year and m["year"] != year:
            return False
        if rtype and m["rtype"] != rtype:
            return False
        if section and section not in m["section"]:
            return False
        return True

    def _period_vec(self, pref):
        """报告期软偏好权重：目标报告期 ×1.0，另一期 ×PERIOD_SOFT_MULT。"""
        if not hasattr(self, "_pv_cache"):
            self._pv_cache = {}
        v = self._pv_cache.get(pref)
        if v is None:
            v = np.array(
                [1.0 if m["rtype"] == pref else PERIOD_SOFT_MULT for m in self.meta],
                dtype=np.float32,
            )
            self._pv_cache[pref] = v
        return v

    @staticmethod
    def _quota_merge(order, mask, n_quota):
        """把命中目标口径的块提到前面，占 n_quota 个席位，其余保持原相对次序。"""
        pref, rest = [], []
        for i in order:
            if len(pref) < n_quota and mask[i]:
                pref.append(int(i))
            else:
                rest.append(int(i))
        return np.asarray(pref + rest, dtype=np.int64)

    def _mask(self, order, company, year, rtype, section):
        """按元数据过滤候选顺序，保持原相对次序。"""
        keep = [int(i) for i in order if self._match(int(i), company, year, rtype, section)]
        return np.asarray(keep, dtype=np.int64)

    # ---------- 跨公司路由 ----------
    def companies(self):
        if not hasattr(self, "_companies"):
            self._companies = sorted({m["company"] for m in self.meta})
        return self._companies

    def route(
        self,
        query,
        top_k=6,
        per_company_k=3,
        max_total=24,
        mode="hybrid",
        rerank=True,
        pool_size=60,
        company=None,
        year=None,
        rtype=None,
    ):
        """按「问题涉及几家公司」决定检索方式，返回 (hits, info)。

        三档：
          multi-company  ≥2 家被点名：逐家各自检索 per_company_k 块，再按
                         「每家第1块 → 每家第2块 …」轮转合并，保证每家都有材料，
                         不会像全局 TopK 那样被一两家公司的块占满。
          single-company 只点名 1 家（或用户手动指定 company）：带公司约束检索。
          global          没点名任何公司：退回全局混合检索。
        """
        comps = [company] if company else detect_companies(query, self.companies())
        y = year if year is not None else detect_year(query)
        if rtype is not None:
            rt, rt_strict = rtype, True
        else:
            rt, rt_strict = detect_period(query)
        # 报告期口径：明确点名 → 硬过滤；只说了年份 → 偏好（优先硬过滤，不足再软加权）。
        # 跨公司题对每家公司用同一套口径，不会出现一家年报、一家半年报。
        rt_hard = rt if rt_strict else None
        rt_soft = None if rt_strict else rt
        info = {"route": "", "companies": comps, "year": y, "rtype": rt, "rtype_strict": rt_strict,
                "per_company_k": per_company_k, "metric": detect_metric(query)[0] or ""}

        if len(comps) >= 2:
            k = per_company_k
            if k * len(comps) > max_total:
                k = max(2, max_total // len(comps))
            by_c = {
                c: self.search(query, mode=mode, top_k=k, company=c, year=y,
                               rtype=rt_hard, rtype_pref=rt_soft,
                               rerank=rerank, pool_size=pool_size)
                for c in comps
            }
            merged = []
            for r in range(k):
                for c in comps:
                    if r < len(by_c[c]):
                        merged.append(by_c[c][r])
            for i, h in enumerate(merged, 1):
                h["rank"] = i
            info["route"] = "multi-company"
            info["per_company_k"] = k
            info["per_company_hits"] = {c: len(v) for c, v in by_c.items()}
            return merged, info

        single = comps[0] if comps else None
        hits = self.search(query, mode=mode, top_k=top_k, company=single, year=y,
                           rtype=rt_hard, rtype_pref=rt_soft,
                           rerank=rerank, pool_size=pool_size)
        info["route"] = "single-company" if single else "global"
        return hits, info

    def _rerank(self, idxs, query, need_figure=True, same_source_cap=2):
        """第二段：在候选池里做轻量的规则重排。

        三条规则，都是针对财报语料的具体观察：
          1. 同源收敛：同一 (公司, 页码) 最多留 2 块 —— 一张跨页大表会产生多个几乎相同的块，
             不收敛的话 top_k 会被同一页占满
          2. 数字密度：问「多少/占比/增速」时，块里必须真的有财务数字，否则降权
             （第八节里大量「合并所有者权益变动表 2026年1—6月 单位：元」这类叙述块
             语义相似度很高但一个数字都没有，是最主要的噪声源）
          3. 表格块小幅加权：结构化表格的信息密度高于同长度的叙述文字
        """
        seen_source = {}
        scored = []
        for pos, i in enumerate(idxs):
            m = self.meta[i]
            s = 1.0 / (1 + pos)  # 保留原始召回名次的分数
            key = (m["doc_id"], m["page"])
            n = seen_source.get(key, 0)
            seen_source[key] = n + 1
            if n >= same_source_cap:
                s *= 0.25
            text = self.texts[i]
            if need_figure and not has_figure(text):
                s *= 0.2
            if m["block_type"] == "table":
                s *= 1.15
            scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored]

    @staticmethod
    def _rrf_scores(pairs, k=60):
        """Reciprocal Rank Fusion 的分数形式：融合多路排名，避免各路分数不可比。

        pairs = [(scores, weight), ...]，返回长度 = 块数的非负分数数组。
        用分数而不是名次数组，是为了能再乘上指标定向权重。
        """
        n = len(pairs[0][0])
        score = np.zeros(n, dtype=np.float32)
        for s, w in pairs:
            order = np.argsort(-s)
            for rank, idx in enumerate(order):
                score[idx] += w / (k + rank + 1)
        return score

    def _fused_scores(self, mode, vs, bs, bs2=None):
        if mode == "vector":
            return self._rrf_scores([(vs, 1.0)])
        if mode == "bm25":
            pairs = [(bs, 1.0)]
            if bs2 is not None:
                pairs.append((bs2, 1.0))
            return self._rrf_scores(pairs)
        pairs = [(vs, 1.0), (bs, 1.4)]
        if bs2 is not None:
            pairs.append((bs2, 1.0))
        return self._rrf_scores(pairs)

    def stats(self):
        companies = sorted({m["company"] for m in self.meta})
        return {
            "n_chunks": len(self.meta),
            "n_docs": len({m["doc_id"] for m in self.meta}),
            "n_companies": len(companies),
            "companies": companies,
            "years": sorted({m["year"] for m in self.meta}),
            "rtypes": sorted({m["rtype"] for m in self.meta}),
            "dim": int(self.emb.shape[1]),
        }
