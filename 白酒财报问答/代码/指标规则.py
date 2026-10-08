# -*- coding: utf-8 -*-
"""指标/口径定向。

解决的问题（第 9 步基线失败模式 1）：公司过滤是对的，但「问的到底是哪个指标、哪个口径」
没有参与排序，于是召回了口径不对的表：

    问「营业收入」  → 召到分渠道小计行、子公司表、贸易业务表、前五名客户表
    问「同比增速」  → 拿局部业务（贸易/线上/子公司）的同比当总指标同比
    问「合同负债」  → 召到收入确认会计政策段（只有定义、没有金额）
    问「毛利率」    → 召到非主营业务口径的表

做法：识别问句里的指标词 → 生成一条长度 = 块数的权重向量 → 乘到融合分上再排序。
规则全部是**通用表头/章节偏好**（「主要会计数据」「主营业务分行业」「比上年同期增减」这类），
不含任何公司名、年份、具体数值，也不针对某一道测试题。

对外只有两个入口：detect_metric(query) / metric_vector(texts, query, cache)。
"""

import re

import numpy as np

# 命中「目标口径表头/章节」的加权（extra = 最想要的口径，strong = 次优口径）；
# 命中噪声词的降权（noise = 明确不是目标口径，mild = 口径偏窄）；
# 块里压根没有该指标词的降权
EXTRA_MULT = 3.6
STRONG_MULT = 2.4
NOISE_MULT = 0.42
MILD_MULT = 0.50
MISSING_MULT = 0.45
# 同比修饰符：有同比列 → 加权；没有同比列 → 降权（问增速时不需要只有绝对值的块）
YOY_STRONG = 2.0
YOY_WEAK = 0.70

RULES = {
    "营业收入": {
        "triggers": ("营业收入", "营收", "销售收入", "营业额", "收入规模"),
        # 最想要的口径：合并口径的主要财务数据。
        # 第二条是「总指标行」的形状：营业收入 → 金额 → 变动值（%可带可不带），
        # 用来兜住那些被切掉表头的表格块（如「主要会计数据」表被切成两段时）
        "extra": (
            r"主要会计数据[\s\S]{0,800}营业收入",
            r"主要财务指标[\s\S]{0,800}营业收入",
            r"营业收入[^|\n]{0,12}[\s\S]{0,80}\d{1,3}(?:,\d{3}){2,}[\s\S]{0,300}-?\d+\.\d{1,2}",
        ),
        # 次优：带同比列的营业收入表
        "strong": (
            r"营业收入比上年同期增减",
            r"营业收入[^|\n]{0,40}同比",
            r"营业总收入[\s\S]{0,60}同比",
        ),
        # 局部口径：不是公司整体营业收入
        "noise": (
            "子公司", "贸易业务", "前五名客户", "线上销售平台", "主要客户", "客户名称",
            "占年度销售", "其他业务收入", "分部", "小计",
            # 主要子公司/参股公司情况表：表头也带「营业收入」，但那是子公司数
            "公司名称", "公司类型", "主要业务", "注册资本", "主要子公司", "参股公司",
        ),
        # 分行业/分产品/分渠道表：有营业收入但是分部数，不是总指标
        "mild": ("营业成本", "毛利率"),
    },
    "合同负债": {
        "triggers": (
            "合同负债", "预收款项", "预收账款", "预收",
            "打款", "先款后货", "回款",
            "未确认收入", "还没确认收入", "尚未确认收入", "未确认的收入", "已收款未确认",
        ),
        # 最想要的口径：科目名旁边就有金额
        "extra": (
            r"合同负债[\s\S]{0,160}\d{1,3}(?:,\d{3}){2,}",
            r"预收款项[\s\S]{0,160}\d{1,3}(?:,\d{3}){2,}",
        ),
        "strong": (
            r"合同负债[^|\n]{0,20}\|",
            r"合同负债[\s\S]{0,160}\d+\.\d{2}",
        ),
        # 只有定义、没有金额的会计政策段
        "noise": (
            "是指", "定义", "对价", "转让商品", "履约义务", "会计政策", "收入确认",
            "尚未履行", "确认条件", "控制权",
        ),
        "mild": (),
    },
    "毛利率": {
        "triggers": ("毛利率", "毛利"),
        "extra": (
            r"主营业务分行业",
            r"主营业务分产品",
            r"毛利率（%）",
        ),
        "strong": (
            r"分销售模式",
            r"分渠道",
            r"毛利率[\s\S]{0,80}\d+\.\d{1,2}%",
        ),
        "noise": ("会计政策", "变动原因", "关键审计事项"),
        "mild": (),
    },
    "净利润": {
        "triggers": (
            "归母净利润", "归母净利", "归属于上市公司股东的净利润",
            "归属于母公司股东的净利润", "归属于母公司所有者的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
            "扣非净利润", "扣非后净利润",
            "净利润", "净亏损",
        ),
        # 最想要的口径：合并口径的主要财务数据里的那一行。
        # 注意「归属于上市公司股东的净利润」这个连续串不会命中
        # 「归属于上市公司股东的**扣除非经常性损益的**净利润」那一行，天然区分了两个口径。
        "extra": (
            r"主要会计数据[\s\S]{0,900}归属于上市公司股东的净利润",
            r"主要财务指标[\s\S]{0,900}归属于上市公司股东的净利润",
            r"主要会计数据[\s\S]{0,900}净利润",
            r"主要财务指标[\s\S]{0,900}净利润",
            # 兜住被切掉表头的表格块：科目名 → 千分位金额 → 变动值
            r"归属于上市公司股东的净利润[^|\n]{0,12}[\s\S]{0,80}\d{1,3}(?:,\d{3}){2,}[\s\S]{0,300}-?\d+\.\d{1,2}",
        ),
        # 次优：合并利润表、带同比列的净利润
        "strong": (
            r"归属于上市公司股东的净利润[^|\n]{0,40}同比",
            r"净利润[\s\S]{0,80}比上年同期增减",
            r"合并利润表[\s\S]{0,900}净利润",
            r"归属于母公司所有者的净利润",
            r"归属于母公司股东的净利润[\s\S]{0,200}\d{1,3}(?:,\d{3}){2,}",
        ),
        # 局部口径 / 不是公司合并口径的利润
        "noise": (
            "少数股东损益", "归属于少数股东", "子公司", "主要子公司", "参股公司",
            "公司名称", "公司类型", "注册资本", "权益法", "投资收益", "对联营",
            "分部", "小计", "营业外收入", "半年度报告",
        ),
        # 母公司口径（不是合并）、以及利润分配类，都不该当成归母净利润
        "mild": ("母公司利润表", "母公司财务报表", "未分配利润", "提取法定盈余", "利润分配"),
    },
    "产销量": {
        "triggers": ("产量", "销量", "产销量", "销售数量", "生产量", "销售量"),
        "extra": (
            r"产销量情况",
            r"生产量[\s\S]{0,80}销售量",
        ),
        "strong": (
            r"主要产品[\s\S]{0,80}单位",
            r"销售量[\s\S]{0,60}库存量",
        ),
        "noise": ("子公司", "前五名客户", "贸易业务"),
        "mild": (),
    },
}

# 同比修饰符：单独出现不构成指标，只在已命中某个指标规则时叠加
YOY = {
    "triggers": ("同比", "增速", "增长率", "增长速度", "增减", "增长", "下滑", "下降", "上升"),
    "strong": (
        r"比上年同期增减", r"本期比上年", r"本年比上年增减", r"比上年增减",
        r"同比增减", r"同比增长", r"同比减少", r"同比下滑", r"同比上升",
        r"增減", r"增减（%）", r"增减\(%\)", r"增减变动", r"增减幅度",
    ),
    "noise": ("小计", "子公司", "贸易业务", "线上销售平台", "前五名客户"),
}


def _compile(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns))


def _escape_join(words):
    """空规则返回 None（不能拼成空正则，否则会匹配所有块）。"""
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words))


_COMPILED = {}


def _get(rule_name, kind):
    """取编译好的正则。extra / strong / mild 是正则；triggers / noise 是普通词。"""
    key = (rule_name, kind)
    if key not in _COMPILED:
        src = YOY[kind] if rule_name == "__yoy__" else RULES[rule_name][kind]
        if not src:
            _COMPILED[key] = None          # 空规则视为不生效
        elif kind in ("extra", "strong", "mild"):
            _COMPILED[key] = _compile(src)
        else:
            _COMPILED[key] = _escape_join(src)
    return _COMPILED[key]


# 每个指标的「口径词」：问法千变万化，但财报里承载该指标的表头/章节是固定的。
# 检索时把口径词并进原始问句，单独走一路 BM25，用来解决「问法里根本没有科目名」
# （如「经销商已经打款但还没确认收入」→「合同负债」）导致的召回失败。
CANON = {
    "营业收入": ("营业收入", "主要会计数据"),
    "合同负债": ("合同负债", "预收款项"),
    "毛利率": ("毛利率", "主营业务分产品"),
    "净利润": ("归属于上市公司股东的净利润", "净利润", "主要会计数据"),
    "产销量": ("产销量", "生产量", "销售量"),
}
YOY_CANON = ("比上年同期增减", "同比")


def augment_query(query, rule_name=None, yoy=False):
    """把指标对应的标准科目名/口径词并进问句，用于额外一路定向检索。"""
    if rule_name is None:
        rule_name, yoy = detect_metric(query)
    if not rule_name:
        return None
    terms = list(CANON.get(rule_name, ()))
    if yoy:
        terms += list(YOY_CANON)
    return query + " " + " ".join(terms)


def detect_metric(query):
    """识别问句在问哪个指标。返回 (指标名或None, 是否问同比)。"""
    hit = None
    for name, rule in RULES.items():
        for t in rule["triggers"]:
            if t in query:
                hit = name
                break
        if hit:
            break
    yoy = any(t in query for t in YOY["triggers"])
    return hit, yoy


def build_vector(texts, rule_name, yoy):
    """按规则算出每个块的权重向量（1.0 = 不干预）。"""
    n = len(texts)
    v = np.ones(n, dtype=np.float32)
    if not rule_name:
        return v

    term = _get(rule_name, "triggers")
    extra = _get(rule_name, "extra")
    strong = _get(rule_name, "strong")
    noise = _get(rule_name, "noise")
    mild = _get(rule_name, "mild")
    yoy_strong = _get("__yoy__", "strong") if yoy else None
    yoy_noise = _get("__yoy__", "noise") if yoy else None

    for i, t in enumerate(texts):
        m = 1.0
        if not term.search(t):
            m *= MISSING_MULT          # 块里连这个指标的词都没有
        if extra is not None and extra.search(t):
            boost = EXTRA_MULT         # 最想要的口径（合并/主营/带金额）
        elif strong is not None and strong.search(t):
            boost = STRONG_MULT        # 次优口径
        else:
            boost = 1.0
        if mild and mild.search(t):
            # 分部表（分行业/分产品/分渠道，带营业成本、毛利率列）：最多降到次优档再打折
            boost = min(boost, STRONG_MULT) * MILD_MULT
        m *= boost
        if noise is not None:
            c = len(noise.findall(t))
            if c:
                m *= NOISE_MULT ** min(c, 2)   # 局部口径 / 只有定义的政策段
        if yoy:
            if yoy_strong.search(t):
                m *= YOY_STRONG
            else:
                m *= YOY_WEAK
            c = len(yoy_noise.findall(t))
            if c:
                m *= NOISE_MULT ** min(c, 2)
        v[i] = m
    return v


def build_mask(texts, rule_name):
    """命中「最想要的口径」的布尔掩码。

    光靠加权救不回来：目标块可能因为问法里没有科目名而排在几千名开外，
    RRF 是名次打分，几千名几乎等于 0 分。所以另外用掩码做「定向配额」——
    保证目标口径的块至少占一半席位。
    """
    extra = _get(rule_name, "extra")
    return np.fromiter((bool(extra.search(t)) for t in texts), dtype=bool, count=len(texts))


def metric_vector(texts, query, cache=None):
    """对外入口。返回 (描述串, 权重向量)。cache 用来避免重复扫描全量文本。"""
    name, yoy = detect_metric(query)
    if not name:
        return "", None, None
    label = name + ("+同比" if yoy else "")
    key = (name, yoy, len(texts))
    if cache is not None:
        if key not in cache:
            cache[key] = (label, build_vector(texts, name, yoy), build_mask(texts, name))
        return cache[key]
    return label, build_vector(texts, name, yoy), build_mask(texts, name)


def quota(top_k):
    """目标口径至少占多少席位（一半，至少 1 个）。"""
    return max(1, (top_k + 1) // 2)


if __name__ == "__main__":
    for q in [
        "贵州茅台2025年营业收入是多少？同比增速多少？",
        "洋河股份2025年末经销商已经打款但公司还没确认收入的金额是多少？",
        "泸州老窖2025年的毛利率是多少？",
        "这12家公司2025年谁的营业收入增速最高？",
        "2025年有哪些白酒公司归母净利润同比下滑？",
    ]:
        print(detect_metric(q), "<-", q)
