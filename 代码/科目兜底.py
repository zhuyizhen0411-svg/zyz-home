# -*- coding: utf-8 -*-
"""陌生财务科目通用兜底（generic fallback）。

背景（v2 测试暴露）：metrics.py 里写死的 5 组规则（营业收入/合同负债/毛利率/净利润/产销量）
只覆盖少数科目，换成存货、货币资金、销售费用这类"规则外科目"就召回失败（答"材料不足"）。

本模块的设计原则：**不认识具体科目，只认识"财报里什么样的块装着科目数据"**。

  ① 不问"存货在哪"，而是问"这块是不是一张装着『科目名 + 金额』的主表"
  ② 不问"销售费用属于哪类"，而是问"这块是不是只有表头没有数字"（是就降权）

所以这里**没有**任何针对某一个科目的权重、口径或噪声配置：
  - 科目词典只是一张「标准科目名 ↔ 常见写法」的对照表（会计准则通用科目），
    仅用于把问句里的说法归一成标准名给 BM25 加一路，条目之间完全平等；
  - 权重信号全部是**报表结构层面**的通用形状：是不是表格、有没有千分位数字、
    有没有「科目行」、是不是主表表头、是不是附注/受限/子公司/会计政策这类非目标块。

只在 metrics.py 的 5 组规则全部没命中时才启用（规则内的题行为不变）。
不改变公司路由、报告期识别、生成侧与引用追溯；跨公司题逐家走同一套逻辑。

对外入口：detect_subject(query) / augment_generic(query) / generic_vector(texts, query, cache)
"""

import re

import numpy as np

# ---------------------------------------------------------------- 权重（通用，与科目无关）
HAS_TABLE_MULT = 1.0        # 是表格块（文本以 [表格] 开头）
NO_TABLE_MULT = 0.55        # 叙述段：即使提到科目也通常是原因/政策说明
NO_FIGURE_MULT = 0.35       # 没有任何千分位金额：多半是「40、销售费用 单位：元」这类空表头
SUBJECT_ROW_MULT = 2.4      # 科目行：科目名 + 分隔符 + 千分位金额
STATEMENT_MULT = 2.2        # 主表表头：主要会计数据 / 资产构成表 / 经营情况表 / 三大报表
SUBJECT_HIT_MULT = 1.9      # 块里出现了问到的科目名
SUBJECT_MISS_MULT = 0.50    # 问到了科目名，但这块里没有
KEYWORD_MULT = 1.5          # 问句里没识别出科目名时，命中问句实义词
NOISE_MULT = 0.50           # 非目标块：受限/子公司/关联方/会计政策定义类

# 科目行形状：2~10 个汉字（或含括号的科目名）→ 分隔符 → 千分位金额。
# 这是"资产负债表/利润表/资产构成表"共同的行结构，与具体科目无关。
SUBJECT_ROW_RE = re.compile(
    r"[\u4e00-\u9fff（）()]{2,12}\s*[|｜\s]\s*-?\d{1,3}(?:,\d{3}){2,}"
)
# 千分位金额（百万级以上）
FIG_RE = re.compile(r"\d{1,3}(?:,\d{3}){2,}")
# 表格块标记（chunk 文本自带）
TABLE_MARK = "[表格]"

# 主表表头：财报里承载"科目 → 金额"的那些表的通用表头/章节名。
# 全部是报表结构词，不含任何科目名、公司名、年份或数值。
STATEMENT_PATTERNS = (
    r"主要会计数据",
    r"主要财务指标",
    r"合并资产负债表",
    r"合并利润表",
    r"合并现金流量表",
    r"资产及负债状况",
    r"资产构成重大变动",
    # 资产构成表：项目名称 / 本期期末数 / 占总资产的比例
    r"项目名称[\s\S]{0,40}本期期末数",
    r"本期期末数[\s\S]{0,60}占总资产",
    # 经营情况表：科目 / 本报告期 / 上年同期 / 同比增减
    r"本报告期[\s\S]{0,40}上年同期",
    r"上年同期[\s\S]{0,40}同比增减",
    # 三大报表的日期列表头（项目 | 附注 | 2024年12月31日）
    r"项目[\s\S]{0,20}附注[\s\S]{0,20}20\d{2}年\d{1,2}月\d{1,2}日",
)
# 非目标块：受限资产、子公司/参股、关联方、会计政策与定义、被投资单位明细等。
# 同样只描述"这类块是什么"，不针对任何科目。
NOISE_PATTERNS = (
    "受限", "抵押", "质押", "保证金", "冻结",
    "子公司", "参股公司", "公司名称", "公司类型", "注册资本",
    "关联方", "被投资单位", "未办妥", "产权证书",
    "会计政策", "是指", "定义", "履约义务", "确认条件", "计量方法",
)

# ---------------------------------------------------------------- 通用科目词典
# 用途只有一个：把问句里的说法归一成"标准科目名"，给 BM25 多一路关键词。
# 这是《企业会计准则》常用科目的平铺清单，条目之间没有任何权重差异，
# 也不含口径偏好 —— 与 metrics.py 里那 5 组「带 extra/strong/noise 的规则」是两回事。
SUBJECT_LEXICON = [
    # 资产
    ("货币资金", ("货币资金", "现金及现金等价物", "账上的资金")),
    ("交易性金融资产", ("交易性金融资产",)),
    ("应收票据", ("应收票据",)),
    ("应收账款", ("应收账款", "应收货款")),
    ("应收款项融资", ("应收款项融资",)),
    ("预付款项", ("预付款项", "预付账款")),
    ("其他应收款", ("其他应收款",)),
    ("存货", ("存货", "库存商品", "原材料")),
    ("其他流动资产", ("其他流动资产",)),
    ("流动资产合计", ("流动资产合计",)),
    ("长期股权投资", ("长期股权投资",)),
    ("投资性房地产", ("投资性房地产",)),
    ("固定资产", ("固定资产",)),
    ("在建工程", ("在建工程",)),
    ("使用权资产", ("使用权资产",)),
    ("无形资产", ("无形资产",)),
    ("商誉", ("商誉",)),
    ("长期待摊费用", ("长期待摊费用",)),
    ("递延所得税资产", ("递延所得税资产",)),
    ("资产总计", ("资产总计", "总资产")),
    # 负债
    ("短期借款", ("短期借款",)),
    ("应付票据", ("应付票据",)),
    ("应付账款", ("应付账款",)),
    ("合同负债", ("合同负债",)),
    ("应付职工薪酬", ("应付职工薪酬",)),
    ("应交税费", ("应交税费",)),
    ("其他应付款", ("其他应付款",)),
    ("长期借款", ("长期借款",)),
    ("应付债券", ("应付债券",)),
    ("负债合计", ("负债合计", "总负债")),
    # 权益
    ("股本", ("股本", "总股本")),
    ("资本公积", ("资本公积",)),
    ("盈余公积", ("盈余公积",)),
    ("未分配利润", ("未分配利润",)),
    ("归属于母公司股东权益", ("归属于母公司股东权益", "归母净资产", "归属于上市公司股东的净资产")),
    ("少数股东权益", ("少数股东权益",)),
    # 损益
    ("营业总收入", ("营业总收入",)),
    ("营业收入", ("营业收入",)),
    ("营业成本", ("营业成本",)),
    ("税金及附加", ("税金及附加",)),
    ("销售费用", ("销售费用",)),
    ("管理费用", ("管理费用",)),
    ("研发费用", ("研发费用", "研发投入")),
    ("财务费用", ("财务费用",)),
    ("投资收益", ("投资收益",)),
    ("公允价值变动收益", ("公允价值变动收益",)),
    ("信用减值损失", ("信用减值损失",)),
    ("资产减值损失", ("资产减值损失",)),
    ("资产处置收益", ("资产处置收益",)),
    ("其他收益", ("其他收益",)),
    ("营业利润", ("营业利润",)),
    ("营业外收入", ("营业外收入",)),
    ("营业外支出", ("营业外支出",)),
    ("利润总额", ("利润总额",)),
    ("所得税费用", ("所得税费用",)),
    ("净利润", ("净利润",)),
    ("归属于母公司股东的净利润", ("归母净利润", "归属于母公司股东的净利润")),
    ("综合收益总额", ("综合收益总额",)),
    # 现金流
    ("经营活动产生的现金流量净额", ("经营活动产生的现金流量净额", "经营现金流", "经营性现金流")),
    ("投资活动产生的现金流量净额", ("投资活动产生的现金流量净额", "投资现金流")),
    ("筹资活动产生的现金流量净额", ("筹资活动产生的现金流量净额", "筹资现金流")),
    ("销售商品提供劳务收到的现金", ("销售商品提供劳务收到的现金",)),
    ("购买商品接受劳务支付的现金", ("购买商品接受劳务支付的现金",)),
    ("购建固定资产支付的现金", ("购建固定资产支付的现金",)),
    ("分配股利利润或偿付利息支付的现金", ("分配股利利润或偿付利息支付的现金",)),
    ("现金及现金等价物净增加额", ("现金及现金等价物净增加额",)),
    ("期初现金及现金等价物余额", ("期初现金及现金等价物余额",)),
    ("期末现金及现金等价物余额", ("期末现金及现金等价物余额",)),
    # 每股收益 / 比率 / 分红（不是科目，但同样是财报里的固定指标行）
    ("基本每股收益", ("基本每股收益", "每股收益")),
    ("稀释每股收益", ("稀释每股收益",)),
    ("加权平均净资产收益率", ("加权平均净资产收益率", "净资产收益率", "ROE")),
    ("资产负债率", ("资产负债率",)),
    ("每股净资产", ("每股净资产",)),
    ("每股经营活动现金流量", ("每股经营活动现金流量",)),
    ("流动比率", ("流动比率",)),
    ("速动比率", ("速动比率",)),
    ("存货周转率", ("存货周转率",)),
    ("应收账款周转率", ("应收账款周转率",)),
    ("现金分红", ("现金分红", "派发现金红利", "每10股派", "分红方案", "利润分配")),
]

# 从问句里抽"科目候选词"前先剥掉的部分：公司名、年份、报告期、疑问词、常见虚词。
_STOPWORDS = (
    "多少", "多少家", "分别是", "分别", "各自", "哪家", "哪些", "谁", "谁的", "怎么", "怎样",
    "是否", "有没有", "还是", "以及", "和", "与", "相比", "比较", "排名", "排序", "前三",
    "请问", "帮我", "查一下", "是多少", "多少元", "多少万元", "多少亿元", "金额", "数值",
    "这家", "这家公司", "公司", "上市公司", "白酒公司", "白酒", "企业", "各家", "每家",
    "上半年", "半年度", "半年报", "年度报告", "年报", "年度", "年末", "年底", "全年",
    "本期", "上年", "同期", "同比", "增长", "下降", "下滑", "增加", "减少", "变动",
    "为什么", "原因", "情况", "账面价值", "余额", "合计", "总计", "占比", "比重",
)
_COMPANY_HINT = ("贵州茅台", "五粮液", "泸州老窖", "山西汾酒", "洋河股份", "古井贡酒", "今世缘",
                 "迎驾贡酒", "口子窖", "水井坊", "舍得酒业", "酒鬼酒", "茅台", "汾酒", "老窖",
                 "洋河", "古井", "迎驾", "舍得", "酒鬼")
_YEAR_RE = re.compile(r"20\d{2}\s*年?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,6}")


def _compile(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns))


_STATEMENT_RE = _compile(STATEMENT_PATTERNS)


def _escape_join(words):
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words))


def _tokenize_cjk(text):
    """中文切词：优先复用检索那套分词器（jieba），取不到再退回 2-gram。"""
    try:
        from 词频检索 import get_tokenize  # 延迟导入，避免 import 期加载词典

        return [t for t in get_tokenize()(text) if any("\u4e00" <= ch <= "\u9fff" for ch in t)]
    except Exception:  # noqa: BLE001
        return _CJK_RE.findall(text)


_NOISE_RE = _escape_join(NOISE_PATTERNS)

# 预编译：词典顺序 = 长名优先（避免"净利润"先于"归母净利润"命中）
_LEX_SORTED = sorted(
    [(std, alias) for std, aliases in SUBJECT_LEXICON for alias in (std,) + tuple(aliases)],
    key=lambda x: -len(x[1]),
)


def detect_subject(query):
    """从问句里识别「在问哪个科目/指标」，返回 (标准科目名列表, 问句实义词列表)。

    两步：
      1. 科目词典匹配（把"归母净资产"这类写法归一成标准名）
      2. 抽问句实义词：剥掉公司名/年份/报告期/疑问词/虚词后剩下的中文片段
         —— 用于「问法里根本没有科目名」的情形（如"压在仓库里没卖出去的货"）。
    """
    subjects = []
    for std, alias in _LEX_SORTED:
        if alias and alias in query:
            if std not in subjects:
                subjects.append(std)
    subjects = subjects[:2]

    body = str(query)
    for c in _COMPANY_HINT:
        body = body.replace(c, " ")
    body = _YEAR_RE.sub(" ", body)
    for w in _STOPWORDS:
        body = body.replace(w, " ")
    kws, seen = [], set()
    for tok in _tokenize_cjk(body):
        if tok in seen or len(tok) < 1:
            continue
        seen.add(tok)
        kws.append(tok)
    kws = [k for k in kws if len(k) >= 1][:12]
    return subjects, kws


def augment_generic(query, subjects=None, keywords=None):
    """给 BM25 再加一路：原句 + 标准科目名 + 通用报表表头词。

    报表表头词是通用的（"本期期末数""上年同期""占总资产的比例"…），
    用来把「只有科目名、没有上下文」的口语问法拉向真正装数字的那些表。
    """
    if subjects is None:
        subjects, keywords = detect_subject(query)
    terms = list(subjects) + list(keywords or [])[:6]
    terms += ["本期期末数", "上年同期", "占总资产的比例", "主要会计数据", "项目名称"]
    return query + " " + " ".join(terms)


def build_generic_vector(texts, subjects, keywords):
    """按通用报表结构信号算权重向量（1.0 = 不干预）。与具体科目无关。"""
    n = len(texts)
    v = np.ones(n, dtype=np.float32)
    subj_res = [_escape_join([s] + [a for std, als in SUBJECT_LEXICON if std == s for a in als])
                for s in subjects] if subjects else []
    kw_re = _escape_join(keywords) if keywords else None

    for i, t in enumerate(texts):
        m = 1.0
        is_table = TABLE_MARK in t
        has_fig = FIG_RE.search(t) is not None
        row = SUBJECT_ROW_RE.search(t) is not None
        stmt = _STATEMENT_RE.search(t) is not None

        m *= HAS_TABLE_MULT if is_table else NO_TABLE_MULT
        if not has_fig:
            m *= NO_FIGURE_MULT          # 空表头块（"销售费用 单位：元"）
        if row:
            m *= SUBJECT_ROW_MULT
        if stmt:
            m *= STATEMENT_MULT
        if subjects:
            if any(r.search(t) for r in subj_res):
                m *= SUBJECT_HIT_MULT
            else:
                m *= SUBJECT_MISS_MULT
        elif kw_re is not None and kw_re.search(t):
            m *= KEYWORD_MULT
        c = len(_NOISE_RE.findall(t))
        if c:
            m *= NOISE_MULT ** min(c, 2)
        v[i] = m
    return v


def build_generic_mask(texts, subjects, keywords):
    """定向配额掩码：装得下"科目 → 金额"的主表块，且不是受限/子公司/政策类。

    与 metrics.py 的配额机制同构：加权救不回排在几千名外的块，用席位保底。
    """
    subj_res = [_escape_join([s] + [a for std, als in SUBJECT_LEXICON if std == s for a in als])
                for s in subjects] if subjects else []
    kw_re = _escape_join(keywords) if keywords else None
    out = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        if TABLE_MARK not in t or not FIG_RE.search(t):
            continue
        if not (SUBJECT_ROW_RE.search(t) or _STATEMENT_RE.search(t)):
            continue
        if _NOISE_RE.search(t):
            continue
        if subjects and not any(r.search(t) for r in subj_res):
            continue
        out[i] = True
    return out


def generic_vector(texts, query, cache=None):
    """对外入口。返回 (描述串, 权重向量, 配额掩码)。"""
    subjects, keywords = detect_subject(query)
    if not subjects and not keywords:
        return "", None, None
    label = ("科目:" + "/".join(subjects)) if subjects else ("实义词:" + "/".join(keywords[:3]))
    key = (tuple(subjects), tuple(keywords), len(texts))
    if cache is not None:
        if key not in cache:
            cache[key] = (label, build_generic_vector(texts, subjects, keywords),
                          build_generic_mask(texts, subjects, keywords))
        return cache[key]
    return (label, build_generic_vector(texts, subjects, keywords),
            build_generic_mask(texts, subjects, keywords))


if __name__ == "__main__":
    for q in [
        "水井坊2024年末压在仓库里还没卖出去的那些货，账面价值多少钱？",
        "古井贡酒和迎驾贡酒相比，2024年末谁账上的货币资金更多？两家分别有多少？",
        "泸州老窖、洋河股份、酒鬼酒2025年上半年各自的销售费用是多少？哪家最高？",
        "迎驾贡酒2024年度的基本每股收益是多少元？比上一年度是增加还是减少？",
        "这12家白酒公司2025年上半年的经营活动产生的现金流量净额，排名前三的是哪几家？",
    ]:
        s, k = detect_subject(q)
        print(f"科目={s} 实义词={k[:6]}\n  <- {q}")
