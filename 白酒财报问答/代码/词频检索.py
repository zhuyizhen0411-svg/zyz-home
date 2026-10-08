# -*- coding: utf-8 -*-
"""BM25 倒排索引与中文分词。

单独成模块的原因：BM25 对象要 pickle 落盘，如果类定义在 __main__ 里，
其它进程反序列化时会报 `Can't get attribute 'BM25' on <module '__main__'>`。
"""

import pickle

import numpy as np

_tok = None


def get_tokenize():
    """延迟初始化：jieba 首次加载词典约 0.2s，不要在 import 时做。"""
    global _tok
    if _tok is None:
        _tok = tokenize
    return _tok


try:
    import jieba

    def tokenize(text):
        return [t for t in jieba.cut(text) if len(t.strip()) > 1]

    TOKENIZER = "jieba"
except Exception:  # noqa: BLE001

    def tokenize(text):
        text = "".join(ch if ch.isalnum() or "\u4e00" <= ch <= "\u9fff" else " " for ch in text)
        toks = []
        for seg in text.split():
            if "\u4e00" <= seg[0] <= "\u9fff":
                toks += [seg[i : i + 2] for i in range(len(seg) - 1)] or [seg]
            else:
                toks.append(seg)
        return toks

    TOKENIZER = "bigram"


STOP = set("的了和与及在对为是不有公司本报告期内其中以及我们之与或年月日")


class BM25:
    """Okapi BM25，倒排索引形式，支持按词频打分。"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.df = {}
        self.tf = []  # 每篇: {term: count}
        self.lens = []
        self.avgdl = 0.0
        self.N = 0

    def fit(self, docs):
        self.N = len(docs)
        self.tf = []
        self.lens = []
        for d in docs:
            cnt = {}
            for t in d:
                if t in STOP:
                    continue
                cnt[t] = cnt.get(t, 0) + 1
            self.tf.append(cnt)
            self.lens.append(sum(cnt.values()))
            for t in cnt:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (sum(self.lens) / self.N) if self.N else 0.0
        return self

    def idf(self, t):
        n = self.df.get(t, 0)
        return float(np.log(1 + (self.N - n + 0.5) / (n + 0.5)))

    def scores(self, query_tokens):
        """返回 ndarray(N,) 的 BM25 得分。"""
        out = np.zeros(self.N, dtype=np.float32)
        q = [t for t in query_tokens if t in self.df]
        for t in q:
            idf = self.idf(t)
            for i, cnt in enumerate(self.tf):
                f = cnt.get(t)
                if not f:
                    continue
                dl = self.lens[i]
                out[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
