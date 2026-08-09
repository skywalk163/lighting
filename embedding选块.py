# -*- coding: utf-8 -*-
"""段言积木『embedding 选块』v0.15 —— 概念图稠密向量检索（零 token，离线可用）。

为什么需要它（解决 v0.14 的『词法盲区』）：
  v0 关键词选块用『字符重叠』打分。需求「斐波那契数列」与「统计双指标」共享一个
  「数」字，竟被误打 7.8 分高置信选中——而它根本不该命中任何块。纯 TF-IDF 只能
  缓解同义改写，无法识别『这个概念库里压根不存在』。

本模块把每个块与查询都映射到一个『受控概念空间』的稠密向量：
  - 若查询命中库内某个概念（如 方差/均值）→ 对应块高相似，正常选中；
  - 若查询命中『库内无对应块』的概念（如 斐波那契/阶乘/素数）→ 向量与所有块余弦≈0
    → 返回空候选 → 上层直接走兜底生成。
这从根本上解决了『误中错块』（而非『漏选』）的问题。

可选升级：若环境装了 sentence_transformers，自动改用真·句向量（仍零 token）。

用法：
    python 积木库/embedding选块.py "计算这组数的方差" --top 3
    python 积木库/embedding选块.py "斐波那契数列第10项" --top 3   # 返回空候选
"""

import argparse
import json
import math
import os

_HERE = os.path.abspath(os.path.dirname(__file__))


def index_path():
    return os.path.join(_HERE, '索引.json')


def load_index(path=None):
    target = path or index_path()
    with open(target, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 概念词典：规范概念 -> 触发词条（含同义/口语/相关词）
# 刻意不把过宽的通用字（数/列/值）映射到具体概念，避免『数』把数列需求
# 误引到统计块——这是修复盲区的核心约束。
# ---------------------------------------------------------------------------
_概念词典 = {
    '求和汇总': ['求和', '加总', '累加', '合计', '总和', '加和', '汇总'],
    '均值平均': ['均值', '平均', '平均值', '平均数', '算平均', '求平均'],
    '极值跨度': ['最大', '极大', '最大值', '最高', '最小', '极小', '最小值', '最低',
                 '范围', '跨度', '极差', '区间'],
    '计数长度': ['计数', '个数', '数量', '多少条', '长度', '字符数', '字数'],
    '方差离散': ['方差', '标准差', '离散', '波动'],
    '中位数':   ['中位数', '中位', '中间值'],
    '财务税务': ['个税', '所得税', '税', '工资', '薪资', '收入', '薪金', '速算'],
    '四舍五入': ['保留', '四舍五入', '取整', '精度', '两位'],
    '分页翻页': ['分页', '翻页', '页码', 'offset', '每页', '页'],
    '文本清洗': ['去空格', '空白', 'trim', '反转', '倒序', '替换', '拼接', '串接',
                 '连接', '切分', '分割', '提取', '包含', '含有', '去重'],
    '大小写':   ['小写', '大写', '大小写', 'tolower', 'toupper'],
    '数字中文': ['人民币大写', '金额大写', '中文大写', '数字转中文', '中文转数字',
                 '念数', '读数', '念出'],
    '数列生成': ['斐波那契', '阶乘', '素数', '质数', '累加和'],
    '列表集合': ['列表', '排序', '去重', '唯一', '偶数', '奇数', '切片', '子列表',
                 '合并', 'range'],
    '数值运算': ['余数', '取余', 'mod', '幂', '次方', '指数', '乘方', '绝对值', '取模'],
    '网络':     ['http', '请求', '接口', 'api'],
    # 领域概念（作为兜底维度，帮助用户说『财务/文本』类词时也能归到对应块）
    '数据':     ['数据', '数值', '统计', '指标'],
    '文本':     ['文本', '字符串', '字符'],
    '财务':     ['财务'],
    '中文':     ['中文'],
    '工具':     ['工具'],
    '生成':     ['生成'],
}

# 扁平化：别名 -> 概念
_别名到概念 = {}
for _c, _ts in _概念词典.items():
    for _t in _ts:
        _别名到概念.setdefault(_t, _c)


def 概念向量(text):
    """把任意文本映射到概念空间的二值向量（概念 -> 1.0）。"""
    text = text or ''
    v = {}
    for _t, _c in _别名到概念.items():
        if _t and _t in text:
            v[_c] = 1.0
    return v


def _块概念向量(b):
    text = (b.get('名称', '') + ' ' + b.get('描述', '') + ' ' + str(b.get('领域', '')))
    v = 概念向量(text)
    d = b.get('领域')
    if d in _概念词典:
        v[d] = 1.0
    return v


def _余弦(a, b):
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# 真·向量检索（可选升级，零 token 仅本地推理）
# ---------------------------------------------------------------------------
def _真向量可用():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _真向量检索(需求, blocks, top):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def _t(b):
        d = b.get('领域')
        d = ' '.join(d) if isinstance(d, list) else str(d)
        return b.get('名称', '') + ' ' + d + ' ' + b.get('描述', '')

    embs = model.encode([_t(b) for b in blocks], normalize_embeddings=True)
    q = model.encode([需求], normalize_embeddings=True)[0]
    sims = embs @ q
    order = sorted(range(len(blocks)), key=lambda i: -sims[i])[:top]
    out = []
    for i in order:
        if sims[i] >= EMBED_FLOOR:
            out.append(_to_candidate(blocks[i], float(sims[i])))
    return out


# ---------------------------------------------------------------------------
# 缓存 + 主入口
# ---------------------------------------------------------------------------
EMBED_FLOOR = 0.08
_缓存 = {}


class _索引:
    def __init__(self, blocks):
        self.blocks = blocks
        self.vecs = [_块概念向量(b) for b in blocks]


def _get(索引):
    key = id(索引)
    if key not in _缓存:
        _缓存[key] = _索引(索引.get('块') or [])
    return _缓存[key]


def _to_candidate(b, score):
    d = b.get('领域') or []
    d0 = d[0] if isinstance(d, list) and d else (d if isinstance(d, str) else '?')
    return {
        '名称': b.get('名称'), '领域': d0, '导出名': b.get('导出名', '?'),
        '路径': b.get('路径', ''), '描述': b.get('描述', ''),
        '分数': round(score, 4),
    }


def embedding_select(需求, index, top=None):
    """概念图 embedding 选块：返回候选列表（与 select_blocks 同构，含 分数）。

    若查询未命中任何库内概念，所有相似度≈0 → 返回空列表，上层据此走兜底。
    """
    blocks = index.get('块') or []
    if not blocks:
        return []

    if _真向量可用():
        try:
            return _真向量检索(需求, blocks, top or 5)
        except Exception:
            pass  # 降级到概念图

    idx = _get(index)
    q = 概念向量(需求)
    scored = []
    for b, vec in zip(idx.blocks, idx.vecs):
        s = _余弦(q, vec)
        if s >= EMBED_FLOOR:
            scored.append((s, b))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    候选 = [_to_candidate(b, s) for s, b in scored]
    return 候选[:top] if top else 候选


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description='段言积木 embedding 选块 v0.15（概念图向量，零 token）')
    p.add_argument('需求', help='自然语言需求文本')
    p.add_argument('--top', type=int, default=5, help='候选数上限')
    args = p.parse_args(argv)

    index = load_index()
    mode = '真·句向量' if _真向量可用() else '概念图向量'
    候选 = embedding_select(args.需求, index, top=args.top)
    print(json.dumps({'需求': args.需求, '模式': mode,
                      '块总数': len(index.get('块') or []),
                      '候选': 候选 if 候选 else '（空：需求未命中任何库内概念）'},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
