# -*- coding: utf-8 -*-
"""段言积木『embedding 选块』v0.16 —— 概念图向量 + 真·句向量（更细语义召回，零 token）。

为什么需要它（解决 v0.14 的『词法盲区』）：
  v0 关键词选块用『字符重叠』打分。需求「斐波那契数列」与「统计双指标」共享一个
  「数」字，竟被误打高置信选中——而它根本不该命中任何块。纯 TF-IDF 只能缓解同义
  改写，无法识别『这个概念库里压根不存在』。

本模块提供两层语义召回，默认走更快更省的概念图，装了 sentence_transformers 后自动
升级为『真·句向量』做更细的语义召回：

  · 概念图向量（默认 / 零依赖）：
    把每个块与查询映射到『受控概念空间』的二值向量。库内无对应概念时余弦≈0 →
    返回空候选 → 上层直接走兜底生成。刻意不把「数/列/值」等过宽字映射到具体概念。

  · 真·句向量（可选 / 零 token 仅本地推理）：
    用 sentence_transformers 把块名+领域+描述 与 查询 编码成句向量，按余弦召回。
    能捕捉「金额写大写 ↔ 转大写」这类字符不重叠但语义相近的细粒度关系，召回更准。
    首次编码后把块向量缓存到 积木库/.embed_cache/，后续调用秒级。

用法：
    python 积木库/embedding选块.py "计算这组数的方差" --top 3
    python 积木库/embedding选块.py "斐波那契数列第10项" --top 3   # 概念图→空候选→兜底
    python 积木库/embedding选块.py "把金额写成人民币大写" --real    # 强制走真·句向量
可选环境变量：
    DUAN_EMBED_MODEL  模型名（默认 shibing624/text2vec-base-chinese）
    DUAN_EMBED_FLOOR  真向量相似度地板（默认 0.15）
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
# v0.16 新增『拼音转换』概念：让「中文转拼音」与「数字转中文」可被本地校验器区分。
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
    '拼音转换': ['拼音', 'pinyin', '转拼音', '拼音转换', '汉语拼音'],
    '数列生成': ['斐波那契', '阶乘', '素数', '质数', '累加和'],
    '列表集合': ['列表', '排序', '去重', '唯一', '偶数', '奇数', '切片', '子列表',
                 '合并', 'range'],
    '数值运算': ['余数', '取余', 'mod', '幂', '次方', '指数', '乘方', '绝对值', '取模'],
    '网络':     ['http', '请求', '接口', 'api'],
    # 领域概念（兜底维度，帮助用户说『财务/文本』类词时也能归到对应块）
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
# 真·句向量检索（可选升级，零 token 仅本地推理；需 sentence_transformers）
# ---------------------------------------------------------------------------
_EMBED_MODEL_ENV = 'DUAN_EMBED_MODEL'
_DEFAULT_MODEL = 'shibing624/text2vec-base-chinese'
_REAL_FLOOR_ENV = 'DUAN_EMBED_FLOOR'
_CACHE_DIR = os.path.join(_HERE, '.embed_cache')

# 概念图向量用的余弦地板（二值向量，相似度天然偏低）
EMBED_FLOOR = 0.08


def _模型名():
    return os.environ.get(_EMBED_MODEL_ENV, _DEFAULT_MODEL)


def _真向量可用():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _块文本(b):
    d = b.get('领域')
    d = ' '.join(d) if isinstance(d, list) else str(d)
    return b.get('名称', '') + ' ' + d + ' ' + b.get('描述', '')


def _缓存路径(model):
    safe = model.replace('/', '__')
    return (os.path.join(_CACHE_DIR, safe + '.emb.npy'),
            os.path.join(_CACHE_DIR, safe + '.meta.json'))


def _真向量检索(需求, blocks, top, model=None):
    """真·句向量召回。命中缓存则复用块向量，仅对查询实时编码。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = model or _模型名()
    m = SentenceTransformer(model)
    emb_path, meta_path = _缓存路径(model)
    sig = [(b.get('名称'), b.get('描述')) for b in blocks]

    embs = None
    if os.path.isfile(emb_path) and os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, encoding='utf-8'))
            if meta.get('sig') == sig:
                embs = np.load(emb_path)
        except Exception:
            embs = None
    if embs is None:
        embs = m.encode([_块文本(b) for b in blocks], normalize_embeddings=True)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.save(emb_path, embs)
        json.dump({'sig': sig, 'model': model},
                  open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False)

    q = m.encode([需求], normalize_embeddings=True)[0]
    sims = embs @ q
    floor = float(os.environ.get(_REAL_FLOOR_ENV, '0.15'))
    order = sorted(range(len(blocks)), key=lambda i: -sims[i])[: (top or 5)]
    out = []
    for i in order:
        if sims[i] >= floor:
            out.append(_to_candidate(blocks[i], float(sims[i])))
    return out


# ---------------------------------------------------------------------------
# 缓存 + 主入口
# ---------------------------------------------------------------------------
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


def embedding_select(需求, index, top=None, real=None):
    """语义选块：返回候选列表（与 select_blocks 同构，含 分数）。

    - 默认：概念图向量（零依赖）。库外概念→余弦≈0→空候选→上层兜底。
    - 若装了 sentence_transformers（且 real 不为 False），自动改用真·句向量做更细召回。
    - real=True 时强制走真向量（不可用则报错提示）。
    """
    blocks = index.get('块') or []
    if not blocks:
        return []

    use_real = real if real is not None else _真向量可用()
    if use_real:
        try:
            return _真向量检索(需求, blocks, top or 5)
        except Exception as e:
            if real is True:
                raise
            print('[embedding] 真向量检索失败，降级概念图：%s' % e)

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
        description='段言积木 embedding 选块 v0.16（概念图 / 真·句向量）')
    p.add_argument('需求', help='自然语言需求文本')
    p.add_argument('--top', type=int, default=5, help='候选数上限')
    p.add_argument('--real', action='store_true',
                   help='强制走真·句向量（需 sentence_transformers）')
    args = p.parse_args(argv)

    index = load_index()
    if args.real and not _真向量可用():
        print('[embedding] 未安装 sentence_transformers，无法使用 --real')
        return 1
    mode = '真·句向量' if (args.real or _真向量可用()) else '概念图向量'
    候选 = embedding_select(args.需求, index, top=args.top,
                          real=(True if args.real else None))
    print(json.dumps({'需求': args.需求, '模式': mode,
                      '块总数': len(index.get('块') or []),
                      '候选': 候选 if 候选 else '（空：需求未命中任何库内概念）'},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
