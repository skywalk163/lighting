# -*- coding: utf-8 -*-
"""段言积木组合总入口 v0.16：需求 + 输入 → 选块 →（契约级接线）→ 校验 → 内联粘合 → 运行。

v0.16 变更（更细语义召回 + 真实 LLM 校验）：
  - embedding 选块升级：概念图向量（默认，零依赖）之际，若装了 sentence_transformers
    自动改用真·句向量（更细语义召回，块向量磁盘缓存）；--real 可强制走真向量
  - LLM 校验器（校验器.validate）作为运行前第二道闸：本地概念交集 + 真实 LLM 判定，
    系统提示内置 few-shot 覆盖『中文转拼音 vs 数字转中文』等细微错配；传给 LLM 的候选
    已富化含输入/输出类型
  - 生成/ 积木自动织入 L1+（层级生成.自动织，--自动层级 触发）

零 token 为常态：选块/校验/接线/层级均为本地；仅『兜底生成』在本地规则也覆盖不了、
且配置了 LLM key 时，才调用一次模型。无 key 时本地规则已覆盖 方差/中位数/绝对值/
斐波那契/阶乘/素数/累加和 等常见库外意图。

「选不准」触发兜底的三类信号：
  1) 无候选（embedding 概念图相似度≈0，即需求库内无对应概念）
  2) top 分数低于阈值，或 契约级接线不可接
  3) 校验器判定 MISMATCH（本地概念无交集 / LLM 认为组合不满足需求）
真实 LLM 仅在以上且本地规则也覆盖不了时才被调用。
"""

import argparse
import os
import sys
import subprocess

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, '..'))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from 选块 import select_blocks, load_index
from 粘合 import synthesize
from 语义选块 import semantic_select
from embedding选块 import embedding_select
from 混合选块 import hybrid_select
import 计划缓存
from 校验器 import validate
from 接线 import 规划, 不可接, 回退步, _推断类型, _匹配度
from 兜底生成器 import generate_block, 注册, local_rule_block


def _全量查表(索引):
    return {b['名称']: b for b in (索引.get('块') or [])}


def _可单参调用(b, 输入类型):
    """块能否用单个共享输入直接调用：恰好 1 个入参且类型接得上。

    v0.18 起用类型系统 v2 判定（列表[数] vs 列表[文本] 可分辨），不再要求字符串
    全等。注意这里只做**否决**不做**排序**：类型能说「接不上」，不能说「哪个更
    符合语义」；候选顺序仍由选块器决定，避免类型匹配度 1.0 的块抢掉语义更贴合但
    标注为 列表[任意] 的块（如 排序列表）。
    """
    if not b:
        return False
    ins = b.get('输入') or []
    return len(ins) == 1 and _匹配度(ins[0].get('类型'), 输入类型) > 0


def _条目转候选(b, 分数=1.0):
    """把 索引.json 里的块条目转成选块候选的形状。"""
    d = b.get('领域') or []
    d0 = d[0] if isinstance(d, list) and d else (d if isinstance(d, str) else '?')
    return {'名称': b.get('名称'), '领域': d0, '导出名': b.get('导出名', '?'),
            '路径': b.get('路径', ''), '描述': b.get('描述', ''), '分数': 分数}


def _默认常数():
    """链式接线遇同类型缺口时填入的常识默认值（配置型参数优先于数据流变量）。"""
    return {'每页': 10, '每页大小': 10, '页大小': 10, '步': 1, '起': 0, '起值': 0}


def _默认阈值(关键词, 语义):
    if 关键词:
        return 3.0
    if 语义:
        return 0.12
    return 0.06  # embedding 概念图 / 混合（余弦，已内置 0.08 地板）


def _建步骤(选中):
    return [{
        '块': c['名称'], '领域': c['领域'], '导出名': c['导出名'],
        '路径': c.get('路径', ''), '说明': c['描述'], '参数': [],
    } for c in 选中]


def _造方案(需求, 共享, 步骤):
    return {
        '需求': 需求,
        '共享': 共享,
        '步骤': 步骤,
        '打印': ['赵果%d' % (i + 1) for i in range(len(步骤))],
    }


def _装配(需求, 候选列表, 输入值, 查表, 链式, top):
    """选块候选 → 可运行方案（非链式并行 / 链式契约接线）。

    返回 方案；若链式接线不可接返回 None（交由上层走兜底）。从 组合() 内联逻辑
    抽出为模块级，供「执行闭环」重试时复用（用单个次优候选重装方案）。
    """
    共享 = [{'名': '赵料', '值': 输入值, '类型': _推断类型(输入值)}]
    if 链式:
        wired, _ = 规划(_建步骤(候选列表[:top]), 共享, 查表, _默认常数())
        if 不可接(wired):
            return None
        退 = 回退步(wired)
        if 退:
            # 类型上接得通，但没接上游产物 ⇒ 实际退化成并行，明说而不是静默
            print('[链式] 这些步骤接不上任何上游产物，已退回共享输入（实为并行）：'
                  + '、'.join(退))
        return _造方案(需求, 共享, wired)

    # 非链式并行装配：所有步骤共用同一个输入，因此必须过滤掉
    # 『签名接不上』的块——否则会合成出 留分([1,2,3]) 这种类型错误的代码。
    可用 = [c for c in 候选列表
            if _可单参调用(查表.get(c['名称']), 共享[0]['类型'])]
    if not 可用:
        print('[类型闸门] 候选块都不接受单个「%s」输入，按原候选装配（可能失败）'
              % 共享[0]['类型'])
        可用 = 候选列表
    elif len(可用) < len(候选列表[:top]):
        跳过 = [c['名称'] for c in 候选列表[:top]
                if c['名称'] not in [k['名称'] for k in 可用]]
        print('[类型闸门] 跳过签名不匹配的块：' + '、'.join(跳过))
    步骤 = _建步骤(可用[:top])
    for s in 步骤:
        s['参数'] = ['赵料']
    return _造方案(需求, 共享, 步骤)


def _兜底(需求, 索引, 候选, 输入值, 块=None, 理由=''):
    print('[兜底] %s，调用生成器：%s' % (理由 or '选块未命中', 需求))
    blk = 块 if 块 is not None else generate_block(需求, 索引, 候选=候选, 库根=_HERE)
    if not blk:
        print('[兜底] 本地规则也无法生成，需配置真实 LLM（见 llm_config.json）')
        return None
    注册(blk, 库根=_HERE)
    blk = dict(blk)
    blk['分数'] = 0.0
    print('[兜底] 已生成并注册新积木：%s（%s）' % (blk['名称'], blk.get('路径')))

    查表 = _全量查表(load_index())
    步骤 = [{
        '块': blk['名称'], '领域': blk.get('领域', '生成'),
        '导出名': blk['导出名'], '路径': blk.get('路径', ''),
        '说明': blk.get('描述', ''), '参数': [],
    }]
    共享 = [{'名': '赵料', '值': 输入值, '类型': _推断类型(输入值)}]
    wired, _ = 规划(步骤, 共享, 查表, _默认常数())
    步骤 = wired
    if 不可接(步骤):
        for s in 步骤:
            s['参数'] = ['赵料']
    方案 = _造方案(需求, 共享, 步骤)
    方案['_兜底'] = True
    return 方案, [blk]


def 组合(需求, 输入值="[1, 2, 3, 4, 5]", top=3, 语义=False, 关键词=False,
        链式=False, 阈值=None, 无兜底=False, 无校验=False, 自动层级=False,
        混合=False, 无缓存=False):
    索引 = load_index()
    查表 = _全量查表(索引)
    策略 = ('关键词' if 关键词 else '语义' if 语义
            else '混合' if 混合 else '概念图')
    输入类型 = _推断类型(输入值)
    # 阈值在第 2 步会被就地填成策略默认值（如 0.06）。缓存键必须两头用同一个值，
    # 否则「读时 None / 写时 0.06」永远算不出同一个键 —— 写得进去、读不出来。
    阈值原 = 阈值

    # 0) 计划缓存：同一需求 + 同一库 + 同一策略 ⇒ 选块/校验/接线的结论必然相同。
    #    库指纹变了（兜底生成新块、契约改动）缓存自动整体作废，不会拿旧方案硬套。
    if not 无缓存:
        命中 = 计划缓存.读(需求, 索引, 策略=策略, top=top, 阈值=阈值原,
                        链式=链式, 输入类型=输入类型)
        if 命中:
            print('[缓存] 命中计划（第 %d 次复用）：%s'
                  % (命中['命中次数'], '+'.join(s.get('块', '?')
                                              for s in 命中['步骤'])))
            共享 = [{'名': '赵料', '值': 输入值, '类型': 输入类型}]
            方案 = _造方案(需求, 共享, 命中['步骤'])
            候选 = [_条目转候选(查表[n]) for n in 命中['候选'] if n in 查表]
            return 方案, 候选

    # 1) 选块
    if 关键词:
        候选 = select_blocks(需求, 索引, top=top)
    elif 语义:
        候选 = semantic_select(需求, 索引, top=top)
    elif 混合:
        候选 = hybrid_select(需求, 索引, top=top)
    else:
        候选 = embedding_select(需求, 索引, top=top)

    # 2) 是否需要兜底
    需要兜底 = False
    理由 = ''
    if not 候选:
        需要兜底, 理由 = True, '无候选（需求未命中任何库内概念）'
    else:
        分数 = 候选[0]['分数']
        if 阈值 is None:
            阈值 = _默认阈值(关键词, 语义)
        if 分数 < 阈值:
            需要兜底, 理由 = True, 'top 分数 %.3f 低于阈值 %.2f' % (分数, 阈值)
        elif not 无校验:
            # 校验器需要候选的完整契约（输入/输出类型），从全量查表还原
            候选_full = [查表.get(c['名称']) or c for c in 候选]
            v = validate(需求, 候选_full)
            if not v['通过']:
                需要兜底, 理由 = True, '校验未过：' + v['理由']

    if 需要兜底:
        if 无兜底:
            print('已关闭兜底，无法生成方案：' + 需求)
            return None
        res = _兜底(需求, 索引, 候选, 输入值, 理由=理由)
        if not res:
            return None
        方案, 候选 = res
    else:
        # 3) 正常装配
        方案 = _装配(需求, 候选, 输入值, 查表, 链式, top)
        if 方案 is None:
            if 无兜底:
                print('接线不可接且已关闭兜底：' + 需求)
                return None
            res = _兜底(需求, 索引, 候选, 输入值, 理由='契约级接线不可接')
            if not res:
                return None
            方案, 候选 = res

        # 4) 能力缺失预检（零 token）：本地规则识别出需求真正需要的能力
        elif not 无兜底:
            lr = local_rule_block(需求)
            if lr and lr['名称'] not in [c['名称'] for c in 候选]:
                已有 = 查表.get(lr['名称'])
                if 已有:
                    # 库内已有该能力，只是选块没排上来 ⇒ 纠正选块，绝不重复生成同名块
                    # （否则 索引.json 会被同名块反复污染，这是基准跑分暴露出的真 bug）
                    print('[纠正] 库内已有更贴合的积木「%s」，改用它（不重复生成）'
                          % lr['名称'])
                    候选 = [_条目转候选(已有, 1.0)] + \
                        [c for c in 候选 if c['名称'] != lr['名称']]
                    方案 = _装配(需求, 候选, 输入值, 查表, 链式, top) or 方案
                else:
                    res = _兜底(需求, 索引, 候选, 输入值, 块=lr,
                              理由='能力缺失（本地规则）')
                    if res:
                        方案, 候选 = res

    # 5) 生成块自动织入 L1+（可选）
    if 自动层级:
        from 层级生成 import 自动织
        建 = 自动织(_HERE)
        if 建:
            print('[自动层级] 新建 L1 积木：' + '、'.join(建))

    # 6) 落缓存。兜底/自动层级刚改过库，这里重新 load 一次让库指纹对上新状态，
    #    否则写进去的条目下一次必然因指纹不符而作废。
    if not 无缓存:
        try:
            计划缓存.写(需求, load_index(), 方案['步骤'], 候选, 策略=策略,
                      top=top, 阈值=阈值原, 链式=链式, 输入类型=输入类型,
                      兜底=bool(方案.get('_兜底')))
        except Exception as e:
            print('[缓存] 写入失败（不影响本次结果）：%s' % e)

    return 方案, 候选


def _运行_单次(方案, 输出, duan):
    """合成 → 运行单个段言文件，返回 (rc, stdout, stderr)。供执行闭环重试复用。"""
    code = synthesize(方案)
    with open(输出, 'w', encoding='utf-8') as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, duan, 'run', 输出],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 124, '', '运行超时（>60s）'
    except Exception as e:
        return 1, '', '运行异常：%s' % e
    return r.returncode, r.stdout, r.stderr


def _成功(rc, out):
    """判断一次段言运行是否真的成功。

    段言 src 后端（cli/duan.py run 默认）会**静默吞掉运行期错误**：
    `a[99]` 越界、`1 除 0` 等都返回 rc=0 且 stdout 为空。
    因此『成功』必须同时要求 rc==0 **且** 有非空 stdout —— 正常生成的方案
    必有「打印」步骤，成功必有输出；只有崩溃才会 rc=0 且空输出。
    （解析错误仍会返回 rc!=0，被此判定一并拦下。）
    """
    return rc == 0 and bool(out.strip())


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description='段言积木组合 v0.16（embedding 真向量选块 + LLM 校验器 + 生成块自动层级）')
    p.add_argument('需求', help='自然语言需求文本')
    p.add_argument('--输入', default='[1, 2, 3, 4, 5]', help='共享输入（段言表达式）')
    p.add_argument('--top', type=int, default=3, help='选块候选数（= 步骤数）')
    p.add_argument('--关键词', action='store_true', help='用 v0 关键词选块（字符重叠）')
    p.add_argument('--语义', action='store_true', help='用语义选块（TF-IDF+同义词）')
    p.add_argument('--链式', action='store_true', help='契约级精确接线（按类型+参数名）')
    p.add_argument('--阈值', type=float, default=None, help='触发兜底的分数阈值')
    p.add_argument('--无兜底', action='store_true', help='关闭 LLM 兜底')
    p.add_argument('--无校验', action='store_true', help='跳过运行前校验器')
    p.add_argument('--自动层级', action='store_true', help='把 生成/ 积木自动织成 L1+')
    p.add_argument('--混合', action='store_true',
                   help='混合选块：概念图召回（空则 TF-IDF 补召回）+ 并列群语义重排')
    p.add_argument('--无缓存', action='store_true', help='跳过计划缓存，强制重算')
    p.add_argument('-o', '--输出', default=os.path.join(_HERE, '组合结果.duan'))
    args = p.parse_args(argv)

    res = 组合(args.需求, 输入值=args.输入, top=args.top,
              语义=args.语义, 关键词=args.关键词, 链式=args.链式,
              阈值=args.阈值, 无兜底=args.无兜底, 无校验=args.无校验,
              自动层级=args.自动层级)
    if not res:
        print('未能生成方案：' + args.需求)
        return 1
    方案, 候选 = res
    选块法 = '关键词' if args.关键词 else ('语义' if args.语义 else 'embedding')
    是兜底 = bool(方案.get('_兜底'))
    print('选块候选（%s%s）：' % (选块法, ' + 兜底生成' if 是兜底 else ''))
    for c in 候选:
        print('  %s（%s）分数=%s' % (c['名称'], c['领域'], c['分数']))

    duan = os.path.join(_REPO, 'cli', 'duan.py')
    索引 = load_index()
    查表 = _全量查表(索引)
    # 执行闭环：主方案跑挂自动换次优候选重跑，都挂再触发兜底生成。
    # 主方案已含 top 候选的装配结果；其余候选各装成单体方案依次试跑。
    # 注意：段言 src 后端会静默吞掉 运行期错误（rc=0 且 stdout 为空），
    # 故「成功」判定必须是 rc==0 且 有非空输出（见 _成功）。
    尝试 = [('主方案', 方案)]
    if not 是兜底 and not args.链式:
        for i, c in enumerate(候选[1:], 1):
            备 = _装配(args.需求, [c], args.输入, 查表, False, args.top)
            if 备:
                尝试.append(('候选%d:%s' % (i + 1, c['名称']), 备))
    成功 = False
    rc = None
    for 标签, 方案_i in 尝试:
        print('\n── 运行（%s）──' % 标签)
        rc, out, err = _运行_单次(方案_i, args.输出, duan)
        if out.strip():
            print(out.rstrip())
        if _成功(rc, out):
            成功 = True
            print('[执行闭环] %s 运行成功 ✓' % 标签)
            break
        print('[执行闭环] %s 运行失败（rc=%d%s），自动换下一候选…'
              % (标签, rc, '' if out.strip() else ' 且输出为空（疑似运行期静默崩溃）'))
        if err.strip():
            print(err.strip()[-600:])
    if not 成功 and not 是兜底 and not args.无兜底:
        print('\n[执行闭环] 所有候选均失败，触发兜底生成重跑…')
        res2 = _兜底(args.需求, 索引, 候选, args.输入, 理由='运行期崩溃，候选均不满足')
        if res2:
            方案2, _ = res2
            print('\n── 运行（兜底生成）──')
            rc, out, err = _运行_单次(方案2, args.输出, duan)
            if out.strip():
                print(out.rstrip())
            if not _成功(rc, out):
                print('[执行闭环] 兜底生成仍未能产出正确结果（需配置真实 LLM）')
                if err.strip():
                    print(err.strip()[-600:])
    return 0 if 成功 else (rc or 1)


if __name__ == '__main__':
    raise SystemExit(_cli())
