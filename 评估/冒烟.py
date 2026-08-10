# -*- coding: utf-8 -*-
"""段言积木『冒烟测试』v0.19 —— 第四把尺子：每一块**真的跑得起来吗**？

前三把尺子分别量「选块对不对」「接线对不对」「契约自洽吗」，但它们都不执行代码。
一块 `.duan` 完全可能契约漂亮、被正确选中、参数接得完美，然后在运行时炸掉
（词法器把标识符切碎、调了不存在的内建、用了静默失效的默认参数……）。
段言的这些坑都只在**运行期**暴露，所以必须有一把尺子真的把每块跑一遍。

做法：对索引里的每一块，
  1. 按 `样例` 字段（若有）或**结构化类型**推导一组默认实参；
  2. 内联该块源码 + 一行调用，写成临时 .duan；
  3. `python cli/duan.py run` 执行，比对是否报错 / 是否命中 `期望` 值。

类型驱动的默认实参正是类型系统 v2 的红利：契约写成 `列表[数]` 才能自动造出
`[3, 1, 4, 1, 5]` 而不是瞎猜。

用法：
    python 积木库/评估/冒烟.py                 # 跑全库
    python 积木库/评估/冒烟.py --只跑 日期 校验  # 只跑指定领域
    python 积木库/评估/冒烟.py --块 闰年 星期几  # 只跑指定块
    python 积木库/评估/冒烟.py --详细           # 打印每块的实参与返回值
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.abspath(os.path.dirname(__file__))
_LIB = os.path.abspath(os.path.join(_HERE, '..'))
_ROOT = os.path.abspath(os.path.join(_LIB, '..'))

sys.path.insert(0, _LIB)
import 类型 as T  # noqa: E402


# ---------------------------------------------------------------------------
# 类型 → 默认实参（段言字面量文本）
# ---------------------------------------------------------------------------
_默认标量 = {
    '数': '6',
    '文本': '"Hello World"',
    '逻辑': '真',
    '任意': '6',
    '空': '空',
}
_默认列表 = {
    '数': '[3, 1, 4, 1, 5]',
    '文本': '["banana", "apple", "cherry"]',
    '逻辑': '[真, 假, 真]',
    '任意': '[3, 1, 4, 1, 5]',
}


def 默认实参(类型串):
    """按结构化类型造一个段言字面量。造不出来返回 None（跳过该块）。"""
    try:
        t = T.解析(类型串 or '任意')
    except Exception:
        return None
    kind = t[0]
    if kind == '基础':
        return _默认标量.get(t[1], '6')
    if kind == '列表':
        inner = t[1]
        if inner[0] == '基础':
            return _默认列表.get(inner[1], '[3, 1, 4, 1, 5]')
        return '[[1, 2], [3, 4]]'
    if kind == '字典':
        return '{"a": 1, "b": 2}'
    if kind == '联合':
        return 默认实参(T.格式化(t[1][0]))
    return None


def 实参表(块):
    """优先用块自带 `样例`（列表，逐参数的段言字面量），否则按类型推导。"""
    样例 = 块.get('样例')
    if isinstance(样例, list) and 样例:
        return [str(x) for x in 样例]
    out = []
    for p in (块.get('输入') or []):
        a = 默认实参(p.get('类型'))
        if a is None:
            return None
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _块源码(路径):
    with open(os.path.join(_LIB, 路径), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith('#') or s.startswith('导出'):
            continue
        out.append(ln.rstrip('\n'))
    return '\n'.join(out).strip('\n')


def 跑一块(块, python=None):
    路径 = 块.get('路径') or ''
    if not os.path.isfile(os.path.join(_LIB, 路径)):
        return {'名称': 块.get('名称'), '状态': '缺文件', '详情': 路径}
    args = 实参表(块)
    if args is None:
        return {'名称': 块.get('名称'), '状态': '跳过', '详情': '无法为该契约造默认实参'}

    调用 = '%s(%s)' % (块.get('导出名'), ', '.join(args))
    源 = _块源码(路径) + '\n\n打印 ' + 调用 + '\n'

    # 复用同一个工位文件：逐块建删临时文件会被批量删除保护拦下，也没必要
    工位 = os.path.join(_LIB, '_冒烟工位.duan')
    tmp = 工位
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(源)
    if True:
        r = subprocess.run(
            [python or sys.executable, os.path.join(_ROOT, 'cli', 'duan.py'),
             'run', tmp],
            capture_output=True, text=True, encoding='utf-8',
            errors='replace', cwd=_ROOT, timeout=120)
        out = (r.stdout or '').strip()
        # v0.24：段言 src 后端会**静默吞掉运行期错误**（越界/除零都 rc=0 且 stdout 空，
        # 仅解析错误才 rc≠0），因此成功信号对齐 组合.py `_成功`：rc==0 且 有非空 stdout。
        ok = (r.returncode == 0) and bool(out) and not out.startswith('错误:') \
            and ('Traceback' not in (r.stderr or ''))
        if not ok:
            首行 = (out.splitlines() or [''])[0]
            细 = (out.splitlines() or ['', ''])[1].strip() if len(out.splitlines()) > 1 else ''
            return {'名称': 块.get('名称'), '状态': '失败',
                    '实参': args, '详情': (首行 + ' ' + 细).strip()[:160]}
        实 = out.splitlines()[-1] if out else ''
        期望 = 块.get('期望')
        if 期望 is not None and str(期望) != 实:
            return {'名称': 块.get('名称'), '状态': '值不符',
                    '实参': args, '详情': '期望 %s 实得 %s' % (期望, 实)}
        return {'名称': 块.get('名称'), '状态': '通过', '实参': args, '返回': 实}


def 跑(领域=None, 块名=None, 详细=False):
    with open(os.path.join(_LIB, '索引.json'), 'r', encoding='utf-8') as f:
        index = json.load(f)
    blocks = index.get('块') or []
    if 领域:
        want = set(领域)
        blocks = [b for b in blocks
                  if want & set(b['领域'] if isinstance(b['领域'], list) else [b['领域']])]
    if 块名:
        want = set(块名)
        blocks = [b for b in blocks if b.get('名称') in want]

    print('══ 积木冒烟测试 ══')
    print('待测 %d 块' % len(blocks))
    结果 = [跑一块(b) for b in blocks]

    统计 = {}
    for r in 结果:
        统计[r['状态']] = 统计.get(r['状态'], 0) + 1
    通过 = 统计.get('通过', 0)
    可判 = sum(v for k, v in 统计.items() if k != '跳过')
    print('── 状态 ──  ' + '　'.join('%s %d' % (k, v) for k, v in sorted(统计.items())))
    print('── 可运行率 ──  %.4f  (%d/%d)'
          % ((通过 / 可判) if 可判 else 1.0, 通过, 可判))

    坏 = [r for r in 结果 if r['状态'] in ('失败', '值不符', '缺文件')]
    if 坏:
        print('── 问题块（%d）──' % len(坏))
        for r in 坏:
            print('  %-12s %-6s %s' % (r['名称'], r['状态'], r.get('详情', '')))
    if 详细:
        print('── 明细 ──')
        for r in 结果:
            print('  %-12s %-6s %s → %s'
                  % (r['名称'], r['状态'],
                     ', '.join(r.get('实参') or []), r.get('返回', r.get('详情', ''))))
    return 0 if not 坏 else 1


def _cli(argv=None):
    p = argparse.ArgumentParser(description='段言积木冒烟测试（每块真跑一遍）')
    p.add_argument('--只跑', nargs='+', dest='领域', help='只跑这些领域')
    p.add_argument('--块', nargs='+', dest='块名', help='只跑这些块')
    p.add_argument('--详细', action='store_true')
    a = p.parse_args(argv)
    return 跑(领域=a.领域, 块名=a.块名, 详细=a.详细)


if __name__ == '__main__':
    raise SystemExit(_cli())
