# -*- coding: utf-8 -*-
"""LLM 兜底生成器 v0.14。

触发场景：选块器对某个需求「选不准」——无候选 或 top 分数低于阈值，或契约级
接线出现无法补齐的参数（无匹配类型且无可默认）。此时调用 LLM（OpenAI 兼容的
chat/completions）按契约生成一块全新的段言积木；无 API key 时降级为本地规则
生成器（覆盖 方差/标准差/中位数/绝对值 等常见但库内缺失的块）。

生成的块写入 积木库/生成/<名称>.duan，并注册进 索引.json，下次同需求零 token 复用。

配置（优先级从高到低，任选其一即可）：
  - 本地密钥文件：积木库/.env（OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL，不入库）
  - 环境变量：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
  - 文件：积木库/llm_config.json {"api_key": "...", "base_url": "...", "model": "..."}
"""

import json
import os
import urllib.request

_HERE = os.path.abspath(os.path.dirname(__file__))


def _read_dotenv():
    """读取 .env 文件注入环境变量（仅当对应变量尚未设置）。零依赖，失败静默。"""
    p = os.path.join(_HERE, '.env')
    if not os.path.isfile(p):
        return
    try:
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, v = line.partition('=')
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def load_config():
    # .env（本地密钥，不入库）优先级最高：先注入环境变量
    _read_dotenv()
    # 1) 以环境变量为基础（已含 .env 注入值）
    cfg = {
        'base_url': os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'model': os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
    }
    # 2) llm_config.json 仅补充环境变量中缺失的项（.env / 环境变量优先，便于本地覆盖）
    p = os.path.join(_HERE, 'llm_config.json')
    if os.path.isfile(p):
        try:
            j = json.load(open(p, encoding='utf-8'))
            for k in ('base_url', 'api_key', 'model'):
                if not cfg.get(k) and j.get(k):
                    cfg[k] = j[k]
        except Exception:
            pass
    # 3) DUAN_NO_LLM=1 强制关闭 LLM（校验器/兜底一并降级为本地规则）。
    #    评估与 CI 必须可复现且零 token：一旦 .env 配了真实 key，校验器会转为 LLM 判定，
    #    主基准结果随模型漂移（v0.28 实测把 89 条基准从 1.0 打到 0.9888），故 CI 默认置此位。
    if str(os.environ.get('DUAN_NO_LLM', '')).lower() in ('1', 'true', 'yes', 'on'):
        cfg['api_key'] = ''
    return cfg


_SYSTEM = """你是一个段言(Duan)中文编程语言的积木生成器。段言使用中文关键字：
段落 名 接收 参数： 定义函数；设 x 为 表达式。 赋值；返回 表达式。 返回；
当 条件： ... 当 结束 循环；如果 条件： 否则如果 条件： 否则： 分支；
列表索引 表[i]（i 必须是整数，用 整数(...) 取整）；长度(表)；
中缀算术：A 加 B / A 减 B / A 乘 B / A 除 B（除 是真除法返回浮点，整数除法用 整数(A 除 B) 或 A 整除 B）；
取余：A 模 B；布尔比较：A 大于 B / A 小于 B / A 等于 B / A 不等于 B。
整数循环：设 i 为 0；当 i 小于 长度(表)： ... 设 i 为 i 加 1。
列表排序用 表.排序()（原地排序，作为语句调用，不接收返回值）。

一个积木文件的格式（注意：导出 名 与 段落 名 必须一致）：
# 注释
导出 块名
段落 块名 接收 输入名：
    设 ... 为 ...
    返回 结果

【硬性约束（v0.28 由真实兜底首跑实测归纳，违反必定过不了体检/冒烟）】
1. 块名与函数名**禁止含中文数字**（十/百/千/万/一/二/三/…）：词法器会把 `十` 识别为数字，
   把标识符切碎导致语法错误。例：`十进制转八进制` ✗ → 改用 `转八进制` ✓。名字控制 2-6 字。
2. 契约类型只准用：`数`、`文本`、`逻辑`、`任意`、`空`、`列表[数]`、`列表[文本]`、`字典[文本,数]`。
   不要写 `字符串`/`整数`/`浮点数`。
3. 只能用下列已注册内建：打印、长度、求和、求最大、求最小、绝对值、范围、整数、浮点数、
   字符串、列表、字典、筛选、映射、枚举、去重、解析JSON、序列化JSON。
   **禁止臆造**（`随机()`/`文本()`/`余弦()`/`平方根()` 等均不存在）。需要开方就用连乘或牛顿迭代。
4. 字符串只用方法式：`.替换(旧,新)`、`.分割(分隔)`、`.包含(子串)`、`.去除空白()`、`.转小写()`、
   `.转大写()`、`.长度()`；不要写 `字符串替换(...)` 这种函数式。
5. 索引下标必须是整数：用 `整数(a 除 b)`；字面量后不能直接跟下标（先 `设 L 为 [...]` 再 `L[0]`）。
6. 每个参数都必须显式使用；没有默认参数机制。取字符用 `文本[i]`，越界会静默崩溃，务必先判长度。
7. **禁止弯引号**：段言词法器不识别 U+201C/U+201D/U+2018/U+2019（“”‘’），出现即 `未知字符` 词法报错。
   字符串一律用直引号 `"`。例：`返回 “偶数”` ✗ → `返回 "偶数"` ✓。
8. `去重(列表)` 直接返回去重后的新列表；必须 `设 X 为 去重(列表)` 再对 X 用 `长度(X)`/`X[i]`，
   不要对 `去重(列表)` 直接套 `长度()`（会拿到函数对象报 `object of type 'function' has no len()`）。
9. `如果` 条件里**不要直接写** `方法调用(...) 等于 假`：这种内联写法会被词法器静默吞错
   （rc=0 且空输出，体检/冒烟判失败）。务必先 `设 命中 为 方法调用(...) 等于 假` 再 `如果 命中 等于 假：`。
   （`设 命中 为 ...` 的形式则正常。）

请只输出一个 JSON 对象，结构：
{"名称":"...","领域":"生成","层级":0,"描述":"...","输入":[{"名":"序列","类型":"列表"}],"输出":{"类型":"数"},"导出名":"块名","源码":"<完整 .duan 文件文本，含注释/导出/段落，换行必须写成 \\n>"}

下面给出一例【正确】的完整输出（以「计算中位数」为例），请严格照此格式与段言语法生成：
{"名称":"中位数","领域":"生成","层级":0,"描述":"计算一组数的中位数","输入":[{"名":"序列","类型":"列表[数]"}],"输出":{"类型":"数"},"导出名":"中位数","源码":"# 积木：中位数\\n导出 中位数\\n段落 中位数 接收 序列：\\n    设 排序表 为 []\\n    设 i 为 0\\n    当 i 小于 长度(序列)：\\n        排序表.追加(序列[i])\\n        设 i 为 i 加 1\\n    排序表.排序()\\n    设 计数 为 长度(排序表)\\n    设 中 为 整数(计数 除 2)\\n    如果 计数 模 2 等于 0：\\n        返回 ((排序表[中 减 1] 加 排序表[中]) 除 2)\\n    否则：\\n        返回 排序表[中]\\n"}
注意：源码 字段是单一 JSON 字符串，其中换行必须写成 \\n（如同上例）；不要输出 JSON 以外的任何解释文字。"""


def _call_llm(需求, 候选, cfg):
    if not cfg.get('api_key'):
        return None, None
    url = cfg['base_url'].rstrip('/') + '/chat/completions'
    cands = '、'.join(c.get('名称', '') for c in (候选 or [])[:5]) or '（无）'
    user = '需求：%s\n已选候选（可能不对）：%s\n请生成一块能直接满足该需求的段言积木。' % (需求, cands)
    payload = {
        'model': cfg['model'],
        'messages': [
            {'role': 'system', 'content': _SYSTEM},
            {'role': 'user', 'content': user},
        ],
        'temperature': 0.2,
        'response_format': {'type': 'json_object'},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + cfg['api_key'],
                 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode('utf-8'))
        content = data['choices'][0]['message']['content']
        return json.loads(content), data.get('usage', {})
    except Exception as e:
        print('[兜底] LLM 调用失败，降级本地规则：%s' % e)
        return None, None


# ---------------------------------------------------------------------------
# 本地规则生成器（无 key 时也能端到端跑通）
# ---------------------------------------------------------------------------
def _本地规则生成(需求):
    q = 需求
    if '方差' in q:
        return _方差模板(总体=('样本' not in q))
    if '中位数' in q or '中位' in q:
        return _中位数模板()
    if '绝对值' in q or '取绝对' in q:
        return _绝对值模板()
    # v0.15：库外『数列/数论』意图的本地兜底（零 token，无需 LLM key）
    if '斐波那契' in q or 'fib' in q.lower():
        return _斐波那契模板()
    if '阶乘' in q or 'factorial' in q.lower():
        return _阶乘模板()
    if '素数' in q or '质数' in q:
        return _素数模板()
    if '累加和' in q or '高斯和' in q or '1到' in q or '1至' in q:
        return _累加和模板()
    # v0.26（3.2 兜底 demo）：十进制转二进制——库内无此能力，本地规则生成演示闭环
    if '二进制' in q or 'binary' in q.lower():
        return _二进制模板()
    return None


def _二进制模板():
    名 = '二进制'
    src = (
        '# 积木：二进制（生成领域，本地规则生成）\n'
        '# 契约：输入 [数 n] → 输出 列表（十进制转二进制的各位）\n'
        '从《列表工具》导入《反转列表》\n'
        '导出 转二进制\n'
        '段落 转二进制 接收 n：\n'
        '    如果 n 等于 0：\n'
        '        返回 [0]\n'
        '    设 表 为 []\n'
        '    当 n 大于 0：\n'
        '        表.追加(n 模 2)\n'
        '        设 n 为 整数(n 除 2)\n'
        '    返回 反转列表(表)\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '把十进制数转成二进制（各位列表）',
        '输入': [{'名': 'n', '类型': '数'}],
        '输出': {'类型': '列表[数]'},
        '稳定性': 'generated',
        '导出名': '转二进制',
        '源码': src,
    }


def _斐波那契模板():
    名 = '斐波那契'
    src = (
        '# 积木：斐波那契（数列生成，本地规则生成）\n'
        '# 契约：输入 [数 n] → 输出 列表（前 n 项）\n'
        '导出 斐波那契\n'
        '段落 斐波那契 接收 n：\n'
        '    如果 n 等于 1：\n'
        '        返回 [1]\n'
        '    设 表 为 [1, 1]\n'
        '    设 i 为 2\n'
        '    当 i 小于 n：\n'
        '        设 末 为 表[长度(表) 减 1]\n'
        '        设 次 为 表[长度(表) 减 2]\n'
        '        表.追加(末 加 次)\n'
        '        设 i 为 i 加 1\n'
        '    返回 表\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '生成斐波那契数列前 n 项',
        '输入': [{'名': 'n', '类型': '数'}],
        '输出': {'类型': '列表[数]'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def _阶乘模板():
    名 = '阶乘'
    src = (
        '# 积木：阶乘（数列生成，本地规则生成）\n'
        '# 契约：输入 [数 n] → 输出 数\n'
        '导出 阶乘\n'
        '段落 阶乘 接收 n：\n'
        '    设 结果 为 1\n'
        '    设 i 为 1\n'
        '    设 上限 为 n 加 1\n'
        '    当 i 小于 上限：\n'
        '        设 结果 为 结果 乘 i\n'
        '        设 i 为 i 加 1\n'
        '    返回 结果\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '计算 n 的阶乘（n!）',
        '输入': [{'名': 'n', '类型': '数'}],
        '输出': {'类型': '数'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def _素数模板():
    名 = '素数'
    src = (
        '# 积木：素数（数论，本地规则生成）\n'
        '# 契约：输入 [数 n] → 输出 数（1=素数，0=非素数）\n'
        '导出 素数\n'
        '段落 素数 接收 n：\n'
        '    如果 n 小于 2：\n'
        '        返回 0\n'
        '    设 i 为 2\n'
        '    当 i 小于 n：\n'
        '        如果 n 模 i 等于 0：\n'
        '            返回 0\n'
        '        设 i 为 i 加 1\n'
        '    返回 1\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '判断 n 是否为素数（1=是，0=否）',
        '输入': [{'名': 'n', '类型': '数'}],
        '输出': {'类型': '数'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def _累加和模板():
    名 = '累加和'
    src = (
        '# 积木：累加和（数论，本地规则生成）\n'
        '# 契约：输入 [数 n] → 输出 数（1+2+...+n）\n'
        '导出 累加和\n'
        '段落 累加和 接收 n：\n'
        '    返回 (n 乘 (n 加 1)) 除 2\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '计算 1+2+...+n 的累加和',
        '输入': [{'名': 'n', '类型': '数'}],
        '输出': {'类型': '数'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def _方差模板(总体=True):
    名 = '方差' if 总体 else '样本方差'
    分母 = '计数' if 总体 else '(计数 减 1)'
    src = (
        '# 积木：%s（数据领域，本地规则生成）\n'
        '# 契约：输入 [列表] → 输出 数（%s）\n'
        '导出 %s\n'
        '段落 %s 接收 序列：\n'
        '    设 计数 为 长度(序列)\n'
        '    设 和 为 0\n'
        '    设 i 为 0\n'
        '    当 i 小于 计数：\n'
        '        设 和 为 和 加 序列[i]\n'
        '        设 i 为 i 加 1\n'
        '    设 均值 为 和 除 计数\n'
        '    设 平方和 为 0\n'
        '    设 j 为 0\n'
        '    当 j 小于 计数：\n'
        '        设 差 为 序列[j] 减 均值\n'
        '        设 平方和 为 平方和 加 (差 乘 差)\n'
        '        设 j 为 j 加 1\n'
        '    返回 (平方和 除 %s)\n'
    ) % (名, 名, 名, 名, 分母)
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '计算一组数的%s' % 名,
        '输入': [{'名': '序列', '类型': '列表[数]'}],
        '输出': {'类型': '数'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def _中位数模板():
    名 = '中位数'
    src = (
        '# 积木：中位数（数据领域，本地规则生成）\n'
        '# 契约：输入 [列表] → 输出 数\n'
        '导出 中位数\n'
        '段落 中位数 接收 序列：\n'
        '    设 排序表 为 []\n'
        '    设 i 为 0\n'
        '    当 i 小于 长度(序列)：\n'
        '        排序表.追加(序列[i])\n'
        '        设 i 为 i 加 1\n'
        '    排序表.排序()\n'
        '    设 计数 为 长度(排序表)\n'
        '    设 中 为 整数(计数 除 2)\n'
        '    如果 计数 模 2 等于 0：\n'
        '        返回 ((排序表[中 减 1] 加 排序表[中]) 除 2)\n'
        '    否则：\n'
        '        返回 排序表[中]\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '计算一组数的中位数',
        '输入': [{'名': '序列', '类型': '列表[数]'}],
        '输出': {'类型': '数'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def _绝对值模板():
    名 = '绝对值'
    src = (
        '# 积木：绝对值（工具领域，本地规则生成）\n'
        '# 契约：输入 [数] → 输出 数\n'
        '导出 绝对值\n'
        '段落 绝对值 接收 数：\n'
        '    如果 数 小于 0：\n'
        '        返回 (0 减 数)\n'
        '    否则：\n'
        '        返回 数\n'
    )
    return {
        '名称': 名, '领域': '生成', '层级': 0,
        '描述': '取一个数的绝对值',
        '输入': [{'名': '数', '类型': '数'}],
        '输出': {'类型': '数'},
        '稳定性': 'generated',
        '导出名': 名,
        '源码': src,
    }


def generate_block(需求, 索引=None, 候选=None, 库根=None):
    """生成一块满足需求的新积木（dict）。返回 None 表示连本地规则都覆盖不了。"""
    blk, 用量 = _call_llm(需求, 候选, load_config())
    if blk is None:
        blk = _本地规则生成(需求)
    if blk is None:
        return None
    if 用量:
        blk['_用量'] = 用量  # token 成本（仅真实 LLM 有；本地规则为 None）
    # 补默认字段 + 兜底拼装最小可用源码
    blk.setdefault('领域', '生成')
    blk.setdefault('层级', 0)
    blk.setdefault('稳定性', 'generated')
    源 = (blk.get('源码') or '').strip()
    if '导出 ' not in 源:
        名 = blk.get('导出名') or blk.get('名称')
        源 = '# 自动生成\n导出 %s\n段落 %s 接收 序列：\n    返回 序列\n' % (名, 名)
        blk['源码'] = 源
    if not blk.get('导出名'):
        blk['导出名'] = blk.get('名称')
    return blk


def local_rule_block(需求):
    """仅用本地规则判断需求是否命中「库缺失但可模板生成」的能力（零 token，不调 LLM）。"""
    return _本地规则生成(需求)


def 注册(块, 库根=None):
    """把生成块写入 生成/<名称>.duan 并追加进 索引.json。返回 .duan 路径（相对库根）。"""
    库根 = 库根 or _HERE
    名 = 块.get('名称') or 块.get('导出名')
    路径 = '生成/%s.duan' % 名
    目录 = os.path.join(库根, '生成')
    os.makedirs(目录, exist_ok=True)

    idx_path = os.path.join(库根, '索引.json')
    idx = json.load(open(idx_path, encoding='utf-8'))

    # v0.28 防同名劫持：生成块与「库内既有非生成块」同名时，绝不改写原条目的 路径。
    # 否则原库块会被指向 生成/<名>.duan，一旦护栏回滚再 注销，整条原始条目就被删掉
    # （实测曾把 几何/两点距离 从索引里抹掉）。此处改名让位，原块保持不变。
    冲突 = next((b for b in idx['块'] if b.get('名称') == 名), None)
    if 冲突 is not None and not str(冲突.get('路径', '')).startswith('生成/'):
        名 = '%s_生成' % 名
        块['名称'] = 名
        路径 = '生成/%s.duan' % 名

    # 去重：已存在则仅更新文件与路径
    if any(b.get('名称') == 名 for b in idx['块']):
        with open(os.path.join(库根, 路径), 'w', encoding='utf-8') as f:
            f.write(块.get('源码', ''))
        for b in idx['块']:
            if b.get('名称') == 名:
                b['路径'] = 路径
    else:
        with open(os.path.join(库根, 路径), 'w', encoding='utf-8') as f:
            f.write(块.get('源码', ''))
        块['路径'] = 路径
        idx['块'].append(块)
    json.dump(idx, open(idx_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    return 路径


待审目录名 = '待审'


def 待审目录(库根=None):
    return os.path.join(库根 or _HERE, '生成', 待审目录名)


def 入待审(块, 库根=None, 原因=''):
    """把未过护栏的生成块写入 生成/待审/ 并记入 待审清单.jsonl（不污染 索引.json）。"""
    import datetime
    库根 = 库根 or _HERE
    名 = 块.get('名称') or 块.get('导出名')
    目录 = 待审目录(库根)
    os.makedirs(目录, exist_ok=True)
    路径 = os.path.join(目录, '%s.duan' % 名)
    with open(路径, 'w', encoding='utf-8') as f:
        f.write(块.get('源码', ''))
    清单 = os.path.join(目录, '待审清单.jsonl')
    row = {'名称': 名, '路径': '生成/%s/%s.duan' % (待审目录名, 名), '原因': 原因,
           '时间': datetime.datetime.now().isoformat(timespec='seconds'),
           '块': 块}
    with open(清单, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return 路径


def 注销(名称, 库根=None):
    """从 索引.json 移除生成块并删除其 .duan 文件（护栏不通过时回滚，避免坏块污染选块）。"""
    库根 = 库根 or _HERE
    idx_path = os.path.join(库根, '索引.json')
    idx = json.load(open(idx_path, encoding='utf-8'))
    原 = len(idx['块'])
    # v0.28：只注销「生成块」（路径在 生成/ 下）。护栏回滚时若名字恰与库内原有块相同，
    # 不能把原始库块一并删掉——那是不可逆的库损坏。
    idx['块'] = [b for b in idx['块']
                 if not (b.get('名称') == 名称
                         and str(b.get('路径', '')).startswith('生成/'))]
    if len(idx['块']) == 原:
        return False
    json.dump(idx, open(idx_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    文件 = os.path.join(库根, '生成', '%s.duan' % 名称)
    if os.path.isfile(文件):
        try:
            os.remove(文件)
        except Exception:
            pass  # 某些受限环境（沙箱/批量删除保护）禁止删除，索引已更新即为回滚生效
    return True


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description='LLM 兜底生成器（零 token 本地规则 / 配置 key 后调真实 LLM）')
    p.add_argument('需求', help='自然语言需求文本')
    p.add_argument('--注册', action='store_true', help='生成后写入 生成/ 并注册进索引')
    p.add_argument('--库根', default=_HERE)
    args = p.parse_args(argv)

    blk = generate_block(args.需求, 库根=args.库根)
    if not blk:
        print('无法生成（本地规则不覆盖，且未配置真实 LLM）：' + args.需求)
        return 1
    if args.注册:
        注册(blk, 库根=args.库根)
        print('已注册：%s（%s）' % (blk['名称'], blk.get('路径')))
    print(json.dumps(blk, ensure_ascii=False, indent=2,
                     default=lambda o: o if isinstance(o, str) else str(o)))
    return 0


if __name__ == '__main__':
    import argparse  # noqa: E402  (置于文件尾，避免顶层 import 顺序问题)
    raise SystemExit(_cli())
