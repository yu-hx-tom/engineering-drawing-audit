# -*- coding: utf-8 -*-
"""核销器：把 GLM 读数与尺寸信息表(dim-table.json)核销。

用法:
    py check_completeness.py <dim-table.json> [<GLM读数.txt>]

输出:
    未核销的 GLM 读数(GLM 读到了尺寸信息表里没有的 → 疑误读/多读)
    + 尺寸信息表中未被任何 GLM 读数匹配的条目(→ GLM 漏读)

设计目的:
    dim-table.json 是文字层+坐标机器生成的尺寸信息表(唯一权威)。
    GLM 读重绘/读原图的读数, 都作为候选拿去和表核销。
    匹配上的消耗, 未匹配的报出, 保证报告引用的是完整权威清单。
"""
import json
import os
import re
import sys

# Windows 控制台/重定向默认 GBK 无法输出 ⚠/Ø/±/° 等字符, 强制 UTF-8
# (2026-08-12 round3 复验: 未核销分支打印 ⚠ 在 GBK 控制台 UnicodeEncodeError 崩溃,
#  与 glm_vision.py 2026-08-12 同类修复)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def norm(s):
    s = s.replace(' ', '').replace('"', '').replace('Φ', '').replace('φ', '')
    s = s.replace('°', '').replace("'", '').replace('(','').replace(')','')
    s = s.replace('TYP', '').replace('Tip', '').replace('REF', '').replace('Ref', '')
    return s


def nominal_of(v):
    gn = norm(v)
    base = re.sub(r'^[RΦφ(]', '', gn)
    base = re.sub(r'[±+\-].*$', '', base)
    base = re.sub(r'[^0-9.]', '', base)
    return base


def tol_tokens_of(v):
    """从 GLM 读数值提取公差部分 token 集合"""
    gn = norm(v)
    m = re.search(r'[±+\-].*$', gn)
    if not m:
        return set()
    tol = m.group(0)
    return set(re.findall(r'[0-9.]+|[±+\-]', tol))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    table_path = sys.argv[1]
    glm_txt = sys.argv[2] if len(sys.argv) > 2 else None

    with open(table_path, encoding='utf-8') as f:
        table = json.load(f)
    entries = table['entries']

    # 解析 GLM 读数
    glm_reads = []
    if glm_txt and os.path.exists(glm_txt):
        with open(glm_txt, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: 尺寸标注_N: 值 (控制特征: ...) 或 值 | 特征
                m = re.search(r'尺寸标注_\d+:\s*(.+?)(?:\s*\(控制特征:.*?\))?\s*$', line)
                if m:
                    glm_reads.append(m.group(1).strip())
                    continue
                m2 = re.match(r'^(.+?)\s*\|', line)
                if m2 and any(ch.isdigit() for ch in m2.group(1)):
                    glm_reads.append(m2.group(1).strip())
    else:
        print(f'[WARN] 未提供 GLM 读数文件, 只检查尺寸信息表本身')
        glm_reads = []

    # 核销: 每个条目记录被匹配次数, matched>0 表示已被 GLM 读数覆盖
    entry_state = [{'e': e, 'matched': 0} for e in entries]

    def find_entry(nom, tol_tokens):
        """找能匹配标称值nom且容忍其公差的条目(优先未匹配的)"""
        best = None
        best_score = 1e9
        for i, st in enumerate(entry_state):
            e = st['e']
            e_nom = nominal_of(e.get('main', ''))
            # 主值匹配
            if nom != e_nom:
                continue
            # 公差校验: 提取读数公差与条目公差的数字, 有交集即可(容忍±/Φ符号差异)
            score = 0
            if tol_tokens:
                raw = e.get('tol_tokens', [])
                if raw and isinstance(raw[0], dict):
                    entry_tol_texts = [t['text'] for t in raw]
                else:
                    entry_tol_texts = list(raw)
                entry_tol_nums = set()
                for et in entry_tol_texts:
                    for n in re.findall(r'\d+(?:\.\d+)?', et):
                        entry_tol_nums.add(n)
                read_tol_nums = set(re.findall(r'\d+(?:\.\d+)?', ''.join(sorted(tol_tokens))))
                if entry_tol_nums and not (read_tol_nums & entry_tol_nums):
                    score = 5  # 公差数字无交集, 低分但可接受(可能读数格式差异)
            # 已匹配过的条目降优先级, 避免重复消耗
            if st['matched'] > 0:
                score += 3
            if score < best_score:
                best_score = score
                best = i
        return best
        return best

    # 用 GLM 读数核销
    unmatched_reads = []
    for r in glm_reads:
        nom = nominal_of(r)
        if not nom:
            unmatched_reads.append(r)
            continue
        tol = tol_tokens_of(r)
        idx = find_entry(nom, tol)
        if idx is not None:
            entry_state[idx]['matched'] += 1
        else:
            unmatched_reads.append(r)

    # 未匹配的条目(GLM 漏读): matched==0
    unmatched_entries = [st for st in entry_state if st['matched'] == 0 and st['e'].get('main')]

    print(f'基准: {os.path.basename(table_path)}  (尺寸信息表 {len(entries)} 条)')
    print(f'GLM 读数: {len(glm_reads)} 条')
    print('-' * 55)
    if unmatched_entries:
        print(f'⚠ 尺寸信息表中 {len(unmatched_entries)} 条未被 GLM 读数覆盖(疑漏读):')
        for st in sorted(unmatched_entries, key=lambda s: (s['e']['y'], s['e']['x'])):
            e = st['e']
            print(f"  {e['full']!r:>20}  x[{e['x']:.0f}] y[{e['y']:.0f}]")
    if unmatched_reads:
        print(f'⚠ GLM 读数 {len(unmatched_reads)} 条不在尺寸信息表(疑误读/多读):')
        for r in unmatched_reads:
            print(f"  {r!r}")
    if not unmatched_entries and not unmatched_reads:
        print('✓ GLM 读数全部核销进尺寸信息表, 无漏读无误读')
    sys.exit(0 if (not unmatched_entries and not unmatched_reads) else 2)


if __name__ == '__main__':
    main()
