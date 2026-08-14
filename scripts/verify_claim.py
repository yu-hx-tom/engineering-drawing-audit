# -*- coding: utf-8 -*-
"""断言校验器：在重绘参数基准里搜索某值，验证"重绘有无X"断言。

用法:
    py verify_claim.py <params.json> "37 3/16"

输出:
    FOUND     在 GLM 完整尺寸或原始 token 里找到
    NOT FOUND 两个基准都没找到 → 才允许写"重绘无X"结论

设计目的（强制关卡）:
    报告里每个"重绘无某尺寸 / 重绘只有X"的断言，必须先用本脚本搜索。
    NOT FOUND 是写"重绘无X"的唯一前提。禁止凭印象或凭视觉输出断言。
"""
import json
import os
import re
import sys


def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode())


def normalize(v):
    s = str(v)
    s = s.replace(' ', '').replace('"', '').replace('(', '').replace(')', '')
    s = s.replace('°', '').replace('Φ', '').replace('φ', '').replace('Δ', '')
    return s.lower()


def loose_match(query, candidate, strict=False):
    """匹配:
    strict=True: 归一化后完整相等 或 候选含查询(用于GLM完整尺寸, 查询=真实尺寸)
    strict=False: 仅用于原始token, 要求候选 token 含数字且长度>=2
    """
    q = normalize(query)
    c = normalize(candidate)
    if not q or not c:
        return False
    if strict:
        return q == c or q in c
    # 原始token宽松: 只匹配数字token, 避免单个1/3/6误报
    if len(q) < 2 and len(c) < 2:
        return False
    return q == c or q in c or c in q


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    params_path, query = sys.argv[1], sys.argv[2]

    with open(params_path, encoding='utf-8') as f:
        params = json.load(f)

    glm_dims = params.get('glm_dims', [])
    raw_tokens = params.get('raw_tokens', [])

    print(f'基准: {os.path.basename(params_path)}  (GLM尺寸 {len(glm_dims)} 条, 原始token {len(raw_tokens)} 条)')
    print(f'查询: {query!r}')
    print('-' * 55)

    hits = []
    # 1. GLM 完整尺寸通道（严格）: 查询必须是某尺寸的完整/子串
    qn = normalize(query)
    for d in glm_dims:
        if loose_match(query, d['value'], strict=True):
            hits.append(('GLM尺寸', d['value'], d.get('feature', '')))
    # 2. 原始 token 通道: 仅当 GLM 通道无命中时才看，避免宽松误报
    if not hits:
        for t in raw_tokens:
            tv = t['text']
            # 只匹配"数值型"token: 含多数字, 或含单位符号(",°,R,Φ)且长度>=2
            has_multi_digit = len(re.findall(r'\d', tv)) >= 2
            has_unit = any(sym in tv for sym in ('"', '°', 'R', 'Φ', 'φ', '/'))
            is_single_digit = tv.strip().isdigit() and len(tv.strip()) == 1
            if is_single_digit:
                continue
            if not (has_multi_digit or has_unit):
                continue
            if loose_match(query, tv, strict=False):
                hits.append(('原始token', tv, f'x{t["x0"]:.0f},y{t["y0"]:.0f}'))

    if hits:
        seen = set()
        print(f'FOUND ({len(hits)} 处):')
        for src, val, extra in hits:
            key = (src, val)
            if key in seen:
                continue
            seen.add(key)
            safe_print(f'  [{src}] {val!r}  {extra}')
    else:
        print('NOT FOUND —— 允许写"重绘无"结论')
        print('(同时查看 redraw-dims.md 总表人工复核)')
    sys.exit(0 if hits else 2)


if __name__ == '__main__':
    import os
    main()
