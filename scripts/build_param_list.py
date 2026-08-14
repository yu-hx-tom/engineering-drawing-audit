# -*- coding: utf-8 -*-
"""从重绘 PDF 文字层生成参数基准（权威）。

用法:
    py build_param_list.py <重绘图.pdf> <输出目录> [--glm-dims <glm_dims.txt>]

产出:
    <输出目录>/params.json   参数基准。含:
        - raw_tokens: 文字层原始 token（坐标+字号，机器读取，不可伪造）
        - glm_dims:   GLM 视觉读出的完整尺寸+公差（英制分数已正确合并）
    <输出目录>/redraw-dims.md  人工可读总表

设计目的（防止"重绘无X"凭印象断言）:
    报告中的"重绘侧"数据必须只引用本脚本产物。任何"重绘无某尺寸"的断言，
    都必须先用 verify_claim.py 在本脚本产出的 params.json 里搜索，输出 NOT FOUND 才能写。

英制分数说明:
    CAD 里 2 5/8" 常以分离 token ('2' '5' '8' '"') 存储，算法合并不可靠。
    因此完整尺寸以 GLM 视觉读数为准（GLM 能正确读英制分数），
    文字层原始 token 只作坐标/存在性佐证。两者必须同时引用。
"""
import fitz
import json
import os
import re
import sys


def extract_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for p in doc:
        d = p.get_text('dict')
        for b in d['blocks']:
            if b['type'] != 0:
                continue
            for l in b['lines']:
                for s in l['spans']:
                    t = s['text'].strip()
                    if not t:
                        continue
                    bb = s['bbox']
                    spans.append({
                        'text': t,
                        'x0': round(bb[0], 1), 'y0': round(bb[1], 1),
                        'x1': round(bb[2], 1), 'y1': round(bb[3], 1),
                        'size': round(s['size'], 1),
                    })
    return spans


def parse_glm_dims(txt_path):
    """解析 GLM 特征提示词输出，提取尺寸清单。

    支持两种格式:
      尺寸标注_13: 37 3/16" (控制特征: ...)
      37 3/16" | 物理特征 | 判断依据
    返回 [{value, feature}]
    """
    dims = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r'尺寸标注_\d+:\s*(.+?)(?:\s*\(控制特征:\s*(.*?)\))?\s*$', line)
            if m:
                dims.append({'value': m.group(1).strip(), 'feature': (m.group(2) or '').strip()})
                continue
            m2 = re.match(r'^(.+?)\s*\|\s*(.*?)(?:\s*\|.*)?$', line)
            if m2 and any(ch.isdigit() for ch in m2.group(1)):
                dims.append({'value': m2.group(1).strip(), 'feature': m2.group(2).strip()})
    return dims


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    pdf_path, outdir = args[0], args[1]
    glm_txt = None
    if '--glm-dims' in sys.argv:
        glm_txt = sys.argv[sys.argv.index('--glm-dims') + 1]
    os.makedirs(outdir, exist_ok=True)

    spans = extract_spans(pdf_path)
    glm_dims = parse_glm_dims(glm_txt) if glm_txt else []

    params = {
        'source': pdf_path,
        'span_count': len(spans),
        'glm_dim_count': len(glm_dims),
        'raw_tokens': sorted(spans, key=lambda s: (s['y0'], s['x0'])),
        'glm_dims': glm_dims,
    }
    json_path = os.path.join(outdir, 'params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=1)
    print(f'[OK] params.json: {len(spans)} 原始token, {len(glm_dims)} GLM完整尺寸')

    md_path = os.path.join(outdir, 'redraw-dims.md')
    lines = ['# 重绘参数基准（脚本生成）', '',
             f'- 源文件: `{os.path.basename(pdf_path)}`', '',
             '## GLM 视觉完整尺寸（英制分数已合并）', '']
    lines.append('| # | 尺寸 | 物理特征 |')
    lines.append('|---:|---|---|')
    for i, d in enumerate(glm_dims, 1):
        lines.append(f'| {i} | {d["value"]} | {d["feature"]} |')
    lines += ['', '## 文字层原始 token（坐标佐证）', '']
    lines.append('| 文本 | x | y | 字号 |')
    lines.append('|---|---:|---:|---:|')
    for s in sorted(spans, key=lambda s: (s['y0'], s['x0'])):
        lines.append(f'| `{s["text"]}` | {s["x0"]:.0f} | {s["y0"]:.0f} | {s["size"]} |')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[OK] redraw-dims.md: {md_path}')


if __name__ == '__main__':
    main()
