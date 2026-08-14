# -*- coding: utf-8 -*-
"""从重绘文字层 token 按坐标聚合建立"尺寸信息表"（唯一权威基准）。

用法:
    py build_dim_table.py <重绘图.pdf> <输出目录>

产出:
    <输出目录>/dim-table.json  尺寸信息表（尺寸+公差+位置+类型），完全由文字层+坐标机器生成
    <输出目录>/dim-table.md    人工可读尺寸信息表

核心思想（用户确立的架构）:
    文字层 token 是唯一权威。普通尺寸和公差在 CAD 里是分离 token，
    通过"字号 + 坐标邻接"聚合：
      - 大字号数字 token → 主值/独立尺寸（如 1730、818.4、独立 7.5）
      - 小字号 token（含 Φ/±/数字）→ 公差上标，按同行右侧邻接归属到主值（如 Φ6 → 1730）
    聚合结果 = 完整的"尺寸值 + 公差 + 位置"表。
    GLM 读重绘/读原图的读数，都作为候选去和此表核销。

聚合规则:
    1. 大字号(>=12)数字 token = 独立尺寸条目（主值），记录位置
    2. 小字号(<12)token = 公差上标，找"同行(y中心差<12)且x紧邻(在其x1右侧<10)"的主值，归属之
    3. 输出条目含: 主值、公差、完整值、x、y、字号、类型(直径/角度/半径/线性/粗糙度/公差上标)
"""
import fitz
import json
import os
import re
import sys


def euclid_dist(x1, y1, x2, y2):
    """欧几里得直线距离 √(Δx² + Δy²), 用于所有坐标最近邻匹配"""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def extract_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for p in doc:
        d = p.get_text('dict')
        page_h = p.rect.height
        for b in d['blocks']:
            if b['type'] != 0:
                continue
            for l in b['lines']:
                for s in l['spans']:
                    t = s['text']
                    if not t.strip():
                        continue
                    bb = s['bbox']
                    spans.append({
                        'text': t,   # 保留原始文本(含前后空格, 供分数聚合识别分子/分母)
                        'x0': round(bb[0], 1), 'y0': round(bb[1], 1),
                        'x1': round(bb[2], 1), 'y1': round(bb[3], 1),
                        'size': round(s['size'], 1),
                        'page_h': page_h,
                    })
    return spans


def merge_fractions(keep):
    """英制分数聚合(2026-08-12 从 860404-TE001 提炼): 把 whole+num+den 碎 token 合并成
    'W N/D' 或 'N/D' 复合 span。

    背景: CAD 出图把分数尺寸拆成独立 token——整数部分、分子、分母、英寸符" 均为独立 span。
    旧逻辑把它们当多个独立主值, 尺寸表被碎片化(860404: 164 条碎片, 24 7/8 -> '24'+'7'+'8',
    3 5/8 -> '3'+'5'+'8'), 无法用尺寸表反向核销英制分数。修复后 83 条, 54 个分数全部合并。
    分母/分子 bbox 常互相重叠(整条 ' 7' 的 x1 被相邻字形撑大), 所以判同行右侧邻接必须用
    分子 x0 而非 x1。分母与分子两种布局(同图并存):
      R. 同行右侧: 分子与分母同一行, 分母在分子右 0-18px('25 5/8' 的 ' 5' 与 '8');
      B. 正下方: 分子在上、分母在下同 x('3 5/8' 的 '5' 与 '8', '1/16' 的 '1' 与 '16')。
    整数 W 三种来源: 合并型 'W N' 内部、后置空格型 'W ' 内部、或裸数字在分数正下方
    (dy 4-16px)。
    识别模式:
      A. 合并型 'W N' (如 '4 3','8 7','6 3','3 5','4 5'): W=整数, N=分子;
      B. 前置空格型 ' N' (如 ' 5',' 7',' 3',' 1'): N=分子, 整数 W 在其正下方;
      C. 后置空格型 'W ' (如 '3 ','5 ','7 ','1 '): W=整数, 分子 N 在其同行右侧;
      D. 裸数字分子(如 '1/16' 的 '1', '5/8' 的 '5'): 找分母与(可选的)整数。
    关键护栏:
      - 英寸符确认: 分母必须命中 '\\"' 英寸符(允许带前后空格)才合并——图框坐标格数字
        (无英寸符)不判分数, 防误配(860404 顶边 1,5,4,3,4 曾被抓成 4 1/5)。
      - 已消费 span 不得复用(防链式错配: 3/4 抢走 ±1/16 的分子 1, 导致 1/16 错配成 16/5)。
      - 整数搜索只用"正下方 dy 4-16px"(实测正确整数都在 6.9-8.1px); 禁止"同行左侧"——
        会抓到几十到几百 px 外的无关数字。
      - 分子与分母必须互相找到才合并。
    """
    is_num = lambda t: bool(re.fullmatch(r'\d+', t))
    is_merged = lambda t: bool(re.fullmatch(r'\d+\s+\d+', t))
    is_padL = lambda t: bool(re.fullmatch(r'\s+\d+\s*', t))
    is_padR = lambda t: bool(re.fullmatch(r'\d+\s+', t))
    inch_marks = [s for s in keep if s['text'].strip() == '"']

    def xc(s):
        return (s['x0'] + s['x1']) / 2

    def yc(s):
        return (s['y0'] + s['y1']) / 2

    def near_inch(den):
        """分数组必须有英寸符确认: 分母右侧 -20..50px 且 y 差 <=25。
        无英寸符的数字(图框坐标格、普通两数)不判分数。"""
        for im in inch_marks:
            if -20 <= im['x0'] - den['x0'] <= 50 and abs(yc(im) - yc(den)) <= 25:
                return True
        return False

    def find_den(num, exclude=set()):
        """找分母: 同行右侧(R)优先, 正下方同 x(B)次之。用 num.x0/num cx, 不用 x1。
        必须命中英寸符确认; 排除 num 自身与已消费 span。"""
        best_r, best_r_d = None, 1e9
        best_b, best_b_d = None, 1e9
        nc = xc(num)
        for o in keep:
            if id(o) in exclude or not is_num(o['text']):
                continue
            # R: 同行右侧 (|dyc|<=6, 分母左缘落在分子起始右侧 0-18px)。
            # 窗口不可太大: 860404 的 '5'(Ø3 5/8 分子) 右 28px 处是 '1/16' 公差分子,
            # 中间隔着 ± 符号, 属不同组; 真实同行分数对 gap 在 1-13px。
            if abs(yc(o) - yc(num)) <= 6 and 0 <= o['x0'] - num['x0'] <= 18:
                d = o['x0'] - num['x0']
                if d < best_r_d:
                    best_r_d, best_r = d, o
                continue
            # B: 正下方同 x (分母 cx 与分子 cx 对齐, 分母在分子下, |dx| 用绝对值)
            if 4 <= o['y0'] - num['y0'] <= 26 and abs(xc(o) - nc) <= 14:
                if abs(xc(o) - nc) < best_b_d:
                    best_b_d, best_b = abs(xc(o) - nc), o
        for cand in (best_r, best_b):
            if cand is not None and near_inch(cand):
                return cand, 'right' if cand is best_r else 'below'
        return None, None

    def find_whole(num, exclude=set()):
        """为分子找整数 W: 正下方 dy 4-16px(所有实测正确合并的整数都在 6.9-8.1px)。
        860404 教训: '同行左侧'会抓到几十到几百 px 外的无关数字(如 3/4 抓走 ±1/16 的
        分子 1, 导致 1/16 失去分母错配成 16/5); 左侧整数场景已由 merged/padR 型覆盖。"""
        best, best_dy = None, 1e9
        for o in keep:
            if id(o) in exclude or not (is_num(o['text']) or is_padR(o['text'])):
                continue
            dy = o['y0'] - num['y0']
            # 实测正确整数的 dy 都在 6.9-8.1px; 放宽到 [4,16] 排除远距离(±1/16 的 5/8
            # 起始在 dy 22.9px 处, 不是整数)
            if 4 <= dy <= 16 and num['x0'] - 20 <= o['x0'] <= num['x0'] + 30:
                if dy < best_dy:
                    best_dy, best = dy, o
        return best

    used = set()
    composites = []
    order = sorted(keep, key=lambda s: s['y0'])

    def mk(whole, num, den):
        t = "%s %s/%s" % (whole['text'], num['text'], den['text']) if whole else "%s/%s" % (num['text'], den['text'])
        x0 = min(s['x0'] for s in (whole, num, den) if s)
        y0 = min(s['y0'] for s in (whole, num, den) if s)
        x1 = max(s['x1'] for s in (whole, num, den) if s)
        size = max(s['size'] for s in (whole, num, den) if s)
        return {'text': t, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y0 + size,
                'size': size, 'page_h': keep[0].get('page_h', 842), '_frac': True}

    def merge_group(num_span, whole_span, den_span):
        used.add(id(num_span))
        used.add(id(den_span))
        if whole_span is not None:
            used.add(id(whole_span))
        composites.append(mk(whole_span, num_span, den_span))

    # (a) merged 'W N': W 和 N 在一个 span 内
    for s in order:
        if id(s) in used or not is_merged(s['text']):
            continue
        W, N = s['text'].split()
        total = len(W) + len(N)
        wfrac = len(W) / float(total)
        num = {'text': N, 'x0': s['x0'] + (s['x1'] - s['x0']) * wfrac, 'y0': s['y0'],
               'x1': s['x1'], 'y1': s['y1'], 'size': s['size']}
        whole = {'text': W, 'x0': s['x0'], 'y0': s['y0'],
                 'x1': s['x0'] + (s['x1'] - s['x0']) * wfrac, 'y1': s['y1'], 'size': s['size']}
        den, mode = find_den(num, exclude=used | {id(s)})
        if den is None:
            den, mode = find_den(s, exclude=used | {id(s)})
        if den is not None:
            used.add(id(s))  # 合并型原 span 本身也要标记已消费
            merge_group(num, whole, den)
            continue
        # 未找到分母: 保留原样(可能是真实"两数"标注)

    # (b) padR 'W ': 整数明确, 分子在同行右侧
    for s in order:
        if id(s) in used or not is_padR(s['text']):
            continue
        W = s['text'].strip()
        whole = {'text': W, 'x0': s['x0'], 'y0': s['y0'],
                 'x1': s['x0'] + (s['x1'] - s['x0']) * 0.6, 'y1': s['y1'], 'size': s['size']}
        num = None
        for o in keep:
            if id(o) in used or not is_num(o['text']):
                continue
            if abs(yc(o) - yc(s)) <= 6 and -2 <= o['x0'] - s['x1'] <= 14:
                num = o
                break
        if num is None:
            continue
        den, mode = find_den(num, exclude=used | {id(num)})
        if den is None:
            continue
        used.add(id(s))  # 原 padR span 标记已消费(whole 是合成的, 不占原 id)
        merge_group(num, whole, den)

    # (c) padL ' N': 分子明确, 整数在下方, 分母同行右侧或下方
    for s in order:
        if id(s) in used or not is_padL(s['text']):
            continue
        N = s['text'].strip()
        num = {'text': N, 'x0': s['x0'], 'y0': s['y0'], 'x1': s['x1'], 'y1': s['y1'], 'size': s['size']}
        den, mode = find_den(num, exclude=used | {id(s)})
        if den is None:
            continue
        whole = find_whole(num, exclude=used | {id(s), id(den)})
        used.add(id(s))  # 原 padL span 标记已消费
        merge_group(num, whole, den)

    # (d) 裸数字分子: 如 '1/16','5/8','1/8','3/4'(分子无空白标记, 单独出现)
    for s in order:
        if id(s) in used or not is_num(s['text']):
            continue
        den, mode = find_den(s, exclude=used | {id(s)})
        if den is None:
            continue
        whole = find_whole(s, exclude=used | {id(s), id(den)})
        merge_group(s, whole, den)

    out = []
    for s in keep:
        if id(s) in used:
            continue
        out.append(s)
    for c in composites:
        out.append(c)
    return out, composites


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pdf_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    spans = extract_spans(pdf_path)

    # 过滤: 排除图框角标(y<20)、页面底部标题栏(y 超过页高-100)、单字符字母(但保留+/-公差符号和单字符数字)、图号
    skip = set('ABCDEFGH')
    skip_text = {'780674', '780673'}  # 图号
    # 自动跳过图号: 文件名前导数字串(如 780672-环形衬板...pdf → 780672)不进入尺寸表
    fn_num = re.match(r'(\d{4,})', os.path.basename(pdf_path))
    if fn_num:
        skip_text.add(fn_num.group(1))
    tol_symbols = {'+', '-', '±'}  # 单字符公差符号

    def is_cjk(t):
        """含 CJK 表意文字/全角字符的 span 是标签或标题栏文字, 不是尺寸值(禁止进入尺寸表)"""
        return any('一' <= ch <= '鿿' or      # CJK 统一表意文字
                   '　' <= ch <= '〿' or      # CJK 标点
                   '＀' <= ch <= '￯'         # 全角字符
                   for ch in t)

    def keepable(s):
        tv = s['text'].strip()
        if tv in skip or tv in skip_text:
            return False
        if len(tv) > 1:
            return True
        return tv in tol_symbols or tv.isdigit() or tv == '"'  # 英寸符供分数聚合确认
    # 底部标题栏区域检测(2026-08-12 从 780674 提炼):
    # 旧实现是全局 y 截止(页高-100), 但 780674 的 D 俯视图角度标注(22.5°/45°)落在
    # 左下角 y=751/782 > 页高-100, 全局截止会把真实视图标注当"标题栏"丢弃。
    # 所有本批图纸(CAD 出图)的标题栏都在右下角: 底部 CJK 标签(标记/处数/绘图/图号…)
    # 的 x0 全部 >= 684, 而视图标注在 x<650。因此把底部截止限定在标题栏 x 区间:
    #   只对 "y0 > 页高-100 且 x0 >= 标题栏左缘-20" 的 span 视为标题栏并丢弃。
    # 标题栏左缘 = 底部(CJK 标签)区最靠左的 x0, 动态计算, 不写死。
    tb_cjk = [s for s in spans if is_cjk(s['text']) and s['y0'] > s.get('page_h', 842) - 180]
    tb_left = (min(s['x0'] for s in tb_cjk) - 20) if tb_cjk else 0
    # 图框边界坐标格数字(1..8, y≈页高-12, 全宽单字符数字)也要丢弃, 非尺寸值
    page_h_ref = spans[0].get('page_h', 842) if spans else 842
    border_top = page_h_ref - 25

    # 底部截止随页面高度自适应(页高-100 大致 = 标题栏起始行):
    # 780672 案例教训: 固定 y0<=660 会把俯视图下部的间距尺寸(254/190/125.2)
    # 及其公差(±5')整块当"标题栏"丢掉。不同图纸标题栏高度不同, 必须相对页高。
    def in_titleblock(s):
        ph = s.get('page_h', 842)
        if s['y0'] > ph - 100 and s['x0'] >= tb_left:
            return True
        # 图框边界坐标格: 页底最下一行(页高-25 以下)的单字符数字
        if s['y0'] > border_top and len(s['text']) == 1 and s['text'].isdigit():
            return True
        return False

    keep = [s for s in spans if 20 <= s['y0']
            and keepable(s) and not is_cjk(s['text']) and not in_titleblock(s)]

    # 英制分数聚合(2026-08-12 从 860404 提炼): 合并 whole+num+den 碎 token 为 'W N/D'。
    # 用原始文本(保留前后空格识别分子/分母); 之后一律去空白, 避免主值带尾随空格。
    keep, merged = merge_fractions(keep)
    for s in keep:
        s['text'] = s['text'].strip()

    # 主值/公差片段分类(2026-08-12 从 780673 提炼):
    # 旧实现用固定字号阈值(主值>=12, 公差<12)划分, 但不同图纸字号体系差异大。
    # 780673 尺寸文字 10.3-11.7, 而粗糙度/形位公差/标题栏文字 12.1-16, 角色被倒置,
    # 导致直径链公差(±10/±5/±6/+7-5)找不到主值而整串丢失、6.3/12.5 被当主值。
    # 修正: 不依赖字号绝对值。公差片段 = ①公差符号(±/+/-);
    #       ②距公差符号 ≤20px 的含数字 span, 且 45px 内存在比它大 ≥1px 的含数字 span
    #       (CAD 公差字必小于其主值字, 用字号余量区分"紧邻主值"(如 356 与 437.6
    #        只差 0.4px, 都是主值)与"公差片段"(如 0.25°(9.5) vs 30°(12.7)))。
    #       其余含数字 span 一律当主值候选(尺寸/半径/角度/粗糙度/GD&T 值),
    #       半径/角度/参考尺寸因此不再被丢弃。
    def is_tol_symbol(t):
        return '±' in t or t in ('+', '-')

    # 比例/剖视视图标签(1 : 9、A-A)不是尺寸, 不得参与片段判定/作主值/作归属目标
    def is_view_label(t):
        return bool(re.fullmatch(r'1\s*:\s*\d+', t)) or t in ('A-A', 'B-B', 'C-C', 'D-D')

    sym_spans = [s for s in keep if is_tol_symbol(s['text'])]

    # 距公差符号 ≤20px 的含数字 span = 片段候选
    cand = []
    for s in keep:
        if not re.search(r'\d', s['text']) or is_view_label(s['text']):
            continue
        scx = (s['x0'] + s['x1']) / 2
        scy = (s['y0'] + s['y1']) / 2
        for sym in sym_spans:
            sc = ((sym['x0'] + sym['x1']) / 2, (sym['y0'] + sym['y1']) / 2)
            if euclid_dist(scx, scy, *sc) <= 20:
                cand.append(s)
                break

    frag_ids = {id(s) for s in sym_spans}
    for s in cand:
        scx = (s['x0'] + s['x1']) / 2
        scy = (s['y0'] + s['y1']) / 2
        for t in keep:
            if id(t) == id(s) or not re.search(r'\d', t['text']) or is_view_label(t['text']):
                continue
            tcx = (t['x0'] + t['x1']) / 2
            tcy = (t['y0'] + t['y1']) / 2
            if euclid_dist(scx, scy, tcx, tcy) > 45:
                continue
            if t['size'] >= s['size'] + 1.0:
                frag_ids.add(id(s))
                break

    frags = [s for s in keep if id(s) in frag_ids]
    mains = [s for s in keep
             if re.search(r'\d', s['text']) and not is_view_label(s['text']) and id(s) not in frag_ids]

    entries = []

    # 每个主值默认独立成条目
    for m in mains:
        txt = m['text']
        # 跳过比例/视图标签
        if re.fullmatch(r'1\s*:\s*\d+', txt) or txt in ('A-A', 'B-B', 'C-C', 'D-D'):
            continue
        entries.append({
            'main': txt,
            'tol': '',
            'tol_tokens': [],
            'full': txt,
            'x': m['x0'], 'y': m['y0'],
            'size': m['size'],
        })

    # 数量前缀组合: "2-" (N-形式) 与相邻大字号数字组合成 "2-∅40" (N个直径X的孔)
    qty_prefix = [e for e in entries if re.fullmatch(r'\d+-', e['main'])]
    for q in qty_prefix:
        best = None
        best_d = 1e9
        for e in entries:
            if e is q or not re.fullmatch(r'\d+(?:\.\d+)?', e['main']):
                continue
            d = euclid_dist(e['x'], e['y'], q['x'], q['y'])
            if d > 60:
                continue  # 直线距离阈值
            if d < best_d:
                best_d = d
                best = e
        if best is not None:
            q['main'] = q['main'] + '∅' + best['main']
            q['full'] = q['main']
            entries.remove(best)

    # 公差片段归属到最近主值。两级判定:
    #   ① 首选"同行右侧邻接": 片段 x0 落在主值右缘附近(-10 ~ +20)且 y 中心接近(≤15)。
    #      CAD 公差的标准布局是公差紧跟所修饰尺寸值右侧同一行(如 `1070 ±7.5`)。
    #      780674 教训: 纯欧氏最近主值会把右下方/右上方的别的主值判为"更近"而抢走公差
    #      (±7.5(x899,y187) 欧氏最近是 R10 TYP(x904,y208), 实际属于正左侧同行的 1070)。
    #   ② 无同行邻接主值时, 回退到欧氏最近主值(阈值45)——
    #      覆盖 780673 直径链场景(±10/±5/±6 各距自己主值右缘, 但仍可用欧氏兜底)。
    # 780673 教训: 直径链上 ±10/±5/±6 纵向堆叠, 旧的"分组成串再归属"会把
    # 三个公差合并成一串(1796±5±6)或找不到主值而整串丢失。片段级独立归属
    # + 同行邻接优先可同时避免"串合并"与"对角线抢归属"两类错误。
    assignments = {}  # id(main) -> [frag, ...]
    for f in frags:
        fcx = (f['x0'] + f['x1']) / 2
        fcy = (f['y0'] + f['y1']) / 2
        best = None        # 欧氏最近主值(兜底)
        best_d = 1e9
        adj = None         # 同行右侧邻接主值(优先)
        adj_d = 1e9
        for m in mains:
            mcx = (m['x0'] + m['x1']) / 2
            mcy = (m['y0'] + m['y1']) / 2
            d = euclid_dist(fcx, fcy, mcx, mcy)
            if d > 45:
                continue
            if abs(fcy - mcy) <= 15 and -10 <= f['x0'] - m['x1'] <= 20:
                if d < adj_d:
                    adj_d = d
                    adj = m
            if d < best_d:
                best_d = d
                best = m
        target = adj if adj is not None else best
        if target is not None:
            assignments.setdefault(id(target), []).append(f)

    # 同一主值的片段组装公差串: 符号与最近数字配对成偏差项, +项在前 -项在后
    def assemble_tolerance(frags_list):
        signs = [g for g in frags_list if g['text'] in ('+', '-', '±')]
        digits = [g for g in frags_list if g['text'] not in ('+', '-', '±') and re.search(r'\d', g['text'])]
        used_digit = set()
        tol_parts = []
        for sg in signs:
            best = None
            best_d = 1e9
            for di, dg in enumerate(digits):
                if di in used_digit:
                    continue
                dist = euclid_dist(sg['x0'], sg['y0'], dg['x0'], dg['y0'])
                if dist < best_d:
                    best_d = dist
                    best = di
            if best is not None and best_d < 15:  # 直线距离阈值
                used_digit.add(best)
                tol_parts.append((sg['text'], digits[best]['text']))
            else:
                tol_parts.append((sg['text'], ''))
        # 未配对的数字(如 Φ7.5 的 7.5 已在符号串里, 或 ±10 单 span) 加进去
        for di, dg in enumerate(digits):
            if di not in used_digit:
                tol_parts.append(('', dg['text']))
        # 拼接: +项(含+号)在前, -项在后, 无符号的Φ/数字最后
        def pk(it):
            if '+' in it[0]:
                return 0
            if '-' in it[0]:
                return 2
            return 1
        tol_parts.sort(key=pk)
        return ''.join(s + d for s, d in tol_parts)

    for m in mains:
        fa = assignments.get(id(m), [])
        if not fa:
            continue
        tols = assemble_tolerance(fa)
        for e in entries:
            if e['main'] == m['text'] and e['x'] == m['x0'] and e['y'] == m['y0']:
                e['tol'] = tols
                e['tol_tokens'] = [{'text': g['text'], 'x': g['x0'], 'y': g['y0']} for g in fa]
                e['full'] = e['main'] + e['tol']
                break

    # 未归属的独立纯数字片段补为独立尺寸条目
    existing_nums = {e['main'] for e in entries}
    for s in frags:
        st = s['text']
        if not re.fullmatch(r'\d+(?:\.\d+)?', st):
            continue  # 只处理纯数字
        if any(c in st for c in ('+', '-', '±', 'Φ', 'φ')):
            continue
        # 避免重复(主值已含)
        if st in existing_nums:
            continue
        # 已作为公差归属的片段不重复成条目。
        # 必须精确匹配 token 文本, 禁止子串匹配:
        # "15" 不得匹配公差 token "15'", "40" 不得匹配坐标 "402.3"(780673 曾因此丢掉 40/15/6)
        if any(any(t['text'] == st for t in e.get('tol_tokens', [])) for e in entries):
            continue
        entries.append({
            'main': st, 'tol': '', 'tol_tokens': [],
            'full': st, 'x': s['x0'], 'y': s['y0'], 'size': s['size'],
        })
        existing_nums.add(st)

    # 输出
    dim_table = {'source': os.path.basename(pdf_path), 'span_count': len(spans), 'entries': entries}
    json_path = os.path.join(outdir, 'dim-table.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dim_table, f, ensure_ascii=False, indent=1)

    md_lines = ['# 尺寸信息表（文字层坐标聚合）', '',
                f'- 源文件: `{os.path.basename(pdf_path)}`', '',
                '| # | 尺寸 | 公差 | 完整值 | x | y | 字号 |',
                '|---:|---|---|---|---:|---:|---:|']
    for i, e in enumerate(sorted(entries, key=lambda e: (e['y'], e['x'])), 1):
        md_lines.append(f'| {i} | {e["main"]} | {e["tol"] or "-"} | {e["full"]} | {e["x"]:.0f} | {e["y"]:.0f} | {e["size"]} |')
    md_path = os.path.join(outdir, 'dim-table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f'[OK] dim-table.json: {len(entries)} 条尺寸')
    print(f'[OK] dim-table.md: {md_path}')


if __name__ == '__main__':
    main()
