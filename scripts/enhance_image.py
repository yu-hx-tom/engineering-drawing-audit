#!/usr/bin/env python3
"""Enhance a cropped image before GLM vision reading.

在将局部裁剪图交给 GLM 识别前做预处理,提升低对比度/低清扫描件的可读性:
- 灰度化(可选,降低色彩干扰)
- 自动对比度 / 直方图均衡 / CLAHE / 对比度增强(按需选择)
- 插值放大到目标宽度(扫描源 crop 局部后分辨率不足时)

用法:
    python enhance_image.py <input.png> [--method autocontrast|equalize|clahe|contrast] \
        [--factor 2.0] [--target-width 1200] [--gray] [--out <path>]

输出: 默认 <input>_enhanced.png。增强后的图再交给 glm_drawing.py 识别。

背景(2026-08-12 round4 用户反馈):
- "先放大再截取局部以保持分辨率": 对 PDF 源 render_pdf.py --crop --dpi 600/800 已是
  高 DPI 渲染裁剪; 对 tif/jpg 扫描源, 源密度仅 200-300dpi, 小标注像素少,
  需插值放大 + 对比度增强后再 GLM 识别。
- "调整图像对比度帮助视觉识别": 低对比度扫描件(浅色标注、手写、反光)在增强后
  GLM 识别更稳, 减少误读/漏读。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def enhance(img, method: str, factor: float, target_width: int | None,
            gray: bool) -> "PIL.Image.Image":
    from PIL import Image, ImageEnhance, ImageOps

    if gray and img.mode not in ("L", "1"):
        img = img.convert("L")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if method == "autocontrast":
        img = ImageOps.autocontrast(img, cutoff=1)
    elif method == "equalize":
        img = ImageOps.equalize(img)
    elif method == "clahe":
        img = _clahe(img, clip=2.0, tile=8)
    elif method == "contrast":
        img = ImageEnhance.Contrast(img).enhance(factor)
        # 对比度增强后再自动对比度收敛一次, 防过曝
        img = ImageOps.autocontrast(img, cutoff=1)
    else:
        raise ValueError(f"未知增强方法: {method}")

    if target_width is not None:
        w, h = img.size
        if w < target_width:
            new_h = round(h * target_width / w)
            img = img.resize((target_width, new_h), Image.LANCZOS)

    return img


def _clahe(img, clip: float, tile: int) -> "PIL.Image.Image":
    """简易 CLAHE(基于 numpy 的局部直方图均衡), 对低对比度扫描件效果明显。"""
    import numpy as np
    from PIL import Image

    if img.mode != "L":
        img = img.convert("L")
    a = np.asarray(img, dtype=np.uint8)

    def _clahe_ch(ch):
        h, w = ch.shape
        tiles_y = max(1, h // (tile * 8))
        tiles_x = max(1, w // (tile * 8))
        out = np.zeros_like(ch)
        # 计算各 tile 的 CDF
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y0, y1 = ty * h // tiles_y, (ty + 1) * h // tiles_y
                x0, x1 = tx * w // tiles_x, (tx + 1) * w // tiles_x
                tile_px = ch[y0:y1, x0:x1]
                hist, _ = np.histogram(tile_px, bins=256, range=(0, 256))
                cdf = hist.cumsum()
                cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
                out[y0:y1, x0:x1] = cdf[tile_px]
        return out

    out = _clahe_ch(a)
    # 双线性插值平滑 tile 边界(简化, 不做精确插值)
    return Image.fromarray(out.astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="裁剪图 GLM 识别前预处理(放大/对比度增强)")
    ap.add_argument("input", help="输入图片")
    ap.add_argument("--method", default="autocontrast",
                    choices=["autocontrast", "equalize", "clahe", "contrast", "none"],
                    help="增强方法(默认 autocontrast)")
    ap.add_argument("--factor", type=float, default=2.0, help="contrast 方法的增强因子")
    ap.add_argument("--target-width", type=int, default=0,
                    help="插值放大到目标宽度(像素); 0=不放大")
    ap.add_argument("--gray", action="store_true", help="先转灰度")
    ap.add_argument("--out", default="", help="输出路径(默认 input_enhanced.png)")
    args = ap.parse_args()

    from PIL import Image

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"[ERROR] 输入不存在: {src}")
    img = Image.open(src)
    if args.method == "none":
        out_img = img
    else:
        out_img = enhance(img, args.method, args.factor,
                          args.target_width or None, args.gray)
    out_path = Path(args.out) if args.out else src.with_name(
        f"{src.stem}_enhanced{src.suffix or '.png'}")
    out_img.save(str(out_path))
    print(f"[OK] {src} -> {out_path} ({out_img.size[0]}x{out_img.size[1]}) "
          f"method={args.method} gray={args.gray} target_width={args.target_width}")


if __name__ == "__main__":
    main()
