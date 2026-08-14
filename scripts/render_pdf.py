#!/usr/bin/env python3
"""Render PDF pages to PNG (and crop/zoom local regions) for engineering-drawing audit.

纯本地渲染,无 API 费用。依赖:PyMuPDF(fitz)、Pillow。

用法:
    python render_pdf.py <input.pdf> <output_dir> [--dpi 300]
    python render_pdf.py <input.pdf> <output_dir> --crop <page> <x> <y> <w> <h> [--dpi 600]
    python render_pdf.py <input.pdf> <output_dir> --page 2,3 --dpi 300

坐标基于页面原始单位(点,72/inch),从页面左上角起。
--crop 输出放大后的局部裁剪图,便于审查手写/模糊/旋转区域。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def render_pages(pdf_path: Path, output_dir: Path, pages: list[int] | None,
                 dpi: int) -> list[Path]:
    """Render selected (1-based) pages to PNG at given DPI. Returns output paths."""
    import pymupdf  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    produced: list[Path] = []

    total = doc.page_count
    if pages is None:
        pages = list(range(1, total + 1))

    for pno in pages:
        if not 1 <= pno <= total:
            print(f"[WARN] 页码 {pno} 超出范围 1..{total},跳过")
            continue
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=matrix)
        out = output_dir / f"page-{pno}.png"
        pix.save(str(out))
        produced.append(out)
        print(f"[OK] page {pno}: {out} ({pix.width}x{pix.height}px, {dpi} DPI)")

    doc.close()
    return produced


def crop_region(pdf_path: Path, output_dir: Path, page_no: int,
                x: float, y: float, w: float, h: float, dpi: int) -> Path:
    """Crop a region on a page (points from top-left) and upscale to DPI. Returns path."""
    import pymupdf
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    if not 1 <= page_no <= doc.page_count:
        sys.exit(f"[ERROR] 页码 {page_no} 超出范围 1..{doc.page_count}")

    page = doc.load_page(page_no - 1)
    page_rect = page.rect  # 页面尺寸(点)
    # 边界检查:坐标单位为点(1/72 inch),原点在页面左上角
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        sys.exit(f"[ERROR] 裁剪坐标无效: x={x} y={y} w={w} h={h}(应 x,y>=0 且 w,h>0,单位:点)")
    if x + w > page_rect.width or y + h > page_rect.height:
        sys.exit(
            f"[ERROR] 裁剪区域超出页面 {page_no} 边界(页面 {page_rect.width:.0f}x{page_rect.height:.0f} 点,"
            f"左上角为原点)。请求: x={x} y={y} w={w} h={h} → 右下角 ({x+w:.0f},{y+h:.0f})\n"
            f"  提示: 先不带 --crop 渲染整页,再从 PNG 按比例换算坐标为点。"
        )

    zoom = dpi / 72.0
    clip = pymupdf.Rect(x, y, x + w, y + h)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    out = output_dir / f"crop-page{page_no}-x{x:.0f}-y{y:.0f}-{w:.0f}x{h:.0f}-dpi{dpi}.png"
    img.save(str(out))
    doc.close()
    print(f"[OK] crop page {page_no} region ({x:.1f},{y:.1f},{w:.1f}x{h:.1f}) → {out} "
          f"({img.size[0]}x{img.size[1]}px, {dpi} DPI)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="本地渲染 PDF 页面/局部为 PNG")
    ap.add_argument("input", help="输入 PDF 路径")
    ap.add_argument("output", help="输出目录")
    ap.add_argument("--dpi", type=int, default=300, help="渲染 DPI(默认 300)")
    ap.add_argument("--page", default=None, help="只渲染指定页,如 '2,3' (1-based)")
    ap.add_argument("--crop", nargs=5, metavar=("PAGE", "X", "Y", "W", "H"),
                    help="裁剪局部:页码 x y w h(页面单位,左上为原点)")
    args = ap.parse_args()

    pdf_path = Path(args.input)
    out_dir = Path(args.output)
    if not pdf_path.exists():
        sys.exit(f"[ERROR] 输入 PDF 不存在: {pdf_path}")

    if args.crop:
        page_no, x, y, w, h = (int(args.crop[0]), float(args.crop[1]),
                               float(args.crop[2]), float(args.crop[3]), float(args.crop[4]))
        crop_region(pdf_path, out_dir, page_no, x, y, w, h, args.dpi)
        return

    pages = None
    if args.page:
        pages = [int(p.strip()) for p in args.page.split(",") if p.strip()]
    render_pages(pdf_path, out_dir, pages, args.dpi)


if __name__ == "__main__":
    main()
