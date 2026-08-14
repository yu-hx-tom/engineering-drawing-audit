#!/usr/bin/env python3
"""批量并行 GLM 视觉识别(工程图纸)。

将多张裁剪图一次性并行提交给 GLM-4.6V,大幅压缩"裁剪→识别"环节的墙钟时间。
工程图专用提示词(engineering-drawing.txt)强制加载, 与 glm_drawing.py 一致。

用法:
    python batch_glm.py <img1> <img2> ... [--workers N] [--extra "补充指令"]
    python batch_glm.py <dir> --glob "*.png" [--workers N]

输出: 每张图一行 JSON {img, success, elapsed_s, desc_len, desc_preview}
      (并行数默认 = 4; GLM API 并发 round4 已实测 4 路可行)
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "engineering-drawing.txt"
VISUALS_SCRIPTS = SCRIPTS_DIR


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="批量并行 GLM 视觉识别")
    ap.add_argument("inputs", nargs="+", help="图片路径或目录")
    ap.add_argument("--glob", default="*.png", help="目录模式")
    ap.add_argument("--workers", type=int, default=4, help="并行数(默认4)")
    ap.add_argument("--extra", default="", help="追加指令")
    ap.add_argument("--out-dir", default="", help="保存每张图完整 desc 到 <dir>/<img>.desc.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    images = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            images.extend(sorted(p.glob(args.glob)))
        else:
            images.append(p)
    images = [i for i in images if i.exists()]
    if not images:
        print("[ERROR] 无有效图片")
        sys.exit(1)

    # 提示词
    base = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = (base + "\n\n" + args.extra).strip() if args.extra else base

    sys.path.insert(0, str(VISUALS_SCRIPTS))
    from config import load_config
    from glm_vision import call_glm_vision

    config = load_config()

    def one(img: Path):
        t0 = time.time()
        r = call_glm_vision(str(img), config, prompt)
        dt = round(time.time() - t0, 1)
        desc = r.get("description", "") if r.get("success") else r.get("error", "")
        # 保存完整 desc 到 out-dir
        if out_dir:
            try:
                (out_dir / f"{img.stem}.desc.txt").write_text(desc, encoding="utf-8")
            except Exception:
                pass
        return {"img": img.name, "success": r.get("success", False),
                "elapsed_s": dt, "desc_len": len(desc),
                "preview": desc[:120].replace("\n", " ")}

    print(f"[INFO] {len(images)} 张图, 并行 {args.workers} 路, 模型 {config.get('glm_model')}")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(one, images))
    total = round(time.time() - t0, 1)

    print(f"[OK] 并行总耗时 {total}s (平均每张 {round(total/len(images),1)}s)")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
