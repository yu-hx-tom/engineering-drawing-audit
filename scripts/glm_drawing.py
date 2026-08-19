"""工程图纸专用 GLM 视觉读取封装

强制加载 visuals-skill/prompts/engineering-drawing.txt 专用提示词,
杜绝回退到默认通用 10 维描述([TYPE]/[SCENE]/[COLOR]...)。

为什么需要本封装:
- glm_vision.py 的逻辑是 `custom_prompt or config.user_prompt`——若调用方漏传
  专用提示词, 会静默回退到通用描述, 对图纸任务产生一堆无用的 TYPE/SCENE/COLOR。
- image_to_text.py 会把 GLM 输出强制归一化到 10 个通用标签, 破坏图纸的
  "视图 | 标注 | 物理特征" 格式输出, 图纸图片不能走它的 GLM 分支。
- 本封装: 专用提示词文件找不到时报错退出(绝不回退), 并支持 --extra 追加
  补充指令(如原图反向查证"找 1705.7 附近直径")。

用法:
  python glm_drawing.py <image.png>                       # 用专用提示词读图
  python glm_drawing.py <image.png> --extra "找 1705.7 附近直径"   # 追加指令
  python glm_drawing.py <image.png> --terse "查 443.9 还是 445"   # 精简定向模式(省 token)
  python glm_drawing.py <image.png> --prompt <file>       # 覆盖提示词文件
  python glm_drawing.py <image.png> --json                # 输出 JSON
"""

import argparse
import json
import sys
from pathlib import Path

# 工程图专用提示词(强制; 相对本脚本: skills/engineering-drawing-audit/scripts/glm_drawing.py
# → ../../visuals-skill/prompts/engineering-drawing.txt)
SKILLS_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_FILE = SKILLS_ROOT / "visuals-skill" / "prompts" / "engineering-drawing.txt"
TERSE_PROMPT_FILE = SKILLS_ROOT / "visuals-skill" / "prompts" / "engineering-drawing-terse.txt"
VISUALS_SCRIPTS = SKILLS_ROOT / "visuals-skill" / "scripts"

# 精简模式默认输出上限(定向核验只需少量 token)
TERSE_MAX_TOKENS = 800


def main():
    parser = argparse.ArgumentParser(description="工程图纸专用 GLM 视觉读取(强制专用提示词)")
    parser.add_argument("image", help="图片路径(渲染图/裁剪图/客户原图)")
    parser.add_argument("--extra", default="", help="追加到提示词末尾的补充指令, 如'找 1705.7 附近直径'")
    parser.add_argument("--terse", action="store_true",
                        help="精简定向模式: 用 engineering-drawing-terse.txt, 只答目标标注不遍历全图, 并压低输出 token(用于定向核验/反向查证)")
    parser.add_argument("--prompt", help="覆盖提示词文件(默认 engineering-drawing.txt，--terse 下默认 engineering-drawing-terse.txt)")
    parser.add_argument("--max-tokens", type=int, default=None, help="覆盖输出 token 上限(默认完整模式 4096 / 精简模式 800)")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    args = parser.parse_args()

    # Windows 控制台/重定向默认 GBK 无法输出 Ø/±/° 等字符, 强制 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # 专用提示词: 找不到就报错退出, 绝不回退到默认通用描述
    prompt_file = Path(args.prompt) if args.prompt else (TERSE_PROMPT_FILE if args.terse else PROMPT_FILE)
    if not prompt_file.exists():
        print(f"[ERROR] 工程图纸专用提示词不存在: {prompt_file}", file=sys.stderr)
        print("        请检查 visuals-skill/prompts/*.txt。拒绝回退到通用描述。", file=sys.stderr)
        sys.exit(1)
    base_prompt = prompt_file.read_text(encoding="utf-8")
    custom_prompt = (base_prompt + "\n\n" + args.extra).strip() if args.extra else base_prompt

    # 复用 visuals-skill 的 GLM 调用
    sys.path.insert(0, str(VISUALS_SCRIPTS))
    from config import load_config
    from glm_vision import call_glm_vision

    config = load_config()
    out_tokens = args.max_tokens if args.max_tokens is not None else (TERSE_MAX_TOKENS if args.terse else None)
    result = call_glm_vision(args.image, config, custom_prompt, max_tokens=out_tokens)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["success"]:
            print("\n=== 工程图纸 GLM 识别 ===")
            print(result["description"])
        else:
            print(f"\n[ERROR] {result['error']}", file=sys.stderr)
            sys.exit(1)
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
