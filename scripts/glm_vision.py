"""glm-4.6v-flash 视觉模型调用：图片 → 结构化语义描述

原图直传（base64 原始字符串，不加 data URI 前缀），质量优先
temperature=0 + thinking=disabled 保证输出确定性
"""

import base64
import time
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

GLM_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


def encode_image_base64(image_path):
    """将原图编码为base64原始字符串（不加 data URI 前缀）

    glm-4.6v-flash 官方文档要求直接传入 base64 编码字符串，
    而非 data:image/xxx;base64,... 形式。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    ext = path.suffix.lower().lstrip(".")
    format_map = {
        "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
        "gif": "gif", "webp": "webp", "bmp": "bmp",
    }
    image_format = format_map.get(ext, "jpeg")

    with open(image_path, "rb") as f:
        raw_data = f.read()

    b64 = base64.b64encode(raw_data).decode("utf-8")
    return b64, image_format


def call_glm_vision(image_path, config, custom_prompt=None, max_tokens=None):
    """调用GLM视觉API，返回结构化描述

    max_tokens: 覆盖 config 的 glm_max_tokens(用于精简模式压缩输出)。
                为 None 时用 config["glm_max_tokens"]。
    请求结构：system(固定) + user[text(固定), image(变化)]
    前缀稳定以最大化API侧缓存命中
    """
    if not HAS_REQUESTS:
        return {"success": False, "description": "", "error": "requests包未安装", "usage": {}}

    api_key = config.get("glm_api_key", "")
    if not api_key:
        return {"success": False, "description": "", "error": "GLM API Key未配置", "usage": {}}

    # 编码原图为 base64 原始字符串（glm-4.6v-flash 要求）
    try:
        image_b64, _ = encode_image_base64(image_path)
    except FileNotFoundError as e:
        return {"success": False, "description": "", "error": str(e), "usage": {}}
    except Exception as e:
        return {"success": False, "description": "", "error": f"图片编码失败: {e}", "usage": {}}

    # 构造请求：user[image, text]，无 system（glm-4.6v 要求 image 在前且无 system 才返回内容）
    # 若该结构返回空，降级重试另一种结构（兼容 glm-4v-flash 等需要 system+text 在前的模型）
    user_prompt = custom_prompt or config.get("user_prompt", "")
    out_tokens = max_tokens if max_tokens is not None else config.get("glm_max_tokens", 4096)

    def build_payload(img_b64, mode):
        """mode=1: image在前无system (glm-4.6v); mode=2: system+text在前 (glm-4v-flash等)"""
        if mode == 1:
            return {
                "model": config["glm_model"],
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": img_b64}},
                        {"type": "text", "text": user_prompt},
                    ]},
                ],
                "temperature": config.get("glm_temperature", 0.0),
                "max_tokens": out_tokens,
                "thinking": {"type": "enabled" if config.get("glm_thinking", False) else "disabled"},
            }
        else:
            return {
                "model": config["glm_model"],
                "messages": [
                    {"role": "system", "content": config.get("system_prompt", "")},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": img_b64}},
                    ]},
                ],
                "temperature": config.get("glm_temperature", 0.0),
                "max_tokens": out_tokens,
                "thinking": {"type": "enabled" if config.get("glm_thinking", False) else "disabled"},
            }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = config["glm_base_url"]
    timeout = config.get("http_timeout", 120)
    retries = config.get("http_retries", 3)

    print(f"[INFO] 调用GLM视觉 (图片: {Path(image_path).name}, 模型: {config['glm_model']})")

    # 结构降级: mode=1 (image在前无system) 优先, 若返回空content则试 mode=2 (system+text在前)
    for mode in (1, 2):
        payload = build_payload(image_b64, mode)
        last_error = ""
        for attempt in range(retries + 1):
            try:
                start = time.time()
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                elapsed = int(time.time() - start)

                if resp.status_code == 200:
                    result = resp.json()
                    description = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = result.get("usage", {})

                    # 若 content 为空(glm-4.6v thinking 或结构不兼容), 换下一个 mode
                    if not description.strip():
                        print(f"[WARN] mode{mode} 返回空content (usage: {usage}), 尝试另一结构")
                        break

                    cached = usage.get("cached_tokens", 0)
                    if cached:
                        print(f"[INFO] GLM响应成功 ({elapsed}s, mode{mode}, 缓存命中: {cached} tokens)")
                    else:
                        print(f"[INFO] GLM响应成功 ({elapsed}s, mode{mode})")

                    return {
                        "success": True,
                        "description": description.strip(),
                        "error": "",
                        "usage": usage,
                    }
                else:
                    error_msg = resp.text[:500]
                    last_error = f"HTTP {resp.status_code}: {error_msg}"
                    print(f"[WARN] GLM请求失败 (尝试 {attempt+1}/{retries+1}, mode{mode}): {last_error[:100]}")

                    # 429 退避重试
                    if resp.status_code == 429 and attempt < retries:
                        time.sleep((attempt + 1) * 1.0)
                        continue
                    # 4xx 不重试
                    if 400 <= resp.status_code < 500 and resp.status_code != 429:
                        break

            except requests.exceptions.Timeout:
                last_error = f"请求超时 ({timeout}s)"
                print(f"[WARN] GLM超时 (尝试 {attempt+1}/{retries+1}, mode{mode})")
            except requests.exceptions.ConnectionError as e:
                last_error = f"连接错误: {e}"
                print(f"[WARN] GLM连接错误 (尝试 {attempt+1}/{retries+1}, mode{mode})")
            except Exception as e:
                last_error = f"异常: {e}"
                print(f"[WARN] GLM异常 (尝试 {attempt+1}/{retries+1}, mode{mode}): {e}")

            if attempt < retries:
                time.sleep((attempt + 1) * 0.5)

    error_type = "API错误"
    if "401" in last_error:
        error_type = "认证失败"
        last_error = "API Key无效"
    elif "413" in last_error:
        error_type = "文件过大"
        last_error = "图片太大，请减小文件体积"
    elif "429" in last_error:
        error_type = "频率限制"
        last_error = "请求频率超限，请稍后重试"

    return {"success": False, "description": "", "error": f"[{error_type}] {last_error}", "usage": {}}


if __name__ == "__main__":
    import sys
    # Windows 控制台/重定向默认 GBK 编码无法输出 Ø/±/° 等字符, 强制 UTF-8 输出
    # (否则读取含 Ø 的图纸标注时 print 抛 UnicodeEncodeError, 已拿到的结果也会丢失)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    sys.path.insert(0, str(Path(__file__).parent))
    from config import load_config, validate_config, print_config

    if len(sys.argv) < 2:
        print(f"用法: python glm_vision.py <image_path> [custom_prompt]")
        print(f"支持格式: {', '.join(sorted(GLM_IMAGE_EXTS))}")
        sys.exit(1)

    config = load_config()
    print_config(config)

    valid, msg = validate_config(config)
    if not valid:
        print(f"[ERROR] {msg}")
        sys.exit(1)

    result = call_glm_vision(sys.argv[1], config, sys.argv[2] if len(sys.argv) > 2 else None)
    if result["success"]:
        print("\n=== GLM视觉描述 ===")
        print(result["description"])
    else:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)
