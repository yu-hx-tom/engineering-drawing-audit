"""配置管理：GLM API + MinerU API 的 Key、模型参数"""

import os
import json
from pathlib import Path

# GLM 固定系统提示词（要求GLM按固定结构输出，便于DeepSeek缓存命中）
SYSTEM_PROMPT = (
    "你是一位资深的图像分析专家，具备计算机视觉、平面设计、数据可视化等多领域经验。"
    "你的任务是将图片内容转换为深度结构化的文字描述，供下游纯文本推理模型（如DeepSeek）使用。"
    "你必须严格按照固定标签格式输出，每个标签独占一行，标签内容紧跟标签。"
    "每个标签的内容要尽可能详尽、准确、专业，不遗漏任何可见信息。"
    "缺少对应内容时填写'无'，不要省略标签。"
)

# GLM 固定用户指令（10个专业维度，内容稳定 + 强制结构化输出）
DEFAULT_USER_PROMPT = (
    "请按以下固定格式深度描述这张图片，每个标签独占一行，内容紧跟标签后换行：\n"
    "[TYPE]\n图片类型（照片/截图/图表/文档/漫画/UI/Logo/图标/地图/技术图纸/其他），并说明判断依据\n"
    "[SUMMARY]\n用2-3句话概括图片的核心内容和主要功能\n"
    "[SCENE]\n详细描述图片场景：主体对象、背景环境、氛围、拍摄/制作角度\n"
    "[TEXT]\n完整列出图片中所有可见文字，包括标题、正文、标注、水印、UI文字等，逐行列出并标注位置\n"
    "[DATA]\n如果是图表/表格，提取所有数据点（坐标轴、数值、图例、单位）；否则填'无'\n"
    "[LAYOUT]\n详细描述版面结构：各元素的位置关系、对齐方式、层级层次、视觉流向\n"
    "[COLOR]\n主色调、辅助色、背景色、强调色，描述配色方案和视觉风格\n"
    "[OBJECTS]\n图中所有可识别的具体对象/元素，逐个列出并简要描述\n"
    "[QUALITY]\n图片质量评估：清晰度、光线、对比度、是否有噪点/模糊/遮挡\n"
    "[PURPOSE]\n推断图片的用途和目标受众（如信息传达/数据展示/装饰/教学/营销等）\n"
    "\n要求：\n"
    "1. 每个标签必须出现，顺序固定不变\n"
    "2. 缺少内容时填写'无'\n"
    "3. 描述要专业、详尽、准确，不遗漏任何可见信息\n"
    "4. 使用中文回答\n"
    "5. 不要添加额外说明或总结"
)

# MinerU 支持的文件扩展名
IMAGE_EXTS = {"png", "jpg", "jpeg", "jp2", "webp", "gif", "bmp"}
DOCUMENT_EXTS = {"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx"}
SUPPORTED_EXTS = IMAGE_EXTS | DOCUMENT_EXTS

DEFAULT_CONFIG = {
    # GLM 视觉模型（仅处理图片）
    "glm_api_key": "",
    "glm_base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "glm_model": "glm-4.6v",      # 智谱高性能视觉模型(106B,12B激活)
    "glm_temperature": 0.0,              # 完全确定性，相同图片输出稳定
    "glm_max_tokens": 4096,              # 10个标签需要更大输出空间
    "glm_thinking": False,              # 关闭思考模式，保证输出确定性且更快
    "system_prompt": SYSTEM_PROMPT,
    "user_prompt": DEFAULT_USER_PROMPT,

    # MinerU 解析（全文件类型）
    "mineru_api_key": "",
    "mineru_base_url": "https://mineru.net/api/v4",
    "mineru_model_version": "vlm",
    "mineru_language": "ch",
    "mineru_enable_table": True,
    "mineru_enable_formula": True,
    "mineru_is_ocr": True,
    "mineru_poll_interval": 3,
    "mineru_poll_timeout": 600,           # 质量优先，给足解析时间

    # HTTP（质量优先，超时宽松）
    "http_retries": 3,
    "http_timeout": 120,
}

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config(cli_overrides=None):
    """加载配置，优先级：命令行 > 环境变量 > 配置文件 > 默认值"""
    config = DEFAULT_CONFIG.copy()

    # 配置文件
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass

    # 环境变量
    env_glm = os.environ.get("GLM_API_KEY")
    if env_glm:
        config["glm_api_key"] = env_glm
    env_mineru = os.environ.get("MINERU_API_KEY")
    if env_mineru:
        config["mineru_api_key"] = env_mineru

    # 命令行覆盖
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                config[k] = v

    return config


def validate_config(config):
    """校验必要配置是否完整"""
    if not config.get("glm_api_key") and not config.get("mineru_api_key"):
        return False, (
            "GLM 和 MinerU API Key 均未配置。请至少配置一个：\n"
            "  GLM:    export GLM_API_KEY='your-key'  或  --api-key\n"
            "  MinerU: export MINERU_API_KEY='your-key' 或  --mineru-key"
        )
    if config.get("glm_api_key") and not config.get("glm_base_url"):
        return False, "GLM API URL 未配置"
    if config.get("mineru_api_key") and not config.get("mineru_base_url"):
        return False, "MinerU API URL 未配置"
    return True, ""


def is_image_file(file_path):
    """是否为图片文件"""
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext in IMAGE_EXTS


def is_supported_file(file_path):
    """是否为支持的文件类型"""
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext in SUPPORTED_EXTS


def get_file_type(file_path):
    """获取文件类型分类：image / document / unknown"""
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext in IMAGE_EXTS:
        return "image"
    if ext in DOCUMENT_EXTS:
        return "document"
    return "unknown"


def print_config(config, show_key=False):
    """打印当前配置"""
    print("=== 当前配置 ===")
    print("  [GLM 视觉]")
    print(f"    API URL: {config['glm_base_url']}")
    print(f"    模型: {config['glm_model']}")
    print(f"    temperature: {config['glm_temperature']}")

    glm_key = config.get("glm_api_key", "")
    if show_key:
        print(f"    API Key: {glm_key}")
    elif glm_key:
        print(f"    API Key: {'*' * (len(glm_key) - 4)}{glm_key[-4:]}")
    else:
        print("    API Key: [未配置]")

    print("  [MinerU 解析]")
    print(f"    Base URL: {config.get('mineru_base_url', 'https://mineru.net/api/v4')}")
    print(f"    模型版本: {config.get('mineru_model_version', 'vlm')}")
    print(f"    语言: {config.get('mineru_language', 'ch')}")
    print(f"    表格识别: {config.get('mineru_enable_table', True)}")
    print(f"    公式识别: {config.get('mineru_enable_formula', True)}")
    print(f"    强制OCR: {config.get('mineru_is_ocr', True)}")

    mineru_key = config.get("mineru_api_key", "")
    if show_key:
        print(f"    API Key: {mineru_key}")
    elif mineru_key:
        print(f"    API Key: {'*' * (len(mineru_key) - 4)}{mineru_key[-4:]}")
    else:
        print("    API Key: [未配置]")

    print(f"\n  支持格式: {', '.join(sorted(SUPPORTED_EXTS))}")
    print(f"  配置文件: {CONFIG_FILE}")
    print()


if __name__ == "__main__":
    print_config(load_config(), show_key=False)
