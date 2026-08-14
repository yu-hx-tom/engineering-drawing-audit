# Engineering Drawing Audit (工程图纸审核 Skill)

自动化审核**客户工程原图**与 **SolidWorks/CAD 重绘图**的尺寸差异,生成答案级审核报告。含文字层尺寸台账提取、自动视图划分、GLM 视觉物理特征标定、原图三层核销、防锚定交叉验证与批量并行识别。

## 功能特性

- **重绘侧机器化锚点**:`extract_dimension_ledger.py` 提取尺寸台账(值/公差/类型/参考属性),`detect_drawing_views.py` 自动划分视图并输出每视图 PNG。
- **GLM 视觉标定**:工程图专用提示词(`prompts/engineering-drawing.txt`),按视图分组捕获 Ø/R/±/度分秒/粗糙度/基准/形位公差。
- **批量并行 GLM**:`batch_glm.py` 多图并行识别,墙钟 = 最慢单张(实测 3.3× 加速)。
- **原图三层核销**:整体自由读 → 台账双向比对(0 GLM) → 差异候选定向核验,禁止逐尺寸调 GLM。
- **防锚定**:客户读数先独立读原图再比对重绘,杜绝"客户值被重绘值吸过去"的误判。
- **低清增强**:`enhance_image.py` CLAHE + 插值放大,提升 200dpi 扫描识别率。
- **答案级报告**:状态 6 类(Match/等价/差异/错误/遗漏/需人工),按客户视图分节,ID 完整性核销。

## 审核流程

```
工程图 PDF(客户原图 + 重绘图)
  │
  ├─ extract_dimension_ledger.py → 尺寸台账(尺寸个数/数值/所在视图)
  ├─ detect_drawing_views.py     → 视图划分(view-regions.json + 每视图 PNG)
  ├─ batch_glm.py 并行            → GLM 逐视图标定物理特征
  │
  ├─ 原图: ①整体自由读 → ②台账双向比对(0 GLM) → ③差异候选定向核验
  │
  └─ 生成答案级审核报告(报告两段式,完整不压缩)
```

## 安装

```bash
# Python 3.10+
pip install pymupdf numpy pillow requests
```

## API Key 配置

GLM 视觉识别需要智谱 GLM API key,通过环境变量加载(脚本内不含任何明文 key):

```bash
# Linux/macOS
export GLM_API_KEY="your-glm-api-key"
# 可选: MinerU 文档 OCR(用于带文字层的 PDF 提取)
export MINERU_API_KEY="your-mineru-api-key"

# Windows PowerShell
$env:GLM_API_KEY = "your-glm-api-key"
```

`scripts/config.py` 中 `glm_base_url` 默认指向智谱官方端点,可用环境变量覆盖。

## 用法

```bash
# 1. 重绘侧: 尺寸台账 + 视图划分
python scripts/extract_dimension_ledger.py "重绘图.pdf" --unit mm -o output/ledger
python scripts/detect_drawing_views.py "重绘图.pdf" --ledger output/ledger/dimension-ledger.json -o output/views

# 2. 渲染 PDF / 高 DPI 局部裁剪
python scripts/render_pdf.py "图纸.pdf" output/render --dpi 300
python scripts/render_pdf.py "图纸.pdf" output/crops --crop 1 <x> <y> <w> <h> --dpi 600

# 3. GLM 视觉识别(单图 / 批量并行)
python scripts/glm_drawing.py "局部图.png"
python scripts/batch_glm.py 视图1.png 视图2.png ... --workers 4 --out-dir output/desc

# 4. 低清裁剪图增强后识别
python scripts/enhance_image.py "低清裁剪.png" --method clahe --gray --target-width 1600 --out 增强图.png
python scripts/glm_drawing.py 增强图.png

# 5. 尺寸表核销(可选)
python scripts/build_dim_table.py "重绘图.pdf" output/dimtable
python scripts/check_completeness.py output/dimtable/dim-table.json "GLM读数.txt"

# 6. 自检
python scripts/extract_dimension_ledger.py --self-test
```

## 目录结构

```
engineering-drawing-audit/
├── SKILL.md                 # 完整审核规范与铁律
├── README.md
├── LICENSE                  # MIT
├── prompts/
│   └── engineering-drawing.txt   # 工程图专用 GLM 提示词
├── references/
│   ├── answer-grade-standard.md  # 答案级报告规范
│   └── audit-checklist.md        # 复查清单
└── scripts/
    ├── extract_dimension_ledger.py  # 尺寸台账提取
    ├── detect_drawing_views.py      # 自动视图划分
    ├── validate_view_crops.py       # 视图像素验收
    ├── glm_vision.py                # GLM 视觉 API 封装
    ├── glm_drawing.py               # 工程图 GLM 封装
    ├── batch_glm.py                 # 批量并行 GLM
    ├── render_pdf.py                # PDF 渲染/裁剪
    ├── enhance_image.py             # 低清增强
    ├── build_dim_table.py           # 文字层坐标尺寸表
    ├── check_completeness.py        # 读数核销
    ├── build_param_list.py
    ├── verify_claim.py
    ├── config.py                    # 配置(空 key + 环境变量)
    └── test_extract_dimension_ledger.py
```

## 状态分类

| 状态 | 含义 |
|---|---|
| Match | 同一物理特征完整标注一致 |
| Equivalent expression | 信息保留,制图表达不同(视图合并/对称/分数转换等) |
| Difference to note | 参考属性移除/小数值变化 |
| Confirmed error | 同一特征数值/符号/公差冲突 |
| Confirmed omission | 客户标注在重绘中无等价表达 |
| Needs manual confirmation | 源图模糊或特征对应不完整 |

## 许可证

MIT License。详见 [LICENSE](LICENSE)。

## 免责声明

- 客户原图是需求来源(权威),重绘图仅作佐证。
- API key 通过环境变量提供,仓库不含任何凭据。
- 本工具辅助工程图纸人工复核,不替代专业工程师判断。
