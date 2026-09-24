# large-4dstem-analysis

[English](README.md) | 简体中文

面向大体积 4D-STEM 数据的命令行分析项目，支持分块读取、虚拟成像、衍射特征分类、ROI 布拉格斑点检测，以及后续晶相和取向候选分析。

统一流程通过一个 YAML 配置文件选择和运行各阶段：

1. **Stage 1**：虚拟成像、径向衍射指纹及无监督特征分类。
2. **Stage 2A**：在感兴趣区域（ROI）内提取布拉格斑点。
3. **Stage 2B**：基于候选晶体结构进行晶体学索引。
4. **Stage 2C（可选）**：整理 pyxem 全衍射图模板匹配验证结果。
5. **共识分析（可选）**：比较不同证据分支，标记一致、歧义和冲突。

对于尚无样品成分或衍射标定的 MIB 数据，可先使用独立的基础分析批处理入口。

## Fe BCC／FCC 逐点相识别

在基础分析和束心位移审计完成后，对全部扫描位置进行晶体学匹配：

```powershell
conda env create -f environment-phase.yml
conda activate fourdstem-phase
python -m fourdstem_pipeline.phase_identification --config configs/phase_identification.yaml
```

也可在已有 **Python 3.12** 环境中执行 `pip install -e ".[phase-identification,test]"`。本流程使用 py4DSTEM 0.14.18 和 NumPy 1.x，与新版 NumPy 基础分析环境分开安装。当前工作区可使用 `.\.conda-phase\python.exe`。

当前三份扫描的候选结构已修正为用户提供的 **Fe BCC／FCC**。默认输入目录为 `C:/Users/jayliu/Documents/Data`；两份 CIF 原样复制到仓库参考目录，晶格常数与原子坐标采用文件原值。程序检查纯 Fe 元素、占位和展开后的空间群，兼容 `Fe0+` 标记。结构来源及 SHA-256 见 [CIF 说明](references/cifs/README.md)。原 Ti 匹配结果仅保留为历史记录，不可作为本批 Fe 相结论。

| 标签 | 结构 | 推断空间群 | 来源 |
| --- | --- | --- | --- |
| `0` | Fe-BCC | 229，Im-3m | 用户 `Fe-BCC.cif` 原样副本，a=2.86303550 Å |
| `1` | Fe-FCC | 225，Fm-3m | 用户 `Fe-FCC.cif` 原样副本，a=3.65555117 Å |
| `-1` | 未索引 | — | 标定、束心或匹配证据不足 |
| `-2` | 歧义 | — | 相竞争、电压假设或扰动检验未通过 |

流程逐点定位直射束，以对称斑点和邻域残差复核；只转换斑点坐标，不插值原始衍射图。异常高计数仅在检测副本中处理。几何匹配保留积分强度至少 30、且不低于该图最强非中心斑点 10% 的斑点；全部检测记录仍保留。

模板使用完整原子结构、运动学结构因子和激发误差，默认区轴步长 2°，自由搜索平面内转角及横向镜像，采用一对一斑点评分，没有模拟评分后备分支。在 `32×32` 空间网格划分训练／验证区域，用最多 128 个训练图样搜索 `0.005–0.05 Å⁻¹/像素`，用独立留出图样复核。每份扫描的三个相共用一个尺度；标定不可靠时保存逐点诊断，但不接受相标签。

未知电压按 80／120／200／300 kV 四种假设处理。接受标签须在全部假设下一致，至少匹配 6 个非中心斑点、具有非共线方向、残差中位数不超过 1.5 像素、归一化相间分差至少 10%，并通过固定分数阈值与随机角度对照。初筛标签还须通过束心沿两轴各 ±1 像素、尺度 ±2% 的独立扰动检验。相图保留原扫描分辨率，不平滑或插值标签。

最终标签还要求模板解释至少 70% 的入选斑点，以减少仅匹配部分斑点时的误接受；这仍不能排除少量重叠晶粒。可运行 `python scripts/validate_phase_identification.py` 复现独立区轴、不同尺度、束心平移、噪声、缺失斑点、离群点、随机角度和混合图样验证；结果保存在输出目录的 `synthetic_validation.json`。

分数不是概率；候选结构条件下的尺度估计不能代替标准样品标定。图样自助抽样区间不包含全部空间相关和模型误差。运动学模板不描述动态衍射；重叠晶粒、畸变、应变、遗漏的相及模板离散都可能导致拒绝或误配。本入口不输出定量取向、应变或材料体积分数。

### 相识别输出与断点续跑

默认输出至 `outputs/phase_identification_fe/`，原始 MIB、CIF 和基础分析结果保持不变。打开 `report_zh.html` 查看汇总，各扫描子目录包含：

- `phase_id.npy`：通过全部检查的标签；`best_candidate.npy`：包含被拒绝位置的最佳候选，**不能直接作为相图**。
- `score.npy`、`margin.npy`、`matched_peaks.npy`、`median_residual_px.npy`、`reason_flags.npy`：评分与拒绝依据。
- `all_results.npy`：全部相／电压假设的模板、转角、匹配数与残差；字段见 `array_schema.json`。
- `peaks/`：束心、全部检测斑点、强度及质量标记；`selected_peak_indices.npy` 记录用于匹配的斑点索引。
- `calibration.json`、`calibration_samples.npz`：标定状态、训练／留出索引、尺度曲线、自助抽样区间和随机对照。
- `pattern_evidence.png`、`representative_evidence.json`：实测图样、模板反射与 hkl 叠加证据。
- `matching_checkpoint.json`：配置、CIF 哈希、原始文件大小／时间、源码哈希和依赖版本。

重复相同命令即可续跑。输入、配置或算法变化时，程序拒绝复用旧缓存；请用 `--output outputs/phase_identification_run2` 指定新目录。不要同时对同一输出扫描运行多个进程。也可分阶段执行：

```powershell
python -m fourdstem_pipeline.phase_identification --stage prepare
python -m fourdstem_pipeline.phase_identification --stage extract --scan scan_01_1045
python -m fourdstem_pipeline.phase_identification --stage identify --scan scan_01_1045
python -m pytest -q
```

`identify` 要求对应扫描已完成斑点提取。若标定失败，完整相图可以全部为 `-1`；这表示当前证据不足，不能解释为非晶或不存在候选相。

本批三份实测数据及合成控制的结果见 [Fe 相识别验证记录](docs/phase_identification_fe_validation.zh-CN.md)。[原 Ti 验证记录](docs/phase_identification_validation.zh-CN.md)已被结构修正取代，仅供历史追溯。

接受率及拒绝原因的逐项重放见 [Fe 匹配诊断](docs/fe_matching_diagnostics.zh-CN.md)，包含扰动敏感性、电压假设、斑点覆盖率和模板搜索对照。

历史 Ti CIF 已从仓库移除，旧 Ti pyxem／取向批处理入口已停用。统一配置仅默认启用 Stage 1／2A，所有 CIF 候选列表为空；旧索引和一致性 API 未迁移到 Fe。本批 MIB 请使用上述独立逐点入口。

## 安装

以下命令均在项目根目录运行。

### 基础分析环境

在已有 Python 环境中安装基础分析依赖：

```powershell
python -m pip install -e ".[basic-analysis]"
```

Windows 下也可以使用项目独立虚拟环境，无需激活脚本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[basic-analysis]"
```

使用此方式时，将后文命令中的 `python` 替换为 `.\.venv\Scripts\python.exe`。

### 完整分析环境

需要 HyperSpy、pyxem 和 py4DSTEM 等组件时，可按仓库环境定义创建 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate large-4dstem
python -m pip install -e ".[basic-analysis,diffraction]"
```

已有 `large-4dstem` 环境时，直接激活并安装项目即可。[environment.yml](environment.yml) 将该环境的 Python 版本限定为 `>=3.10,<3.13`；安装声明和可选依赖组见 [pyproject.toml](pyproject.toml)。

Windows 下若 `python` 打开 Microsoft Store 或立即退出，请先确认调用的是所选 Conda 或虚拟环境中的解释器。

## MIB 数据基础分析

### 批量运行

下面的命令分析 `data` 目录下的全部 `*.mib` 文件，假定每份文件包含逐行同向排列的 `256×256` 个扫描点：

```powershell
python -m fourdstem_pipeline.basic_analysis --data data --output outputs/basic_analysis --scan-shape 256 256
```

完成后打开 `outputs/basic_analysis/report_zh.html` 查看中文汇总报告，也可阅读同目录的 `report_zh.md`。

该入口支持单芯片、已处理的 U16 MIB 文件，要求每帧具有 384 字节帧头。它逐帧核对帧头和序号，以大端 `>u2` 内存映射方式读取数据，按 `16×16` 个扫描点分块处理，保留原始扫描和探测器分辨率，不进行合并像素（binning）。

运行前应确认实际扫描行列数和排列方式。程序不会仅凭文件名推断蛇形扫描、回扫帧或扫描步长。

输出目录必须为空或尚不存在。重复分析时请指定新目录，例如 `--output outputs/basic_analysis_run2`。各文件分析阶段的失败会单独记录，并继续处理后续文件；存在失败文件时，批处理返回非零退出码。

### 分析内容

- 原始计数质量检查：零强度帧、零计数比例、计数分布、强度离群点及行间强度变化。
- 虚拟明场（BF）、环形暗场（ADF）、分环积分图和总强度图。
- 平均衍射图、最大值衍射图及径向强度曲线。
- PCA/NMF 特征分析、四类无监督衍射特征分区、各类占比和代表区域。
- 文件间径向曲线对比、数值验证、配置快照及中文图文报告。
- 扫描相关亮斑位移复核，以及高计数像素的坐标和值。

径向曲线先按各自最大值归一化，再进行 PCA 和 KMeans 分类；NMF 提供补充特征。默认四类和随机种子 `0` 是探索参数，**四类不代表四个晶相**。每份文件独立拟合类别和探测器掩膜，不同文件中的相同类别编号没有自动对应关系。

### 束斑位移与结果解读

程序在均匀抽取的最多 `16×16` 个扫描点上寻找中心区域亮斑，以稳健仿射拟合检查亮斑位置是否随扫描坐标系统变化。若发现明显位移，会在报告和质量检查清单中标注其对固定束心分类的影响。

**存在明显束斑位移时，近同心的类别图不能直接解释为材料分区或晶相分布。** 平均衍射图中的宽亮区也可能来自不同扫描位置亮斑的叠加，不代表单帧束斑尺寸。基础分析保留固定掩膜成像结果，不自动执行束移校正；后续材料分区应先处理逐点束心变化。

可对已完成的基础分析批次重新复核：

```powershell
python -m fourdstem_pipeline.beam_audit outputs/basic_analysis
```

该命令会读取原始数据的抽样点，并更新输出目录中的诊断、报告和质量状态；原始 MIB 文件不变。

其他解释约定：

- 扫描坐标使用扫描点，衍射坐标使用探测器像素；文件名中的纳米步长仅记录为待核实元数据。
- 以 `4095` 检查疑似饱和基于 12-bit 假设。超过此范围的值同时记为原始计数异常，不直接断言为物理饱和。
- 零计数不等于坏像素，行间强度变化也可能来自真实样品结构；程序不会自动删除这些点。
- 图像可独立调整显示对比度，定量比较应使用保存的数值数组。逐文件生成的虚拟探测器掩膜也可能不同。
- 基础分析不提供晶相识别、定量晶体取向、晶格间距或应变结论。

### 基础分析输出

```text
outputs/basic_analysis/
  report_zh.html                 # 中文汇总报告
  report_zh.md
  comparison.csv                 # 文件间质量指标对比
  radial_comparison.png
  batch_summary.json             # 文件状态、结果摘要及异常
  analysis.log
  environment.txt
  scan_01_<文件标识>/
    report_zh.html               # 单份数据报告
    config.yaml                 # 实际运行配置
    input_metadata.json
    geometry.json               # 中心估计及虚拟探测器参数
    raw_quality.json
    raw_quality_arrays.npz
    verification.json
    beam_motion.json
    beam_motion_samples.npz
    beam_motion.png
    high_count_pixels.json
    representative_regions.json
    representative_patterns.npz
    overview.png
    feature_classes.png
    quality.png
    virtual/
    fingerprints/
    fingerprint_classes/
```

`data/`、`outputs/` 和 `.venv/` 默认由 `.gitignore` 排除，原始数据、分析结果和本地环境不会随普通 Git 提交保存。

## 统一多阶段流程

### 快速开始

先用合成数据执行仅包含 Stage 1 的轻量检查：

```powershell
fourdstem-pipeline --config configs/pipeline_smoke.yaml
```

按实际实验数据修改配置后，运行所有启用的阶段：

```powershell
fourdstem-pipeline --config configs/pipeline.yaml
```

等价的 Python 模块入口：

```powershell
python -m fourdstem_pipeline.cli pipeline --config configs/pipeline.yaml
```

[configs/pipeline.yaml](configs/pipeline.yaml) 是统一基础分析示例，保留 `data/0617-4d` 路径及 `512×512` 扫描尺寸，历史 Ti CIF 候选已清空，默认仅运行 Stage 1／2A。使用新数据时应核对输入路径及扫描尺寸；本地三份 `256×256` 扫描数据的基础分析可使用前述批处理命令自动生成独立配置。Fe 匹配使用 `configs/phase_identification.yaml`。

### 配置结构

| 配置节 | 用途 |
| --- | --- |
| `pipeline` | 启用阶段及统一流程输出目录 |
| `project` / `data` / `preprocess` | Stage 1 输入与预处理 |
| `geometry` / `virtual_images` | 探测器几何参数与虚拟图像 |
| `phase_screening` / `orientation` / `sample_mask` | 特征分类、取向预览及样品掩膜 |
| `stage2a` | ROI 布拉格斑点提取及 py4DSTEM 检测参数 |
| `stage2b` | CIF 模板生成、匹配及候选晶相 |
| `stage2c` | pyxem/HyperSpy 极坐标模板匹配验证输入及质量检查 |
| `consensus` | Stage 2B 与 Stage 2C 的一致性及冲突分析 |

通过阶段列表选择运行范围：

```yaml
pipeline:
  stages: [stage1, stage2a, stage2b]
```

仅运行基础阶段时设为 `[stage1]`；需要 ROI 布拉格检测时设为 `[stage1, stage2a]`。

关闭 Stage 1 取向预览：

```yaml
orientation:
  enabled: false
```

省略 `enabled` 时保留原有默认启用行为。

显式使用固定帧 MIB 内存映射读取时，需要同时提供帧布局参数：

```yaml
data:
  path: data/your_scan.mib
  backend: mib_memmap
  scan_shape: [256, 256]
  detector_shape: [256, 256]
  dtype: '>u2'
  mib_header_bytes: 384
  lazy: true
```

历史 pyxem NPZ 导入 API 仅保留用于读取已有归档。原 Ti 模拟脚本及批处理入口已停用；归档中的 Ti 相编号不能重新解释为 Fe BCC／FCC。

阶段间路径由统一流程自动传递：Stage 2A 接收 Stage 1 的 `stage1_dir`，Stage 2B 接收 Stage 2A 的 `stage2_dir`，Stage 2C 接收 Stage 2A/2B 目录，并输出可供共识分析使用的 `stage2c_manifest.json`。

### 多阶段输出

| 输出 | 内容 |
| --- | --- |
| `pipeline_summary.json` | 各阶段状态、路径、错误及跳过原因 |
| Stage 1 目录 | `stage1_summary.json`、质量检查、报告、虚拟图像、衍射指纹及 ROI 候选 |
| Stage 2A 目录 | `stage2_summary.json`、布拉格斑点图、ROI 摘要、基准指标及图库 |
| Stage 2B 目录 | `stage2_indexing_summary.json`、晶相/取向证据及报告 |
| Stage 2C 目录 | `stage2c_summary.json`、标准化 pyxem 数组及验证清单 |

典型目录结构：

```text
outputs/<run>/
  stage1_summary.json
  pipeline/pipeline_summary.json
  stage2/roi_bragg/
    stage2_summary.json
    stage2b_indexing/
      stage2_indexing_summary.json
    stage2c_pyxem_validation/
      stage2c_summary.json
      stage2c_manifest.json
```

Stage 2B 评估布拉格矢量的晶体学证据，Stage 2C 评估全衍射图的极坐标模板匹配证据。共识层保留一致、歧义与冲突信息，不强制将不同证据合并为唯一晶相图。

## 命令参考

| 命令 | 功能 |
| --- | --- |
| `fourdstem-pipeline` | 运行统一多阶段流程 |
| `fourdstem-run` | 仅运行 Stage 1 |
| `fourdstem-dry-run` | 检查 Stage 1 配置并估算资源 |
| `fourdstem-stage2` | Stage 2A ROI 布拉格斑点检测 |
| `fourdstem-stage2b` | Stage 2B 索引 |
| `fourdstem-stage2c` | 标准化 pyxem 验证输出 |
| `fourdstem-evidence-qc` | Stage 2A 证据可信度诊断 |
| `fourdstem-bin-export` | 对原始数据进行 binning 并导出 EMD/H5 |
| `fourdstem-crop-export` | 裁剪扫描维度并导出 EMD/H5 |

各阶段可独立运行，并读取同一份统一配置：

```powershell
fourdstem-run --config configs/pipeline.yaml
fourdstem-stage2 --config configs/pipeline.yaml
fourdstem-stage2b --config configs/pipeline.yaml
fourdstem-stage2c --config configs/pipeline.yaml
```

也可通过模块入口调用：

```powershell
python -m fourdstem_pipeline.cli run --config configs/pipeline.yaml
python -m fourdstem_pipeline.cli stage2 --config configs/pipeline.yaml
python -m fourdstem_pipeline.cli stage2b --config configs/pipeline.yaml
python -m fourdstem_pipeline.cli stage2c --config configs/pipeline.yaml
```

## 开发与验证

安装测试依赖。现有工作流测试包含 Parquet 输出检查，因此还需 `pyarrow`：

```powershell
python -m pip install -e ".[basic-analysis,test]" pyarrow
python -m pytest tests/test_workflow.py tests/test_basic_analysis.py -q
```

基础分析测试覆盖 MIB 字节序与帧完整性、分块计数一致性、束心估计、扫描相关束移、取向开关，以及报告生成和重复复核。

## 坐标约定

所有阶段统一使用以下顺序：

| 对象 | 顺序 |
| --- | --- |
| 四维数据轴 | `nav_y, nav_x, q_y, q_x` |
| ROI 边界框 | `y0, y1, x0, x1` |
| 点或中心坐标 | `y, x` |

Stage 1 ROI 候选使用预处理后的扫描坐标；Stage 2A 根据 `r_bin` 将其转换为原始扫描坐标。
