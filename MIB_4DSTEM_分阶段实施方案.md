# Merlin MIB 4D-STEM 晶体学分析：分阶段实施方案

**版本：** v1.0（2026-10-08）  
**目标：** 不转换整份原始 MIB，以 `RosettaSciIO → Dask → py4DSTEM` 完成可信的原点校正、Bragg 峰检测、倒易空间标定、相与取向分析，并在 ROI 验证后扩展到全扫描。  
**原则：** 原始 MIB 只读；先验证后扩展；所有坐标、单位、版本和 QC 必须可追溯；不强制输出低置信度相标签。

## 0. 已确认的数据与问题

| 项目 | 当前值 / 结论 |
|---|---|
| 原始数据 | `data/3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib` |
| 文件大小 | 8,615,100,416 bytes（约 8.023 GiB） |
| 扫描维度 | 256 × 256（目前可重建；没有 `.hdr`，还需确认扫描方向） |
| 单张 DP | 256 × 256；动态范围 12-bit，存储类型 `>u2` |
| 帧数 / 帧头 | 65,536 帧 / 384 bytes |
| 采集标称参数 | scan step 5.8 nm；曝光 0.62 ms；CL 110（单位待核对）；加速电压未提供 |
| 现有基础结果 | `origin_sampled.npz`、`origin_fitted.npz`、`bf_fixed/corrected.npy`、`adf_fixed/corrected.npy`、Mean DP 前后对比 |
| 核心发现 | 原点随扫描位置大幅、近似线性移动；原点对齐后平均 DP 的 Bragg 斑点明显清晰；固定 BF/ADF 中的大圆形伪影明显减弱 |
| 关键风险 | 少量直射束误识别；缺少绝对 q 标定；真实束偏转与系统漂移尚未分离；不同 ROI 的坐标与校准不能混用 |

**内存预算（uint16 原始像素，不含中间数组）：** 32×32 ROI 约 128 MiB；64×64 ROI 约 512 MiB；整个数据约 8 GiB。不要对全场调用 `data.compute()`，也不要直接对全场构造 float32 DataCube。

## 1. 总体流程与阶段关卡

```text
MIB（原始只读）
   ↓ RosettaSciIO / Dask
P0 记录环境、元数据与现有基线
   ↓ Gate 0
P1 原点测量质量控制 + 稳健拟合 + 独立验证
   ↓ Gate 1
P2 动态中心 BF/ADF 与原点校正验证
   ↓ Gate 2
P3 选取代表性 ROI → py4DSTEM Bragg 峰检测
   ↓ Gate 3
P4 将 Bragg 峰转换为相对原点像素坐标
   ↓ Gate 4
P5 倒易空间尺度、畸变和方向标定
   ↓ Gate 5
P6 相候选匹配与取向分析（仅在标定可信时）
   ↓ Gate 6
P7 全场分块、断点续跑、汇总与论文级 QC
```

**工期估算：** P0–P2 约 1–3 个工作日（已有部分结果）；P3–P4 约 2–4 天；P5 约 2–4 天（取决于是否有标样/已知晶格）；P6 约 3–5 天；P7 约 2–4 天。为实施规划而非进度承诺。

## 2. P0：固定数据和可复现基线（0.5 天）

**目标：** 把当前已经成功的读取与成像流程固定下来，避免未来参数改变后无法追溯。

**执行：**
1. 记录 `python --version`、`python -m pip check`、`python -m pip freeze`；暂不为了 MIB 读取降级/升级 py4DSTEM。
2. 运行 RosettaSciIO `file_reader(str(path), lazy=True, navigation_shape=(256,256), chunks=(1,16,256,256))`；检查 `shape=(256,256,256,256)`、dtype 和头部记录。
3. 保存随机 16 张原始单点 DP 与四角/中心位置 DP；核验扫描图、帧次序与样品边界是否合理。
4. 备份现有 `origin_sampled.npz`、Mean DP 与 BF/ADF 对比图，不覆盖原文件。
5. 写入运行配置（数据路径、扫描尺寸、坐标约定、探测器大小、软件版本、随机种子）。

**交付：** `outputs/mib_pipeline/P0_baseline/metadata.json`、`environment.txt`、`dp_spotcheck.png`、`baseline_manifest.json`。

**Gate 0：** 数据尺寸、计数、文件大小和 16 张 DP 检查一致；错误或不确定的扫描顺序已明确标注。无须先转换 HDF5。

## 3. P1：原点场质量控制（1–2 天）

**已有输入：** `outputs/origin_check/origin_sampled.npz`，含抽样扫描坐标 `scan_x/scan_y` 和检测原点 `origin_x/origin_y`（32×32）。

**实施：**
1. 首先区分坐标：图像索引 `(det_y, det_x)`，Bragg 点导出使用 `(qx_px, qy_px)`；所有数组都注明实际顺序，禁止直接把 CoM 当作直射束中心。
2. 对抽样 DP 估计直射束中心：中央搜索窗口 + 高斯平滑仅作为初值；对候选强斑局部进行质心或二维峰拟合，并记录峰强、宽度、是否靠近搜索边缘等质量分数。
3. 分别拟合 `origin_x=a0+a1*scan_x+a2*scan_y` 和 `origin_y=b0+b1*scan_x+b2*scan_y`。采用稳健拟合（如 Huber / `soft_l1`），标识离群点并保留原测量值。
4. 保存 `residual_x = measured_x - fitted_x`、`residual_y`、二维残差 `hypot(...)`，绘制直方图、残差热图、有效点比例。
5. 用**未参与拟合**的扫描位置（例如每个扫描象限再选 10–20 张 DP）独立验证拟合预测；同时分别检查晶粒内部与晶界附近的拟合偏差。
6. 如残差呈明显空间弯曲，可与二阶面比较；优先使用更简单、能通过独立验证的模型，避免把真实局部偏转过拟合为仪器偏移。

**建议起始 QC 规则（可随束斑尺寸调整）：**
- 每个抽样点需要记录 `valid/invalid`，局部异常点不应被误当成全局梯度。
- 独立验证集原点位置误差的中位数目标 ≤ 2 px，95 分位目标 ≤ 5 px；它们是工程试运行阈值，不是物理普适精度。
- 若误差超过直射束盘半径的明显比例，暂停对衍射峰坐标的高精度校正。
- 误差地图不应保留连续的大块系统性趋势；若仍有趋势，重新分析模型，而不是仅提高拟合阶数。

**交付：** `P1_origin/origin_measured_sampled.npz`、`origin_model.npz`（全场 256×256 的拟合原点）、`origin_residual_maps.png`、`origin_qc.json`、`origin_validation.png`。

**Gate 1：** 原点测量、离群点、拟合误差和独立验证都有文件记录；明确保留原始 CoM/原点，不直接对原始 MIB 执行位移写回。

## 4. P2：动态虚拟成像与对齐质量验证（0.5–1 天）

**目标：** 确认平面模型足以校正导致 BF/ADF 假圆斑的系统性偏移，同时保留真实组织衬度。

**实施：**
1. 读取 `P1_origin/origin_model.npz`，将每个扫描位置虚拟 BF/ADF 掩膜的圆心设置为 `(origin_y[y,x], origin_x[y,x])`，**移动积分掩膜而不是平移原始 DP**。
2. 起始参数为 BF `r≤15 px`、ADF `18≤r<50 px`；用径向强度分布和对齐后的中心盘尺寸作敏感性测试，例如 BF 半径 10/15/20 px，ADF 18–50/25–60 px。
3. 保存 BF/ADF 的原始总计数、Total、归一化 BF/Total 与 ADF/Total；比较校正前后，避免仅凭独立自动拉伸的图片断言定量改善。
4. 使用抽样 DP 做校正前/后平均图，记录中心盘峰位置和宽度、中央峰对齐误差、峰清晰度；抽样集合固定以保证可比性。
5. 对照 Total / CoM / BF / ADF：大圆反差显著减弱且晶界结构保留，但不自动解释成孔洞或物理缺陷。

**交付：** `P2_virtual/bf_corrected.npy`、`adf_corrected.npy`、`bf_normalized.npy`、`adf_normalized.npy`、`bf_adf_compare.png`、`mean_dp_before_after.png`、`detector_radius_sweep.png`。

**Gate 2：** 原点跟踪对齐效果可重复；BF/ADF 不再主导于假圆形背景；拟合原点及虚拟探测器半径均有记录。即便 BF 视觉很白，也通过数值分布判断，而非仅凭 PNG。

## 5. P3：小 ROI Bragg 峰检测（1–2 天）

**选择 ROI：** 初始使用 3 个 32×32 ROI，分别覆盖单个相对均一晶粒、晶界附近、纹理/散射特征不同的区域。可以先检查下列候选位置：`(32:64,32:64)`、`(112:144,112:144)`、`(192:224,160:192)`；最终 ROI 必须根据校正 ADF 与对应单点 DP 调整，不能机械采用示例坐标。

**执行：**
1. `roi = data[y0:y1, x0:x1].compute().astype(np.uint16)`；构造 `py4DSTEM.DataCube(data=roi)`。仅小 ROI 转换为本机字节序。
2. 首先在每个 ROI 抽取 10–20 张 DP，检查直射束盘半径、孤立坏点、衍射峰重叠、最亮峰是否为直射束。必要时建立坏点掩膜。
3. 先调用 `inspect.signature(dc.find_Bragg_disks)` 核对本地 **py4DSTEM 0.14.18**；不要照抄其他版本不存在的参数。`find_Bragg_disks()` 支持逐个位置或指定位置检测；先调单张，再扩大 ROI。
4. 对照两种检测：`template=None` 的原始峰值基线；使用经确认的直射束盘/探针模板进行互相关检测。若无真空区，不能将包含晶格峰的平均 DP 当作“真空探针”。
5. 起始试验参数：`corrPower=1.0`、`sigma=1`、`minRelativeIntensity` 取 0.03/0.05/0.10、`minPeakSpacing` 取 6/8/10 px、`edgeBoundary` 取 10/20 px，`subpixel='poly'` 优先于代价更高的高倍上采样。参数属于起点，不是最终结论。
6. 绘制单点 DP 与检测峰叠加图；记录每张 DP 的峰数、峰强、中心盘误检、噪声区假峰和漏检；不直接对整个 256×256 扫描运行 `find_Bragg_disks()`。

**交付：** `P3_bragg/roi_manifest.csv`、`bragg_raw_roi_<id>.npz`（保留像素原坐标）、`bragg_overlays_<id>.png`、`peak_count_map_<id>.png`、`bragg_param_sweep.csv`。

**Gate 3：** 每个 ROI 随机/分层选择至少 20 张 DP 人工核验；有可解释的检测质量统计，可靠晶粒内部的主要峰不被大量漏检，背景假峰可控；不以峰数越多越好。

## 6. P4：Bragg 峰坐标校正与跨 ROI 合并（1–2 天）

**原则：** 不修改原始 DP；在检测峰的坐标层减去系统性参考原点。

对全局扫描位置 `(y,x)` 和原始探测器峰 `(qx_raw,qy_raw)`：

```text
qx_rel_px = qx_raw_px - origin_x_model[y, x]
qy_rel_px = qy_raw_px - origin_y_model[y, x]
```

**实施：**
1. py4DSTEM 在局部 ROI 计算 Bragg 峰后，将局部扫描索引 `(i,j)` 加上 ROI 起点 `(y0,x0)`，写入全局 `(scan_y,scan_x)`。
2. 统一自定义数据契约：`scan_y, scan_x, qx_raw_px, qy_raw_px, qx_rel_px, qy_rel_px, peak_intensity, peak_quality, roi_id`。明确 py4DSTEM 和 NumPy 的轴方向差异，写一个单张 DP 的人工坐标对照测试。
3. 对比校正前后的 Bragg Vector Map (BVM)；跟踪中心盘及已识别的同一反射在相邻扫描位置的坐标散布。分别检验单晶粒内部与晶界附近，避免把真正取向变化当成残差。
4. 同时保存原点模型版本、参数、残差、输入数据文件名和 ROI 坐标。不要将相对像素坐标错误标注为 `1/Å`。

**交付：** `P4_bragg_corrected/peaks_corrected_<id>.npz`、`bvm_before_after_<id>.png`、`peak_centering_qc.json`、`coordinates_spec.md`。

**Gate 4：** 中央峰归零及同一晶粒内的重复峰坐标具有可接受的稳定性；不同 ROI 拼接无固定平移差；全局扫描坐标无行列互换问题。

## 7. P5：倒易空间标定（2–4 天，依赖实验信息）

**现在缺少：** 实际 TEM 加速电压、衍射模式细节、探测器几何/像素尺寸或可信的衍射标样。文件名中的 `cl 110` 不能单独保证物理尺度精度。

**实施：**
1. 记录加速电压、相机长度、会聚半角（若可得）、探测器尺寸/几何；区分衍射盘中心与几何中心。
2. 优先使用标样或已知 `d` 间距建立 `pixel → 1/Å` 比例；严格写明约定：`g=1/d`（周期/Å）或 `q=2π/d`（弧度/Å），不可混用。
3. 如具有多条足够可靠的环或峰，检验椭圆/非线性畸变；没有证据时不要强加椭圆校正。
4. 用**未参与拟合**的反射验证尺度与几何：记录 `d_meas`、`d_ref`、相对误差、峰匹配残差及标定来源。
5. 如果无可靠标样，阶段输出暂保留 `px` 或相对 `q`，允许做定性峰空间分析，但**不将它当作完成绝对晶体学定标**。

**交付：** `P5_calibration/calibration.json`、`q_scale_validation.csv`、`ellipse_qc.png`（有条件才生成）、`calibration_report.md`。

**Gate 5：** 有独立标定来源、单位约定和验证残差；若没有，则明确标记 `uncalibrated`，暂停定量晶格常数和应变分析。

## 8. P6：相结构、取向与应变（3–5 天，条件阶段）

**目标：** 从“峰检测成功”推进到可信晶体学结果，而不强行对每一像素给相名。

**实施顺序：**
1. 建立 CIF 候选结构，明确参考晶格参数、空间群/对称性、束能和散射模型；先使用对照点验证模拟 DP 与实验峰的尺度、角度和系统消光是否一致。
2. 若样品确认为 Ti 系合金，可从 Ti-hcp / Ti-bcc 开始，并加入能产生相似投影的候选相作为混淆检验；未知材料不预设它必然是 Ti。
3. 对每个 ROI 用多个候选带轴/取向拟合；保留最佳分数、第二名分数、匹配峰数、模板可观测比例、未解释实验峰比例与重投影残差。
4. 设立 `high_confidence / ambiguous / low_peak / unindexed`，允许保持未知；注意高相关分数不等于相结构唯一确认。
5. 先输出 ROI 级相/取向图，并检查与晶粒边界、局部单点 DP 的一致性；只有通过标定和参考晶格验证后，再考虑相内应变或跨晶粒应变比较。
6. 应变分析需要统一参考晶格和足够稳定的峰位置；在畸变、坐标及相识别未验证前，不输出有物理意义的定量应变图。

**交付：** `P6_indexing/phase_map_<id>.npy`、`orientation_<id>.npz`、`confidence_<id>.npy`、`indexing_qc.csv`、`examples_exp_vs_sim.png`；满足前提时才输出 `strain_<id>.npy`。

**Gate 6：** ROI 级结果可用单点叠加图复核；误判、候选相简并与低峰数区域有明确标志；不报告未经独立验证的全场高置信度相比例。

## 9. P7：全场 256×256 扩展、恢复与交付（2–4 天）

**实施：**
1. 将扫描网格切成 `32×32` 的 64 个 tile（或内存允许时 `64×64` 的 16 个 tile）；原点模型始终使用全局坐标。
2. 单 tile 流程：MIB 延迟切片 → NumPy ROI → Bragg 检测 → 校正峰坐标 → （标定/索引）→ 保存稀疏结果 → 释放内存。
3. 实现 `--tile-size`、`--resume`、`--dry-run`、`--workers 1`、`--roi` 和 `--output`；只在测得内存有余量后增加并发。
4. 每 tile 保存 `status.json`、运行参数、异常日志、QC；成功后原子写入完成标记，失败 tile 可重试，不能让半成品被误认作成功。
5. 保存峰表 / 相/取向/置信度 / Origin / QC，不必生成另一份完整的原始 4D HDF5；若其他软件确实要求 HDF5，再按需导出。
6. 最终做晶粒边界拼接检查、tile 接缝检查、低置信度掩膜和统计汇总。

**交付：** `P7_fullscan/tiles/`、`merged_peaks.*`、`phase_map.npy`、`orientation.*`、`qc_summary.json`、`run_manifest.json`、`figures/`。

**Gate 7：** 64/64 tile 状态清晰，能断点续跑，边界无明显人为接缝；最终统计明确有效像素分母和被剔除/未索引区域。

## 10. 项目文件规划

```text
large-4dstem-analysis/
├── data/                          # 原始 MIB（只读；不纳入 Git）
├── scripts/
│   ├── 00_capture_baseline.py     # P0
│   ├── 01_origin_qc.py            # P1
│   ├── 02_virtual_corrected.py    # P2
│   ├── 03_bragg_roi.py            # P3
│   ├── 04_correct_bragg.py        # P4
│   ├── 05_calibrate_q.py          # P5
│   ├── 06_index_roi.py            # P6
│   └── 07_run_tiles.py            # P7
├── configs/
│   └── mib_3-1.yaml
├── tests/
│   ├── test_origin_fit.py
│   ├── test_coordinate_convention.py
│   ├── test_bragg_correction.py
│   └── test_tile_resume.py
└── outputs/mib_pipeline/
    ├── P0_baseline/
    ├── P1_origin/
    ├── P2_virtual/
    ├── P3_bragg/
    ├── P4_bragg_corrected/
    ├── P5_calibration/
    ├── P6_indexing/
    └── P7_fullscan/
```

**注：** `scripts/00_...py` 等是**待实现的建议文件名**，并非宣称这些脚本已存在。可将核心函数集成回项目已有的 `loaders/bragg/calibration/phase/orientation` 模块，脚本仅作为 CLI 入口。

## 11. 配置文件草案 `configs/mib_3-1.yaml`

```yaml
input:
  path: 'data/3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib'
  scan_shape: [256, 256]
  detector_shape: [256, 256]
  dtype_expected: '>u2'
  chunks: [1, 16, 256, 256]
metadata:
  scan_step_nm: 5.8
  exposure_ms: 0.62
  voltage_kv: null             # 必须从实验记录补充
  camera_length_mm: 110        # 来自文件名，待确认
origin:
  samples_stride: 8
  search_window: [64, 192, 64, 192]  # detector y0,y1,x0,x1
  model: robust_plane
  validation_points_per_quadrant: 12
  median_target_px: 2.0
  p95_target_px: 5.0
virtual:
  bf_radius_px: 15
  adf_inner_px: 18
  adf_outer_px: 50
bragg:
  roi_size: [32, 32]
  min_relative_intensity_candidates: [0.03, 0.05, 0.10]
  min_peak_spacing_candidates: [6, 8, 10]
  edge_boundary_candidates: [10, 20]
  correlation_power: 1.0
  subpixel: poly
fullscan:
  tile_size: [32, 32]
  workers: 1
output:
  root: outputs/mib_pipeline
```

## 12. 第一轮建议执行清单（先做 P1，不急于索引）

- [x] 复制并只读备份 `origin_sampled.npz`。
- [x] 完成 `01_origin_qc.py`：测量有效标志、稳健面拟合、异常点、误差热图和直方图。
- [x] 从未参与拟合的位置抽取 ≥40 张 DP（四象限各约 10–12 张）独立验证原点模型。
- [x] 输出原点模型和统一坐标契约，并运行坐标单元测试。
- [x] 用 P1 合格模型重跑动态 BF/ADF，记录半径灵敏度与数值分布。
- [x] 再启动第一个 32×32 ROI 的 Bragg 峰参数扫描。

**第一轮执行记录（2026-10-08）：** 见 [实施与验收报告](docs/mib_round1.zh-CN.md)。48 个独立验证点误差中位数 0.0591 px、P95 0.1730 px；全场动态 BF/ADF 已重算；首个 ROI 完成 18 组参数扫描。首轮清单完成不代表 P3 全阶段 Gate 3 通过，模板比较、多 ROI 和领域人员复核仍属后续工作。

**首个明确里程碑：** 产出 `origin_qc.json + origin_residual_maps.png + origin_model.npz`；下一阶段以这三份结果为可信输入，而不是只以“图像看起来更清楚”为验收标准。

## 13. API 与方法参考

- RosettaSciIO Quantum Detector MIB reader：<https://rosettasciio.readthedocs.io/en/latest/supported_formats/quantumdetector.html>（`lazy`、`chunks`、`navigation_shape`）。
- RosettaSciIO Lazy Loading：<https://rosettasciio.readthedocs.io/en/latest/user_guide/lazy.html>。
- py4DSTEM DataCube / BraggVectors API：<https://py4dstem.readthedocs.io/en/latest/api/classes.html>（`find_Bragg_disks`、`get_probe_size`、`get_vacuum_probe`、`BraggVectors.raw`）。
- py4DSTEM Origin calibration：<https://py4dstem.readthedocs.io/en/latest/api/process.html>（`fit_origin` 等）。

**版本提示：** py4DSTEM 在线文档不一定与本地 0.14.18 完全一致，真正调用前用 `inspect.signature()` 查证具体 API；项目应在环境清单中固定版本。
