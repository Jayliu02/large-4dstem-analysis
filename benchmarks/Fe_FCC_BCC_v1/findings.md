# Fe BCC/FCC 合成基准验证发现（v1，2026-09-25）

基准定义与坐标契约见 `benchmarks/fe_fcc_bcc/README.md`；门结果见
`runs/v2/validation_report.json`。本文件记录全部发现，**未对生产流程代码做任何修改**
（仅基准生成器/评估器）。

## 结论摘要

- **T03（四晶粒双相 32×32）端到端通过全部 14 个门**：标定 0.019938 Å⁻¹/px
  （真值 0.020，误差 0.3%，`calibrated_conditionally`）、Bragg 召回 1.0、位置误差
  中位数 0.0011 px / p90 0.0019 px、束心误差 0 px、1024/1024 相识别正确、0 拒判、
  逐区（256 像素/区）正确率 1.0。
- **T01/T02（单相均匀 16×16）校准门按设计拒判**（发现 1，预注册风险 #2，数学性质，
  非缺陷）；位置/召回/中心/尺度门全部通过。
- 取向信息性指标：面内角误差中位数 0.016°（目标 ≤1° 通过）；错向中位数 3.69°
  （目标 ≤2° 未达，成因 = 发现 5 + 发现 6，数据/算法语义所限，非缺陷）。

---

## 发现 1：单相均匀扫描的标定尺度退化（T01/T02，bit 16）—— 数学性质

BCC 倒空间花样在 ×√2 标度下闭合（每个 BCC 环映射到其他 BCC 环），FCC 在 ×2 标度下
闭合。因此单相均匀扫描的校准得分地形**多模态且峰值近似等高**：

- T01 实测：最优 0.01994 Å⁻¹/px 得分 0.7992，竞争者（1.106×）得分 0.7975，
  分离 0.0021 < 门限 0.03 → `assess_scale` 分离门拒绝 → 状态 `uncalibrated`，
  全部 256 图样拒判 bit 16（尺度项本身 0.3% 误差，通过）。

这是计划预注册风险 #2：**如实报告，不调参**。含义：本管线对单相均匀扫描的校准判定
为"不可单独标定"，这是保守但正确的行为；多相/多取向扫描（如 T03、真实多晶数据）不受影响。

## 发现 2：像素对齐对称斑的重复检测击穿覆盖率门（v1 T03 BCC[001]，0°）

束心 (127.5, 127.5) 恰在像素边界上，轴向斑落在半整数像素坐标；9×9 局部最大滤波在
对称平顶处把**同一个斑检测两次** → 检测数 nobs=12（真值 8），匹配覆盖率 8/12 < 0.70
→ 全区 100% 拒判 bit 2（v1）。

按计划 §5 只改基准生成器：T03 BCC[001] 面内角 0°→10°（物理任意，仅打破像素对齐
退化；见 `benchmarks/fe_fcc_bcc/config.yaml` 内注释）。v2 全部通过。

**对真实数据的启示**：高对称花样在像素对齐/接近对齐时，重复检测会使
`minimum_observed_fraction` 门的覆盖率为 8/12≈0.67 而拒判——真实数据中的此类
拒判可按此模式排查（本次真实数据重跑见 `outputs/phase_identification_fe_v2` 诊断）。

## 发现 3：basic_analysis 未崩溃（计划风险 1 未触发）

T01/T02 的 256 个全同图样未触发 KMeans 空类别 ValueError（工作流引擎已处理退化
聚类），`beam_motion.json` 正常写出（零运动，截距 127.5/127.5）。`build_dataset.py
--basic-artifacts` fallback 未使用。

## 发现 4：匹配器的 (y,x)/(x,y) 坐标交换（被花样镜像对称吸收）

`phase_matching.py` 的 `score_pose` 把模板 npz 存储的 `qxy`（列序为 (x,y)，来自
py4DSTEM `generate_diffraction_pattern` 的 qx/qy）当作 (y,x) 使用
（`predicted[j,0]=y`、`predicted[j,1]=x`）。因此管线报告位姿在 (x,y) 平面上的有效映射为

    q_obs = P · Rz2(angle) · diag(1, mirror) · q_t，  P = x↔y 交换（improper）

而非直接的 `Rz2(angle)·diag(1,mirror)`。交换的镜像部分被观测斑集合的镜面对称
（Friedel 对 + 本基准全部四个花样的具体集合对称）吸收——**得分与相识别不受影响**
（T03 全通过为证），但 `all_results.npy` 报告的 `in_plane_angle_rad` /
`transverse_mirror` 语义含一个 P 交换：直接按 (x,y) 约定解释面内角时会差一个
对角线镜像。`evaluate.py` 的取向重建已按上式修正（见 `evaluate.py` 内注释）。

**含义**：下游若使用报告的 angle/mirror 做取向测定，需先应用 P 交换；建议在管线
文档/`array_schema.json` 中记录该约定（本基准不改生产代码）。

## 发现 5：平分地形——模板行报告的取向分辨率 ~4°

2.5 px 匹配容差下，带轴倾斜 ~4° 的相邻模板行产生的斑位移 <0.06 px，得分与真值行
完全相同（0.9981 平齐）。`fit_shortlist` 的严格大于覆盖保留**首个找到**的平分行，
报告行与真值带轴差 2.6–4.1°（中位数 3.69°，逐 pattern：BCC[001] 4.05°、BCC[011]
2.61°、FCC[001] 3.33°），行一致性 0/1024。面内角不受影响（0.016°）。

**含义**：`template_index` 不是 ≤4° 精度的取向测定工具；它是评分描述符之一。
如需精确取向，需在平分邻域内做细化（超出本次验证范围）。

## 发现 6：FCC[111] {220} 六边形的 Σ3 面内歧义（60°）

FCC[111] 的 {220} 环在探测器上是 6 重对称的六边形，而晶体沿 [111] 只有 3 重对称
（Friedel 反演使 2D 花样升为 6 重）。面内 60° 旋转（= 绕 [111] 的 60° 晶体旋转，
即 Σ3 孪晶取向）产生的花样与真值完全相同：管线报告位姿 2D 6/6 匹配（得分 0.998），
与真值 3D 错向 60.12°。**任何管线都无法从该花样区分这两个取向**——属数据极限，
非缺陷。报告的 angle 语义仍正确（在 60° 等价类内）。

## 生成器侧修正记录（未改生产代码）

| 项 | 原 | 现 | 原因 |
|---|---|---|---|
| YAML `mib_suffix` | 未加引号（`0003`→int 3） | 引号 `'0003'` | 文件名 `-0003.mib` 与 scan 命名一致 |
| T03 BCC[001] 面内角 | 0° | 10° | 发现 2 的像素对齐退化 |
| 强度模型 | 0.025 / 30.0 | 0.25 / 300.0 | 对齐计划 §强度模型（最弱斑 ≥300 counts ≥10× 检测底限） |
| 参考源 | ReciPro（未安装） | `sim_reference.py`（pymatgen TEMCalculator） | 计划暂停规则记录替代 |

## 复现性

- 数据集两跑 `--check` 哈希一致（确定性）；T01/T02 重建后载荷哈希不变。
- 全程无随机数（v1 基准无噪声层）。
- run 清单：`runs/v1/run_manifest.json`、`runs/v2/run_manifest.json`。

## 真实数据重跑（2026-09-25，复现历史运行）

用 `configs/phase_identification_v2.yaml`（仅改 `data.directory: data` 与输出目录，其余与
原配置逐字节一致）对 `data/` 中 3 个真实 .mib 扫描重跑全部阶段（prepare → 逐 scan
extract → identify → phase_diagnostics `--replay --shortlist-sample 256`），与历史
`outputs/phase_identification_fe/` 对比：

| 产物 | 结果 |
|---|---|
| 60 个 .npy + 11 个 .npz（含 8 个模板库、all_results、peaks、标定采样） | **逐位相同**（`np.array_equal`） |
| `calibration.json` ×3 | 完全相同：0.016293708591051115 Å⁻¹/px，`calibrated_conditionally` |
| 接受图样数 | 44 + 0 + 1 = **45/196,608**（scan_02_1049 为 0）——与历史精确一致 |
| `batch_summary.json` / `phase_summary.json` | 数值全同（仅 `file` 路径因数据目录迁移不同） |
| `gate_diagnostics.json` ×3（含 perturbations、shortlist_sample 共 12 键） | 全同（忽略路径/scan 名） |
| `perturbation_replay.npz` ×3、`phase_comparison.csv`、`report_zh.md` | 全同 |
| 收尾 `pytest tests` | 108 passed, 1 skipped |

执行备注：identify 阶段实际 ~10 分钟（~370 图样/秒），远快于"数小时"的估计；三个 extract
若并行会耗尽系统内存（~2.8 GB/进程 ×3），需串行。合成基准（T01–T03）结论与本次真实数据
重跑共同完成计划全部验证步骤，生产流程代码零改动。

## 门结果汇总

| 数据集 | 通过 | 失败 | 失败门 | 性质 |
|---|---|---|---|---|
| T01 (BCC[001] 16×16) | 8 | 3 | calibration_status, acceptance_correct, acceptance_rejected | 发现 1，按设计 |
| T02 (FCC[001] 16×16) | 8 | 3 | 同上 | 发现 1，按设计 |
| T03 (四晶粒 32×32) | 14 | 0 | — | — |

整体报告 `overall FAIL (6 failing gates)` 即 T01/T02 的 6 个按设计失败项。
