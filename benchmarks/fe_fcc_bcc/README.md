# Fe BCC/FCC 合成 4D-STEM 基准（v1）

按 `docs/Fe_FCC_BCC_4DSTEM_MVP_PLAN.md` 与已批准计划（`toasty-juggling-iverson.md`）构建的
带已知真值的合成基准，用于逐环节诊断相鉴定流程
（`python -m fourdstem_pipeline.phase_identification`）在真实数据上仅接受 ~0.023% 图样的原因。

## 环境

- 解释器：`.\.conda-phase\python.exe`（Python 3.12，py4DSTEM 0.14.18，pymatgen 2026.9）
- 所有命令从仓库根目录执行
- 生产代码零改动；基准只新增 `benchmarks/fe_fcc_bcc/` 代码与一个新配置

## 命令

```powershell
# 0. 前置（须全绿）
python -m pytest tests -q
# 1. CIF 审计 + 独立消光规则核对（BCC h+k+l 偶 / FCC 同奇偶，容差 1e-6）
python benchmarks/fe_fcc_bcc/inspect_cifs.py
# 2. 独立运动学参考花样（pymatgen TEMCalculator，非 py4DSTEM 引擎）
python benchmarks/fe_fcc_bcc/sim_reference.py --template-dir outputs/phase_identification_fe/templates
# 3. 数据集构建（跑两遍；第二遍 --check 验哈希与真值复现）
python benchmarks/fe_fcc_bcc/build_dataset.py
python benchmarks/fe_fcc_bcc/build_dataset.py --check
# 4. basic_analysis（T01/T02 单花样扫描会因 KMeans 空类别崩溃 → 用 fallback，见下）
python -m fourdstem_pipeline.basic_analysis --data benchmarks/Fe_FCC_BCC_v1/datasets/T03 --output benchmarks/Fe_FCC_BCC_v1/basic/T03 --scan-shape 32 32
python benchmarks/fe_fcc_bcc/build_dataset.py --basic-artifacts   # 为缺 basic 产物的数据集写零运动 beam_motion.json + 均值图
# 5. 冻结 phase CLI 配置
python benchmarks/fe_fcc_bcc/build_dataset.py --write-phase-configs v1
# 6. phase CLI（每数据集；共享 output_dir；每跑约 2-5 分钟模板生成）
python -m fourdstem_pipeline.phase_identification --config benchmarks/Fe_FCC_BCC_v1/runs/v1/configs/T01.yaml --stage all
python -m fourdstem_pipeline.phase_identification --config benchmarks/Fe_FCC_BCC_v1/runs/v1/configs/T02.yaml --stage all
python -m fourdstem_pipeline.phase_identification --config benchmarks/Fe_FCC_BCC_v1/runs/v1/configs/T03.yaml --stage all
# 7. 评估
python benchmarks/fe_fcc_bcc/evaluate.py --run-dir benchmarks/Fe_FCC_BCC_v1/runs/v1
```

## 坐标契约（config.yaml 冻结）

- `q_x = (col − c_x)·Δq`，`q_y = s_y·(row − c_y)·Δq`，**s_y = +1**
- Δq = 0.020 Å⁻¹/px，中心 (row, col) = (127.5, 127.5)（像素边界中心），256×256 探测器，300 kV
- 数据轴序 `(scan_y, scan_x, det_y, det_x)`；斑坐标 (y, x)
- 强度模型：直射束 3500、最强斑 1200，斑幅 = 1200×I_norm；**最弱保留斑 ≥ 300 counts**
  （≥10× 检测下限：`find_pattern_peaks` 阈值 = 0.008×直射束平滑信号 ≈ 28 counts）
- 无噪声、无随机数，uint16 无削顶（全局 max ≤ 4094）
- 晶体→实验室：`R0(zone)` 为最小扭转旋转使 `R0 @ zone = ẑ`；斑 `g` 出现在
  `qxy = (R0 @ g)[:2]`；面内角 θ：`R(θ) = Rz(θ) @ R0`（逆时针）。
  已对 [001]/[011]/[111] 四种取向与流程模板库逐斑验证（0.00 mÅ 残差）

## 数据集

| 数据集 | 扫描 | 内容 |
|---|---|---|
| T01 | 16×16 | 单相 Fe-BCC [001]，0°，全部 256 图样相同 |
| T02 | 16×16 | 单相 Fe-FCC [001]，0°，全部 256 图样相同 |
| T03 | 32×32 | 四晶粒：BCC[001]/0°、BCC[011]/25°、FCC[001]/10°、FCC[111]/35°，各 16×16 象限 |

每个方向保留的参考斑数 ≤ 14（≥ 25% 最强斑强度），远低于流程 `peaks.max_peaks=48`，
因此召回率分母 = 全部真值斑，无检测上限截断。

## 预注册指标（evaluate.py 输出 validation_report.json）

| 指标 | 门 |
|---|---|
| Bragg 召回率（真值斑 vs `peaks/peak_yx.npy`，1.5 px 贪婪一对一） | ≥ 90% |
| 位置误差中位数 / p90 | ≤ 1 px / ≤ 2 px |
| 束心误差中位数（`peaks/center_yx.npy` vs 127.5） | ≤ 1 px |
| 尺度误差（`calibration.json` vs 0.020 Å⁻¹/px，status=calibrated_conditionally） | ≤ 1% |
| T01/T02 正确率 / 拒判率 | ≥ 95% / ≤ 5% |
| T03 总体正确率 / 每区正确率 | ≥ 95% / ≥ 90% |
| 取向（信息性） | 先过自洽门（真值斑 vs 模板库 ≤ 0.05 px 可表示），再报 misorientation 中位 ≤ 2°、面内 ≤ 1° |

## 偏离记录（相对 MVP 计划原文）

1. **参考花样源**：ReciPro 未安装 → 用仓内独立运动学模拟器 `sim_reference.py`
   （pymatgen `TEMCalculator`：Mott-Bethe 电子散射因子 + 相对论波长，ZOLZ 由
   beam_direction 选取），不经 py4DSTEM Crystal 引擎。替代原因与条件记录在
   `reference/sim_reference_manifest.json`。
2. **T01/T02 扫描尺寸 16×16**（原文 8×8）：校准门 `min_validation_patterns=32`
   下 8×8 零余量；16×16 得 128/128 训练/验证划分。
3. **`sim_reference.py` 代替 SpotInfo 导出**：CSV 列
   `h,k,l,qx_A,qy_A,d_hkl_A,intensity_norm,amplitude,observable` + log1p PNG 预览。
4. **MIB 交付格式**：交付真实 384 字节帧头模板（取自 `data/` 首个 .mib，仅重写序列号
   字节 [4:10]）+ `>u2` 大端 payload 的单片 MIB；`inspect_mib` 校验通过、零生产代码改动。
   同时交付 HDF5（`/datacube/data` uint16 + `/metadata`）作规范副本，MIB 与 H5 载荷
   sha256 互证（`build_manifest.json`）。

## 非独立性声明

参考生成器与相鉴定侧（py4DSTEM Crystal 模板）共用同一 CIF 文件、同一 pymatgen CIF
解析器与同一运动学 |F|² 物理。二者在结构因子实现（pymatgen Mott-Bethe vs py4DSTEM）、
取向机制与全部栅格化/量化（本仓库实现）上独立。两个运动学引擎或 pymatgen CIF 解析共有的
缺陷可能逃逸本基准；`inspect_cifs.py` 的解析消光规则是兜底。斑的几何位置是精确点阵量，
并已在有模板库处逐斑验证。
