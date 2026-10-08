# MIB 第一轮实施与复现

按 `MIB_4DSTEM_分阶段实施方案.md` 第 12 节执行，范围为 P0 基线、P1 原点质控、P2 动态虚拟成像，以及第一个 ROI 的 P3 参数扫描。没有执行绝对 q 标定、相索引或应变分析。

## 执行命令

在仓库根目录使用已有 `.conda-phase` 环境（Python 3.12.14、py4DSTEM 0.14.18；本轮没有更改依赖）：

```powershell
./.conda-phase/python.exe scripts/01_origin_qc.py
./.conda-phase/python.exe scripts/02_virtual_corrected.py
./.conda-phase/python.exe scripts/03_bragg_roi.py
./.conda-phase/python.exe -m pytest tests/test_mib_round1.py -q -p no:cacheprovider
./.conda-phase/python.exe scripts/validate_mib_round1.py
```

配置为 `configs/mib_3-1.yaml`，每个阶段支持 `--config` 和 `--output`。P1 会先捕获 P0；也可单独运行 `00_capture_baseline.py`。重复运行覆盖本轮派生结果；需要保留不同实验时请使用不同的输出目录。历史 `outputs/origin_check` 不被改写，其全部文件复制到 `P0_baseline/backup`，检查 SHA-256 并设置只读。P2/P3 拒绝未通过 P1 或与 QC 哈希不一致的模型。

原始 MIB 通过 RosettaSciIO 只读内存映射、Dask 切片访问；虚拟成像每次最多读取 16 张 DP，首个 Bragg ROI 读取 32×32 张 DP（原始像素约 128 MiB）。不物化整份 4D 数据。输出目录默认 `outputs/mib_pipeline`。

## 本轮实测结果

| 项目 | 结果 |
|---|---|
| MIB | 256×256 扫描、256×256 DP、`>u2`、65,536 帧、384-byte 帧头 |
| 文件大小 | 8,615,100,416 bytes，与结构计算一致 |
| 环境检查 | `pip check` 通过；版本清单保存在 `P0_baseline/environment.txt` |
| P1 训练点 | 1,024 个；1,020 个通过测量质量规则，20 个大残差点保留并标记 |
| P1 独立验证 | 四象限各 12 个，共 48 个，均未参与拟合，全部通过测量质量规则 |
| 验证误差 | 中位数 0.0591 px，P95 0.1730 px；通过 2/5 px 工程阈值 |
| P2 | 全场 65,536 张 DP；BF 半径 10/15/20 px，ADF 环 18–50/25–60 px |
| BF/Total | 中位数 0.3715，P05–P95 为 0.2585–0.4003 |
| ADF/Total | 中位数 0.2127，P05–P95 为 0.1644–0.3246 |
| P3 ROI | `[32:64,48:80]`，经校正 ADF 和 12 张 DP 初查选取的左上方晶粒内部 |
| P3 扫描 | 18 组参数 × 20 张固定 DP，共 360 条统计；完整 ROI 另运行一组基线 |
| P3 基线 | 相对强度 0.05、峰间距 8 px、边界 10 px，共 4,892 个原始峰（含中央束） |

P1 用中央窗口平滑峰作为初值，再对原始局部像素作扣背景质心测量；记录峰强、背景、信噪比、宽度、窗口边缘标记。平面拟合使用 soft_l1 损失，保留所有质量有效训练点，包括残差离群点，不根据模型吻合程度筛除独立验证点。

残差图显示局部异常簇；去除已标记异常后仍有约亚像素级局部结构，不能据此宣称所有区域都具有 0.06 px 的真实物理精度。右下象限验证中位误差约 0.113 px，高于其他象限。当前平面通过工程阈值，因此未提升到二阶模型；未分离仪器漂移与真实束偏转，也尚无人工晶界标签用于分层精度结论。

P2 对原始 DP 移动积分掩膜，保存原始计数及 Total、原始 CoM 和归一化结果；CoM 不参与原点拟合。共享显示范围的前后图显示大圆形背景明显减弱，晶粒边界保留。平均 DP 比较使用固定的 48 个验证位置；显示用平移插值保留总计数约 97.92%，边缘裁切损失有记录，不用于 BF/ADF 积分。中央 128×128 区域 RMS 半径从 39.42 px 变为 26.91 px，包含 Bragg 散射，不能当作直射束盘半径或独立物理精度。

## 坐标和本地 API 修正

统一数组顺序为 `(scan_y,scan_x,det_y,det_x)`，x 是列，y 是行，单位都是探测器像素。完整说明见 `P1_origin/coordinates_spec.md`。py4DSTEM 对未转置数组的 `qx` 实际对应第一探测器轴（行），所以导出时 `qx_raw_px=raw.qy`、`qy_raw_px=raw.qx`。已用已知非对称峰和非方形 DP 实测验证。

本地 py4DSTEM 的 `_find_Bragg_disks_single` 在 `template=None` 时设置 `cc=DP`，随后仍执行 `ifft2(cc)`，导致直接调用并非原始峰检测。兼容函数检查本地实现，必要时通过公开 `filter_function=np.fft.fft2` 参数抵消这次 IFFT；不修改安装包。已知峰测试验证修正后的峰坐标。此修正只用于无模板基线。

## 交付及下一阶段边界

首要里程碑是 `P1_origin/origin_qc.json`、`origin_residual_maps.png` 和 `origin_model.npz`。此外交付原始测量及验证 NPZ、残差直方图、独立验证叠加图、可靠点残差细节图，以及完整 P2 数组/半径扫描图、P3 峰表/参数扫描表/20 张叠加图。`run_manifest.json` 记录产物、源码和配置哈希，并校验原结果不变、模型一致、验证集隔离、归一化和峰坐标。

第一轮六项清单已执行。**P3 全阶段 Gate 3 尚未通过**：目前只有一个 ROI 的无模板基线；0.05 阈值叠加图中可见部分弱峰漏检，0.10 阈值漏检更明显，不能以峰数作为质量结论。还需对比可信探针模板、补充晶界和不同纹理 ROI，并由领域人员核验假峰/漏峰。没有将包含晶格峰的平均 DP 冒充真空探针。

扫描方向/蛇形顺序、加速电压和绝对倒易尺度仍未确认。文件名中的 CL 110 仅记录为单位未确认的名义值。所有衍射坐标保留 px，不给出定量相、取向或应变结论。
