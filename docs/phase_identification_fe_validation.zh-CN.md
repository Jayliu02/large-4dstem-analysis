# Fe BCC／FCC 相识别验证记录

验证日期：2026-09-24。当前入口为 `fourdstem_pipeline.phase_identification`，配置为 `configs/phase_identification.yaml`，输出为 `outputs/phase_identification_fe/`。原 Ti 结果保留为历史记录，不用于本批 Fe 相结论。旧 pyxem／取向及一致性流程未迁移。

## 结构与运行条件

- `Fe-BCC.cif`：空间群 229，展开 2 个 Fe 原子，a=2.86303550 Å。
- `Fe-FCC.cif`：空间群 225，展开 4 个 Fe 原子，a=3.65555117 Å；中性 `Fe0+` 在模拟前规范化。
- 两份文件均从用户 `Documents/Data` 原样复制，SHA-256 及来源见 [结构说明](../references/cifs/README.md)；运行审计见 `templates/structure_audit.json`。
- 每相 406 个区轴方向，步长 2°；分别生成 80／120／200／300 kV 模板。匹配阈值及束心／尺度扰动规则沿用原配置。
- 输出标签为 `0=Fe-BCC`、`1=Fe-FCC`、`-1=未索引`、`-2=歧义`。最佳候选数组包含被拒绝图样，不能直接作为确认相图。

## 回归与合成验证

完整测试：**105 通过、1 跳过**。覆盖 Fe 元素与空间群检查、BCC／FCC 消光、两相竞争、标定拒绝、扰动检查、动态标签／汇总、缓存续跑和指纹不一致拒绝。日志保存在输出目录的 `tests.log`。

```powershell
.\.conda-phase\python.exe -m pytest -q --basetemp outputs/pytest_fe_full -o cache_dir=outputs/pytest_fe_cache
.\.conda-phase\python.exe scripts/validate_phase_identification.py --calibration
```

采用偏离模板网格 0.4° 的独立区轴、随机平面内角度与束心，在 `0.009 / 0.015 / 0.025 Å⁻¹/像素` 下生成控制图样。退化条件为 0.35 像素噪声、15% 随机缺峰及两个离群点。

| 控制类型 | 数量 | 结果 |
| --- | ---: | --- |
| 纯相／退化图样 | 70 | 接受 45，错误接受 0，其余拒绝 |
| 随机角度负对照 | 70 | 全部拒绝 |
| 两相几何混合图样 | 70 | 全部拒绝 |
| 已知尺度恢复 | 0.015 Å⁻¹/像素 | 恢复为 0.01495165，误差约 0.32%；128 个留出图样全部支持 |

这些接受统计在随机对照阈值及束心／尺度扰动门槛之前计算；有限混合控制全部拒绝不等于已证明单相纯度。合成验证使用相同的运动学正向模型，不代替实测真值验证。完整记录为 `synthetic_validation.json` 和 `synthetic_calibration/`。

## 实测数据

三份原始 MIB 位于 `C:/Users/jayliu/Documents/Data`，按 `256×256` 原扫描分辨率处理。基础分析的文件名、大小、修改时间及尺寸已与原始数据核对，记录见 `input_locations.json`。Fe 模板、斑点提取、标定和匹配均在新目录重新计算；原始数据及历史输出保留。

复现或断点续跑：

```powershell
.\.conda-phase\python.exe -m fourdstem_pipeline.phase_identification --config configs/phase_identification.yaml
```

本次将三份扫描作为独立进程运行，每个进程使用 4 个计算线程，并在全部结束后统一生成汇总；普通入口按顺序执行相同算法。依赖快照保存在 `environment_freeze.txt`。

三份扫描均完成全部 65,536 个位置，共 196,608 个位置。各扫描独立估计的工作尺度均为 `0.0162937086 Å⁻¹/像素`，均通过候选结构条件下的尺度验证；这仍不是标准样品给出的仪器标定。

| 扫描 | 留出支持／有效图样 | Fe-BCC | Fe-FCC | 歧义 | 未索引 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1045 | 113／355 | 0 | 44 | 7,455 | 58,037 |
| 1049 | 105／381 | 0 | 0 | 7,083 | 58,453 |
| 1052 | 108／307 | 0 | 1 | 7,897 | 57,638 |

相计数只包含通过全部匹配、跨电压一致性及束心／尺度扰动检查的位置。合计 45 个 FCC 位置通过，不能据此将整个扫描区域判为 FCC，也不能把 BCC 的零接受数解释为不存在 BCC。其余位置保留歧义或未索引，没有平滑或插值填充。尺度验证通过不等于每个图样均能可靠判相。

最终核验检查了三份扫描的全部提取／匹配断点、源码及配置指纹、数组形状、标签范围和计数、斑点筛选索引、留出样本隔离、代表图样的一对一反射配对、报告文件与链接。新提取的七组观测数组与历史结构无关提取结果逐值一致。原始 MIB 的大小和修改时间保持不变；未声称计算过完整 MIB 的 SHA-256。详细结果见 `verification.json`，中文汇总入口为 `report_zh.html`。
