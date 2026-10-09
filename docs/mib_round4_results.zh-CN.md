# MIB 第四轮结果与人工复核入口

已按 [第四轮计划](mib_round4_plan.zh-CN.md) 执行真实背景注入实验，并生成可离线操作的人工复核页面。第三轮检测器保持冻结；前三轮产物哈希未改变，没有新增实验峰表或物理标定结论。

## 注入回收结果

使用第三轮固定的 72 张复核 DP，每张选择孤立和拥挤两个位置，分别注入实测正峰模板与窄 Gaussian，峰顶期望增量为 8/16/32 counts。144 个目标位置全部可用，共完成 **864 次注入**，保存实际加入的整数计数、目标位置、随机种子和逐例匹配结果。

| 条件 | 形状 | 8 counts | 16 counts | 32 counts |
|---|---|---:|---:|---:|
| 孤立 | 经验正峰模板 | 69/72（95.8%） | 72/72（100%） | 72/72（100%） |
| 孤立 | Gaussian，σ=2 px | 42/72（58.3%） | 72/72（100%） | 72/72（100%） |
| 距已有非中央峰约 10 px | 经验正峰模板 | 16/72（22.2%） | 39/72（54.2%） | 52/72（72.2%） |
| 距已有非中央峰约 10 px | Gaussian，σ=2 px | 3/72（4.2%） | 16/72（22.2%） | 35/72（48.6%） |

三个 ROI 的强注入孤立位置均超过运行前设定的 90% 阈值，诊断检查通过。但**拥挤峰仍明显漏检**：即使 32 counts 增量，Gaussian 仅回收 48.6%，经验正峰模板仅回收 72.2%。不能把孤立峰的通过结论推广到拥挤衍射峰，也不能将第三轮简化合成数据的完美检出当成真实实验准确率。

原始 DP 在注入前被重算，结果与第三轮峰表一致。同位置原图若已有检出，不计为新增回收；原始计数不被改写。每次试验仅加入一个新的 Poisson 信号，原始背景在多个条件间复用；因此试验存在相关性，不按 864 个完全独立实验计算精度置信区间。相同峰顶增量下两种形状的总计数不同，不能据此孤立推断“峰宽导致差异”。本轮未按结果调参。

## 使用离线复核页面

在浏览器打开 `outputs/mib_round4/review.html`。页面含 72 张固定 DP、第三轮 1,034 个检出峰、局部 SNR 与半计数支持标记，图像全部内嵌，无外部资源或联网依赖。

1. 点击峰圈或使用右侧下拉框，标为有效、假峰或不确定。初始均为未复核。“当前图全部标为有效”是显式操作，会覆盖本图已有逐峰标记。
2. Shift+点击图像补记主要漏峰；可删除误加标记。log/linear 切换仅改变显示。
3. 检查全部检出峰及主要漏峰后，勾选该 DP 完成；未处理的峰会阻止完成标记。
4. 填写可追溯的复核人标识，导出 JSON。本地草稿依赖当前浏览器，换设备或清理浏览器前应导出。
5. 用 CLI 导入，验证来源并生成审计记录：

```powershell
./.conda-phase/python.exe scripts/06_mib_round4.py --import-review <导出的JSON路径>
```

JSON 绑定 `bundle_id`、DP ID、峰 ID 和固定源产物。导入拒绝错误来源、重复/未知 ID、缺失峰、越界漏峰坐标及“已完成但仍有未复核峰”的记录。标注者身份与时间为填写者声明，并非独立认证。原始导入内容与决定保存在 `reviews/<sha256>/`，`current_review.json` 指向当前有效记录。

验收规则运行前固定：每 ROI 至少 20 张完整复核 DP，无不确定标记，假峰比例 ≤5%，每张完整 DP 的主要漏峰平均数 ≤0.5。完成但超阈值记为 failed，覆盖不足记为 pending。只有人工规则和注入诊断均通过时，这一复核工作流才将 Gate 3 置为通过；校准状态始终独立判断。复核注入未检验的形状、取向和重叠情况仍需领域判断。

## 当前状态与验证

本轮没有伪造人工标注或替用户确认峰，因此 **Gate 3 仍为 pending**。根据此前明确的“暂无标定”，**Gate 5 仍未通过、坐标仍为 detector_px**。没有运行全场物相、取向或应变分析。

29 项测试通过，包括计数注入守恒、零注入、原始数据不变、来源绑定与错误标注拒绝、pending/failed/passed 区分，以及人工通过也不能绕过失败诊断的审计测试。页面 JavaScript 的标记、完成保护、漏峰添加、导出和来源检查通过 Node 事件级测试；未做完整浏览器截图测试。真实产物校验通过：864 个增量记录与案例一致，72 张复核 DP 来源固定，前三轮保持不变。

```powershell
./.conda-phase/python.exe scripts/06_mib_round4.py
./.conda-phase/python.exe scripts/06_mib_round4.py --verify
./.conda-phase/python.exe -m pytest tests/test_mib_round1.py tests/test_mib_round2.py tests/test_mib_round3.py tests/test_mib_round4.py -q -p no:cacheprovider
```

配置为 `configs/mib_round4.yaml`。使用 `--output` 保存不同试验。已有导入复核时禁止原地重跑；使用 `--verify` 检查或另选输出目录。

重点产物：`review.html`、`review_bundle.json`、`injection/recovery_curves.png`、`injection/recovery_qc.json`、`injection/injection_cases.json`、`injection/added_counts.npz`、`gate_status.json`、`run_manifest.json` 和 `report_zh.md`。初始 `gate_status.json` 记录交付时的 pending 状态；若后来导入人工复核，以经校验的 `current_review.json` 为当前复核结果。

下一步先完成页面中的实测复核，重点关注相邻峰和最弱斑。若继续改进检测器，应针对本轮暴露的拥挤峰分离问题建立新的开发/验收分组，不在已经查看的 72 张图上反复优化后宣称独立精度。绝对尺度仍需独立标定来源。
