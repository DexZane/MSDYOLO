# MSDYOLO 命名迁移、P0-A.1 与历轮审核综合报告

日期：2026-07-31  
审核对象：Claude Code  
代码基线：`e54112f`，标签 `p0a-complete`  
本轮范围：连续小写一次性迁移、P0-A.1 独立组件、文档合并与本地验证

## 一、结论

本轮把 MSDYOLO 自有代码迁移为用户指定的“连续小写”项目规范，同时完成了 P0-A.1 的独立技术闭环：

- 单模型、清晰视图 stop-gradient 教师和退化视图学生的边界明确；
- 真实 Detect raw output 可微稀疏解码已实现；
- 教师、学生和 GT 使用独立索引并保持一对一；
- 可检测性由退化后 GT 像素短边、教师置信度、传感器退化和角度可靠性共同决定；
- 分类、中心、尺度和角度使用不同衰减强度和不同损失；
- Shapely 成为明确依赖，不再静默回退到不准确的水平框近似；
- P0-A.1 仍是独立组件，尚未接入训练包装器，推理结构未改变。

## 二、命名规范

### 2.1 强制规则

| 类型 | 规则 | 示例 |
|---|---|---|
| Python 模块 | 连续小写 | `clearbranch.py` |
| 函数和方法 | 连续小写 | `processbatch` |
| 变量和参数 | 连续小写 | `teacherconfidence` |
| 实例属性 | 连续小写 | `self.clearbranch` |
| YAML 键 | 连续小写 | `batchsize` |
| 类 | PascalCase | `DistillationMatch` |
| 多词常量 | 连续大写 | `SHAPELYAVAILABLE` |
| 文档、配置、日志 | 小写连字符 | `msdyolo-baseline.yaml` |

不使用 snake case，也不使用 `batchSize` 式驼峰。

### 2.2 例外

- Python 强制形式：`__init__.py`、`__name__`、`__file__` 等；
- YOLO、DOTA 和第三方接口：例如 `state_dict`、`non_blocking`、`create_dataloader`；
- pytest 注入的外部 fixture，例如 `tmp_path`；
- 上游 `utils/nms_rotated` 编译扩展及其他未由 MSDYOLO 新增的代码；
- DOTA 数据目录约定 `labelTxt`。

### 2.3 网上标准核对

官方 PEP 8：

- 模块和包名建议使用短小写名称，必要时可用下划线；
- 函数和变量推荐 snake case；
- 类名推荐 CapWords；
- PEP 8 同时明确项目自己的风格约定可以在该项目内优先。

来源：<https://peps.python.org/pep-0008/>

因此，连续小写不是 PEP 8 对函数和变量的推荐，而是用户明确指定的项目规范。报告和代码不得把它包装成 Python 官方标准。

pytest 官方支持通过 `python_files`、`python_classes` 和 `python_functions` 自定义收集规则，因此以下配置是受官方机制支持的：

```ini
python_files = check*.py
python_classes = Check*
python_functions = check*
```

来源：<https://docs.pytest.org/en/stable/example/pythoncollection.html>

## 三、完整迁移映射

### 3.1 文件和目录

| 旧名称 | 新名称或处理 |
|---|---|
| `utils/cbranch.py` | `utils/clearbranch.py` |
| `utils/nms_rotated_pure.py` | `utils/rotatednms.py` |
| `tests/test_all.py` | `tests/checkall.py` |
| `tests/test_baseline.py` | `tests/checkbaseline.py` |
| `tests/test_p0.py` | `tests/checkp0.py` |
| `tests/test_rotated_iou.py` | `tests/checkrotatediou.py` |
| 无 | `tests/checkp0a1.py` |
| 无 | `tests/checknaming.py` |
| `data/dota_test.yaml` | `data/dota-test.yaml` |
| `data/dota_test/` | `data/dota-test/` |
| `test_000.*` 至 `test_004.*` | `test000.*` 至 `test004.*` |
| `docs/cp4pre_techdef.md` | `docs/cp4pre-techdef.md` |
| 无 | `docs/claude-review.md` |
| CP4 v0.1、v0.2 | 删除，v0.4 为唯一有效版本 |
| `docs/gpt12_submission.md` | 删除，内容整合到本报告 |
| `docs/p0a_completion_report.md` | 删除，内容整合到本报告 |
| 已跟踪 cache、pyc、旧日志 | 删除并补充忽略规则 |

保留的配置只有：

- `configs/msdyolo-baseline.yaml`
- `configs/msdyolo-degradation.yaml`
- `configs/msdyolo-clearbranch.yaml`

旧重复配置和未实现的 full 配置已删除。

### 3.2 主要 Python 符号

| 旧名称 | 新名称 |
|---|---|
| `train_one_batch` | `trainonebatch` |
| `run_demo_mode` | `rundemo` |
| `_save_module_states` | `moduletrainingstates` |
| `_restore_module_states` | `restoretrainingstates` |
| `forward_clear_branch` | `forwardclearbranch` |
| `forward_degraded_branch` | `forwarddegradedbranch` |
| `_get_default_config` | `defaultconfig` |
| `load_from_file` | `loadfromfile` |
| `_recursive_update` | `recursiveupdate` |
| `save_to_file` | `savetofile` |
| `apply_ablation_mode` | `applyablationmode` |
| `_create_gaussian_kernel` | `creategaussiankernel` |
| `_apply_noise` | `applynoise` |
| `get_config` | `getconfig` |
| `extract_from_yolo_output` | `extractfromyolooutput` |
| `_empty_predictions` | `emptypredictions` |
| `compute_center_distance` | `computecenterdistance` |
| `match_predictions` | `matchpredictions` |
| `_setup_modules` | `setupmodules` |
| `process_batch` | `processbatch` |
| `is_baseline_mode` | `isbaselinemode` |
| `measure_inference_speed` | `measureinferencespeed` |
| `profile_batch` | `profilebatch` |
| `get_average_stats` | `getaveragestats` |
| `save_stats` | `savestats` |
| `box_iou_rotated` | `rotatediou` |
| `obb_nms_python` | `rotatednms` |
| `obb_nms_per_class` | `classwiserotatednms` |
| `SHAPELY_AVAILABLE` | `SHAPELYAVAILABLE` |

旧 `YOLOOutputDecoder` 被 P0-A.1 的 `SparsePredictions` 和 `decodesparse` 取代，不保留兼容别名。

### 3.3 参数、属性和 YAML 键

| 旧名称 | 新名称 |
|---|---|
| `batch_size` | `batchsize` |
| `img_size` | `imagesize` |
| `config_path` | `configpath` |
| `output_path` | `outputpath` |
| `key_path` | `keypath` |
| `kernel_size` | `kernelsize` |
| `enable_psf` | `enablepsf` |
| `enable_downsample` | `enabledownsample` |
| `enable_noise` | `enablenoise` |
| `psf_kernel_size` | `psfkernelsize` |
| `psf_sigma` | `psfsigma` |
| `downsample_scale` | `downsamplescale` |
| `noise_type` | `noisetype` |
| `noise_level` | `noiselevel` |
| `noise_generator` | `noisegenerator` |
| `psf_blur` | `psfblur` |
| `upsample_mode` | `upsamplemode` |
| `clear_branch` | `clearbranch` |
| `extract_sparse` | `extractsparse` |
| `sparse_extractor` | `sparseextractor` |
| `module_states` | `moduletrainingstates` 返回值 |
| `top_k` | `topk` |
| `conf_threshold` | `confidencethreshold` |
| `num_classes` | `numclasses` |
| `model_training` | `modeltraining` |
| `batch_indices` | `batchindices` 或 P0-A.1 的 `batchindex` |
| `class_ids` | `classids` |
| `match_threshold` | `matchthreshold` |
| `match_strategy` | 删除；P0-A.1 固定为两阶段匹配 |
| `use_class_filter` | `useclassfilter` |
| `match_iou_threshold` | `matchiouthreshold` |
| `iou_threshold` | `iouthreshold` |
| `score_threshold` | 删除；调用前使用显式置信度筛选 |
| `device_id` | 仅保留在上游扩展接口 |
| `loss_type` | `losstype` |
| `compute_loss_fn` | `computeloss` |
| `loss_output` | `lossoutput` |
| `loss_items` | `lossitems` |
| `clear_predictions` | `clearpredictions` |
| `angle_weight` | `angleweight` |
| `detectability_weight` | `detectabilityweight` |
| `log_interval` | `loginterval` |
| `input_tensor` | `inputtensor` |
| `batch_idx` | `batchindex` |
| `batch_fn` | `batchfunction` |
| `skip_first` | `skipfirst` |
| `save_path` | `savepath` |
| `save_stats` | `savestats` |
| `output_dir` | `outputdir` |
| `ablation_mode` | `ablationmode` |
| `with_degradation` | `withdegradation` |
| `with_clear_branch` | `withclearbranch` |
| `eval_mode` | `evalmode` |
| `salt_pepper` | `saltpepper` |

CLI 保持标准外部形式：

```text
--batch-size
--img-size
--single-batch
--dry-run
```

解析后分别进入 `batchsize`、`imagesize`、`singlebatch` 和 `dryrun`。

## 四、上一轮审核内容整合

### 4.1 已解决

| 上一轮问题 | 本轮状态 |
|---|---|
| 真实 CPU 单批次是否能完成前向、损失、反向和更新 | 已在 P0-A 基线版本验证；本轮迁移后再次纳入强制验收 |
| 旋转 IoU 数值错误 | 已由 Shapely 精确多边形计算修复 |
| Shapely 未写入依赖 | 本轮加入 `Shapely>=2.0` |
| 缺少 Shapely 时静默水平框回退 | 本轮改为明确 `RuntimeError` |
| 上游 NMS 回退丢失极小框过滤 | 本轮恢复短边小于 `0.001` 的过滤并增加回归测试 |
| 教师 eval output 当作 raw list 使用 | 本轮正确解包 eval tuple |
| 教师和学生使用同一个候选索引 | 本轮由 `studentindex`、`teacherindex` 分离 |
| 教师与 GT 匹配缺少类别、置信度和旋转 IoU 闭环 | 本轮完成 |
| 学生匹配错误限制类别 | 本轮改为尺度归一化中心距离，不限制类别 |
| GT 匹配可能重复 | 本轮师、学、GT 均一对一 |
| 路由尺寸来自教师预测面积 | 本轮改用退化后的 GT 像素短边 |
| 角度可靠性缺少 CSL 熵 | 本轮加入归一化熵和长宽比双因子 |
| 空匹配零损失设备和类型错误 | 本轮由学生张量构造零损失 |
| CP4 声称已有不存在的梯度测试 | 本轮新增真实梯度测试 |
| FP16 且 `Detect.inplace=False` 时解码类型和坐标不一致 | 本轮按上游 FP32 grid/anchor 行为修复并加入真实模型测试 |
| E2 与 E3 的预期关系错误 | 本轮明确二者模型更新应相同，E3 是计算控制组 |
| E1 与 E8 重复 | 本轮重新定义消融矩阵 |
| DOTA 写为 mIoU | 本轮统一为旋转检测 mAP |
| 生产模块含临时测试函数 | 本轮全部移入 `tests/` |
| 重复配置和废弃文档 | 本轮清理并保留唯一有效版本 |

### 4.2 仍未完成

| 项目 | 状态与原因 |
|---|---|
| P0-A.1 接入 `MSDYOLOTrainer` | 有意留到下一阶段，避免独立组件与训练闭环同时变化 |
| 蒸馏总损失权重和 warmup | 尚未冻结，需要 trainer 集成实验 |
| GPU 峰值显存和吞吐量 | 本地无目标 GPU，等待上云 |
| 完整 DOTA 训练与消融 | 数据集在确定云端实验后准备 |
| 真实精度增益和负迁移结论 | 尚无实验数据，不能提前声明 |

### 4.3 本轮新增修复

- 训练入口删除了重复且不可达的优化器更新代码；
- 配置中的 `training.device` 现在实际参与设备选择；
- 非空但不存在的权重路径会明确失败；
- dry-run 会验证合并后的真实路径；
- 默认 `pytest -q` 不再被上游 doctest 模块的命令行解析中断；
- 本地数据路径和五个样本名称同步迁移；
- 增加 AST、YAML 和路径命名守卫。

## 五、P0-A.1 技术实现

### 5.1 清晰教师

同一模型先保存逐模块状态，再以 eval 和 no-grad 处理清晰视图，正确读取 eval tuple 的第二项 raw outputs。稀疏结果全部 detach，最后逐模块恢复状态。

### 5.2 可微稀疏解码

`decodesparse` 复现 Detect 的 sigmoid、grid、stride 和 anchor 公式。Top-K 依据 objectness 与最大类别概率的乘积。学生 gather 后仍在计算图内；教师输出完全分离。

### 5.3 一对一匹配

`DistillationMatch` 显式保存四类索引。教师以类别、置信度和旋转 IoU 匹配 GT；学生以 GT 短边归一化的中心距离匹配，不要求类别一致。

### 5.4 可检测性路由

知识生存度同时包含：

- 教师置信度；
- PSF 模糊；
- 降采样倍率；
- 噪声；
- 退化后的 GT 有效短边。

角度可靠性同时包含：

- 教师 180 维 CSL 分布的归一化熵；
- GT 长宽比。

分类、中心、尺度和角度分别使用平方根、一次、平方、平方乘角度可靠性的衰减。

### 5.5 四分量损失

- 分类：温度 KL；
- 中心：输入尺寸归一化 Smooth L1；
- 尺度：对数空间 Smooth L1；
- 角度：180 维 CSL 温度 KL。

## 六、测试证据

上一轮已核实：

- 原测试：41/41 通过；
- 旋转 IoU 专项：16/16 通过；
- 真实 CPU 单批次：5 张合成 DOTA 图像加载成功，loss 为有限值 `0.7626`；
- 模型前向、真实 `ComputeLoss`、反向传播和优化器更新均执行。

本轮在最终工作树上独立复验：

- `pytest -q`：66 passed，0 failed；
- 命名守卫：文件、Python AST、YAML、有效文档路径和 Python 示例均通过；
- 解码器：真实 YOLOv5-OBB 的 FP32/inplace 与 FP16/non-inplace 两条 eval 路径均一致；
- 梯度：学生梯度非零，教师无梯度且 BatchNorm 统计不变；
- 匹配：师生候选顺序不同时仍匹配同一 GT，所有索引一对一；
- 路由：全部权重位于 `[0,1]`，退化增强或短边减小时不增加，近方形角度权重降低；
- 旋转 IoU：相同、分离、细长、垂直和边界情形均通过；
- Shapely：缺失时明确失败，上游回退接口保留极小框过滤；
- dry-run：合并配置和真实路径验证通过；
- 真实 CPU 单批次：5/5 图像和标签加载，0 corrupted，前向、真实 `ComputeLoss`、反向传播和优化器更新完成，有限 loss 为 `0.762591`；
- `python -m py_compile trainmsd.py utils/*.py`：通过；
- `git diff --check`：通过。

测试产生两类已知上游警告，不影响本轮通过状态：

1. `pkg_resources` 已被 setuptools 标记为弃用；
2. 本机未编译 C++ 旋转 NMS，因此测试使用严格 Shapely 回退。

## 七、Phase 状态

| Phase | 当前状态 |
|---|---|
| Phase 1 基础设施 | 已完成并保持 baseline 等价性测试 |
| Phase 1 真实单批次 | 本轮迁移后复验通过，有限 loss 为 `0.762591` |
| Phase 2 文献 CP3 | 已完成 22 篇矩阵和 8 篇直接竞争文献对比 |
| CP4-Pre 技术定义 | v0.4 已合并为唯一版本 |
| P0-A.1 独立组件 | 已实现并通过 66 项全量测试 |
| P0-A.2 trainer 集成 | 未开始 |
| GPU 与完整 DOTA 实验 | 等待上云 |

## 八、对创新性的谨慎表述

当前可保留的研究假设是：

> 在传感器退化下，类别、中心、尺度和旋转角知识的可传递性不同；基于目标有效像素足迹、教师置信度和角度可靠性进行分量级路由，可能减少统一蒸馏造成的负迁移。

此时可以说“提出并实现待验证机制”，不能说：

- 首次提出；
- 已证明减少负迁移；
- 已优于 CRKD-YOLO、MSCD 或 Orientation-KD；
- 已达到低显存目标；
- 已取得 DOTA 精度提升。

这些结论必须由下一阶段实验支撑。

## 九、下一阶段计划

1. 在独立组件测试保持通过的前提下接入 `MSDYOLOTrainer`；
2. 增加蒸馏预热、总权重、有效匹配数和四分量日志；
3. 验证 baseline、退化、清晰分支和完整蒸馏配置；
4. 在 GPU 上测量峰值显存、吞吐量和串行教师额外开销；
5. 准备完整 DOTA 数据并运行修订后的消融矩阵；
6. 以多随机种子报告 mAP、短边分桶 AP、长宽比分桶 AP 和置信区间；
7. 获得真实数据后再冻结论文创新声明。

## 十、请 Claude 重点复核

1. `decodesparse` 是否在所有 YOLOv5-OBB Detect 版本中保持 anchor 单位一致；
2. 教师和学生的两阶段一对一匹配阈值是否需要离线校准；
3. sensor factor 的几何平均是否应替换为离线可检测性曲线；
4. 温度、总蒸馏权重和 warmup 的最小实验矩阵；
5. trainer 集成时如何确保清晰 dense output 在学生前向前释放；
6. GPU 验证应记录 allocated 与 reserved 两套峰值，避免错误宣称显存节省。
