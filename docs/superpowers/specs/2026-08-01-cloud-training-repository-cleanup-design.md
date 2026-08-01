# MSDYOLO 云端训练与仓库收敛设计

## 目标

修复从全新云端实例执行 `bash scripts/setup.sh` 到 DOTA v1.5 数据准备和 MSDYOLO 训练启动的完整链路，同时消除重构后遗留的双份 Python 包、重复配置和失效脚本。最终仓库必须只有一套可维护实现，并用自动测试覆盖曾导致 `match=0`、负坐标、路径缺失和安装失败的关键契约。

本轮不重新设计检测模型、蒸馏算法或 DOTA 评测方法，也不承诺在本地环境复现 V100 上的完整首 epoch。真实云端成功标准仍是完整 MSD 配置的首个 epoch 出现 `match > 0`；本地验证负责证明达到这一结果所需的数据、配置、导入和单批训练前置条件。

## 已确认的问题

当前仓库虽然已经在 `msdyolo/data/scripts/split_dota.py` 中输出像素坐标，但仍存在以下断点：

1. 下载器把 `val/labelTxt` 作为必需目录。DOTA v1.5 官方 val 无标注时，下载器会先退出失败，`setup.sh` 因 `set -e` 无法执行后续的空目录创建。
2. `setup.sh` 只检查 `split/train/images` 是否存在，可能复用中断、过时或仍为归一化坐标的切分结果。
3. splitter 输出 `labels/`，loader 固定读取 `labelTxt/`，依赖 setup 事后创建符号链接。
4. `setup.sh` 用全局 `pkill -f "msdyolo.train"` 清理进程，可能终止不属于本次任务的训练。
5. 一键训练使用 baseline 配置，而 `MSDYOLOTrainer.baselineforward()` 按设计固定返回 `matchcount=0`，与“首 epoch `match > 0`”的成功标准矛盾。
6. 根目录 `models/`、`utils/`、数据 YAML、训练 YAML 和多个 setup/download/split 脚本与 `msdyolo/` 包形成双轨实现；现有测试仍主要导入旧根目录模块。
7. 根目录 `train.py` 是旧 YOLOv5 训练器，`msdyolo/train.py` 是新 MSDYOLO 训练器，不能在没有兼容测试的情况下把其中任何一个静默删除。
8. 仓库没有针对下载目录整理、像素标签、切分完整性或 setup 幂等性的自动回归测试。

## 选定方案

采用完整收敛方案：`msdyolo/` 成为唯一 Python 实现，根目录命令只保留兼容入口；配置、数据工具和云端脚本各自只有一个权威位置。迁移顺序必须是先建立行为测试、再改引用、最后删除重复文件。

相比只修改 `setup.sh`，该方案改动更大，但能够消除造成当前问题反复出现的结构性原因。相比无兼容层地直接删除旧树，本方案保留常用根目录命令，减少使用者迁移成本。

## 目标仓库结构

```text
MSDYOLO/
├── msdyolo/
│   ├── train.py
│   ├── val.py
│   ├── detect.py
│   ├── export.py
│   ├── models/
│   ├── utils/
│   └── data/
│       ├── dota.yaml
│       ├── hyps/
│       └── scripts/
│           ├── download_dota.py
│           ├── prepare_dota.py
│           └── split_dota.py
├── configs/
│   ├── models/
│   └── train/
├── scripts/
│   ├── setup.sh
│   └── ddp_train.sh
├── tests/
│   └── fixtures/
├── docs/
│   └── archive/
├── train.py
├── val.py
├── detect.py
└── export.py
```

根目录 `models/` 和 `utils/` 在所有调用方迁移到 `msdyolo.*` 后删除。根目录四个 Python 命令变成轻量包装器，仅导入并调用相应包入口。当前根目录 `val.py` 迁移到新的 `msdyolo/val.py` 后再改为包装器。

用户可编辑的模型与训练配置保留在 `configs/models/` 和 `configs/train/`。DOTA 数据集描述和超参数保留在 `msdyolo/data/`，并通过打包配置纳入安装产物。重复的根目录 YAML 删除；合成测试数据迁移到 `tests/fixtures/`。

## 云端数据准备数据流

`scripts/setup.sh` 只负责编排，不在 shell 中实现复杂的数据判断：

1. 使用当前解释器的 `python -m pip`，先安装 Python 3.12 兼容的 setuptools，再安装项目及 OpenDataLab 依赖。
2. 调用 `python -m msdyolo.data.scripts.download_dota` 下载缺失数据。下载模块只调用 SDK，不在 Python 进程内执行 pip。
3. 调用 `python -m msdyolo.data.scripts.prepare_dota` 整理 OpenDataLab 的嵌套目录、创建允许为空的 val 标签目录、验证原始 train 数据并执行切分。
4. 数据准备成功后下载缺失的预训练权重。
5. 默认用完整 MSD 配置启动训练；用户可通过参数覆盖配置或只执行准备。

下载成功契约为：`train/images`、非空 `train/labelTxt` 和 `val/images` 存在。`val/labelTxt` 可以不存在，准备步骤会创建空目录。下载失败必须返回非零退出码并给出准确路径，不吞掉异常。

## DOTA 标签与切分契约

splitter 直接输出 `labelTxt/`，不再通过 `labels/` 和符号链接适配 loader。它接受带 DOTA 元数据头或不带头的标签文件：能解析为 8 个数值坐标、类别和可选 difficult 的行才作为目标，元数据行按内容忽略，不固定跳过前两行。

参数必须满足 `subsize > gap >= 0` 且进程数大于零。每个输出目标必须满足：

- 正好 10 列；
- 前 8 列为有限像素坐标；
- 坐标在闭区间 `[0, subsize]`；
- 类别属于 DOTA v1.5 的 16 类；
- difficult 为 `0`、`1` 或 `2`；
- 裁剪后多边形不是完全退化的单点。

目标仍按中心点是否落在 patch 内决定归属，跨边界顶点被裁剪。输出不进行 `[0,1]` 归一化。验证器除逐行检查外，还要求训练集至少有一个标签，并在数据集层面观察到大于 `1` 的坐标，从而识别整批误归一化标签。

## 幂等性与生成物安全

每次成功切分写入完成标记，记录：源图片和标签数量、源文件状态摘要、`subsize`、`gap`、类别版本和标签格式版本。复用已有切分结果必须同时满足完成标记匹配、图片非空、训练标签非空和逐行验证通过。

缺失标记、摘要变化、参数变化或标签验证失败时，准备步骤在独立临时目录重新生成。只有新结果完整通过验证后才替换生成目录；失败时保留原结果供排查。`--force-resplit` 强制走同一安全重建流程。

生成目录属于可再生数据，不进入 Git。代码不会删除原始 DOTA 下载目录。

## 训练启动与成功语义

裸命令 `bash scripts/setup.sh` 保持“一条命令准备并后台训练”的体验，但不再执行全局 `pkill`。启动成功后将 PID 写入项目运行目录。若 PID 文件对应的进程仍存活，setup 明确失败并提示用户处理；过期 PID 文件可安全替换。

支持以下控制方式：

- `--prepare-only`：完成安装、下载、整理、切分和验证后退出；
- `--force-resplit`：忽略有效完成标记并安全重建；
- `--foreground`：以前台进程运行训练；
- `--config PATH`：覆盖默认完整 MSD 配置。

baseline 配置保留用于消融，其 `match=0` 是预期语义。默认云端配置必须启用 degradation、clear branch 和 distillation，并使用 DOTA v1.5 split 路径、1024 图像尺寸和 4 个 worker。README 必须区分：数据链路健康检查关注非空 targets 与有限、非零 detection loss；`match > 0` 只适用于完整 MSD 模式。

真实训练日志在完整模式的首 epoch 结束时若始终没有匹配，应给出明确警告，包含 targets 数量、置信度、类别和 IoU 过滤统计的诊断入口，而不能把 baseline 的零匹配误报为数据故障。

## 兼容入口与旧文件处理

根目录 `train.py`、`val.py`、`detect.py` 和 `export.py` 在清理后仍可执行，并转发到包入口。包装器的 `--help` 和主要参数必须通过 smoke test。若旧根训练器具有新训练器尚未提供的参数，README 只承诺当前受测试的 MSDYOLO 参数；不保留两个长期分叉的训练实现。

完全重复或只用于一次性重构的脚本删除，包括旧 download/split/setup 变体、`restructure.sh`、`fix_imports.py` 和 `cleanup.sh`。仍有用途的 `ddp_train.sh` 更新为包入口和新配置路径。历史 README、原始上游说明和重构计划移动到 `docs/archive/`，以保留出处但不与当前文档竞争。

未跟踪的 `scripts/verify.sh` 属于用户工作，本轮保持原样，不纳入删除、修改或提交。

## 错误处理

所有数据准备命令采用非零退出码表示失败。错误信息必须指出失败阶段、实际路径和恢复动作。不得使用广泛的 `|| true` 隐藏标签移动、目录创建、切分或验证错误；仅允许对明确幂等且随后有状态验证的操作容忍“已存在”。

网络下载错误保留 SDK 原始异常摘要并列出人工下载方案。不可读图片、非法标签和不支持的类别计入错误报告；训练集出现任一非法标签时准备失败，不静默忽略。

## 测试策略

实施采用测试先行。新增或迁移的测试至少覆盖：

1. 有头和无头 DOTA 标签均正确解析；归一化标签、NaN、负值、未知类别和退化多边形被拒绝。
2. 跨 patch 目标被正确平移和裁剪，输出仍为像素坐标。
3. `gap >= subsize`、负 gap 和非正进程数立即失败。
4. OpenDataLab 嵌套 train 标签被整理；无标签 val 被接受并创建空目录。
5. 有效完成标记允许复用；不完整、旧格式和参数变化触发安全重建。
6. 默认云端配置启用完整 MSD，worker 为 4；baseline 明确保持零匹配语义。
7. 所有测试从 `msdyolo.*` 导入；删除根目录包后无导入回退。
8. 四个根目录兼容入口与安装后的 console entry point 可显示帮助并加载配置。
9. setup 参数解析、prepare-only 路径和 PID 冲突处理可在不联网、不启动真实训练的情况下测试。

完成前执行：完整 pytest、shell 语法检查、临时合成 DOTA 数据端到端准备、editable install 后的 CLI smoke test，以及 CPU 单批训练。无法在本地证明的 V100、CUDA 12.4、完整 DOTA 下载和首 epoch 匹配将单独列为云端验收步骤，不伪装为已验证。

## 文档与交付

README 更新为唯一当前操作指南，包含一键命令、prepare-only、恢复切分、前后台训练、PID 管理、baseline 与 full 的匹配语义及云端验收命令。旧文档中的失效路径要么更新，要么移动到 archive 并标注历史状态。

交付报告列出删除、移动和新增文件，说明未触碰的用户文件，并附每项验证命令及结果。
