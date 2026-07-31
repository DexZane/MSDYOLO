# MSDYOLO CP4-Pre 技术定义 v0.4

状态：P0-A.1 独立组件已实现，尚未接入训练包装器  
日期：2026-07-31  
取代：v0.1、v0.2、v0.3 全部旧定义

## 1. 科学问题

MSDYOLO 研究的中心问题是：

> 在 PSF 模糊、空间降采样和噪声退化下，类别、中心、尺度和旋转角知识具有不同的可辨识性。统一蒸馏会把已经无法从退化输入恢复的精细几何知识强加给学生，可能造成负迁移。

本项目采用单模型、串行双视图自蒸馏：

- 清晰视图由同一模型在 `eval` 和 `torch.no_grad()` 下产生教师知识；
- 退化视图由同一模型在训练状态下产生学生预测并正常反向传播；
- 不加载大型外部教师；
- 不保存 FPN 密集特征；
- 推理时仍只有原 YOLOv5-OBB 模型，结构不变。

## 2. 阶段边界

P0-A.1 只交付以下独立组件：

1. 可微稀疏 raw output 解码；
2. 清晰教师分支；
3. 教师、学生与 GT 的一对一匹配；
4. 可检测性路由；
5. 四分量蒸馏损失；
6. 独立单元测试。

本阶段不把蒸馏总损失接入 `MSDYOLOTrainer`。只有下一阶段完成训练闭环、权重系数和日志定义后才允许启用蒸馏配置。

## 3. 数据格式

### 3.1 Detect raw output

每个尺度输出：

```python
rawoutputs = [
    p3,  # (batch, anchors, height, width, 201)
    p4,
    p5,
]
```

201 维定义：

- `0:2`：中心偏移 raw logits；
- `2:4`：宽高 raw logits；
- `4`：objectness raw logit；
- `5:21`：16 类 raw logits；
- `21:201`：180 维 CSL raw logits。

### 3.2 DOTA target

当前仓库的目标张量为 `(N, 187)`：

```text
0       batchindex
1       classid
2:4     cx, cy，像素坐标
4:6     longedge, shortedge，像素尺寸
6       angle，弧度
7:187   180维 CSL 标签
```

坐标和边长均不是归一化数值。

## 4. 可微稀疏解码

唯一公开入口：

```python
sparse = decodesparse(rawoutputs, model, topk)
```

每个尺度严格复现 YOLOv5-OBB Detect 公式：

```text
xy = (sigmoid(rawxy) × 2 - 0.5 + grid) × stride
wh = (sigmoid(rawwh) × 2)² × anchor × stride
confidence = sigmoid(objectness) × max(sigmoid(classlogits))
```

grid 与 anchor grid 按上游实现以 FP32 构造。`Detect.inplace=True` 时，最终 `xywh` 回写到 raw dtype；`inplace=False` 时保留上游类型提升结果。FP16 两条分支都必须与真实模型 eval output 一致。

Top-K 按 `confidence` 选取。以下信息必须同时返回：

- 解码后的 `xywh`；
- objectness、类别和 CSL 原始 logits；
- `scaleindex`；
- `anchorindex`；
- `gridx` 与 `gridy`；
- `rawindex`。

Top-K 和 gather 只用于选择候选，不切断学生张量的反向传播。

## 5. 清晰教师分支

教师输出必须按真实 eval tuple 解包：

```python
model.eval()
with torch.no_grad():
    decodedoutput, teacherraw = model(cleanimages)
    teachersparse = decodesparse(teacherraw, model, topk)
```

执行前记录每个子模块的训练状态，执行后逐模块恢复。教师所有输出都要 `detach`；BatchNorm 运行统计不得更新。

教师知识来源是“同一参数模型在更高质量清晰视图上的预测”，因此构成具有独立输入信息源和 stop-gradient 边界的自蒸馏，不是同一预测复制两遍。

## 6. 一对一匹配

匹配结果必须显式区分双方索引：

```python
DistillationMatch(
    batchindex,
    studentindex,
    teacherindex,
    targetindex,
)
```

匹配分两阶段进行：

1. 教师候选与 GT：
   - 教师最大概率类别等于 GT 类别；
   - 教师置信度超过阈值；
   - 教师旋转框与 GT 的精确旋转 IoU 超过阈值；
   - 按 IoU 从高到低贪心，教师和 GT 均不可重复。
2. 学生候选与已经获得教师匹配的 GT：
   - 不限制学生类别；
   - 距离为中心欧氏距离除以 GT 短边；
   - 按距离从低到高贪心，学生和 GT 均不可重复。

教师和学生 Top-K 的排序互不相关，损失不得用同一个候选索引读取两侧预测。

精确旋转 IoU 依赖 `Shapely>=2.0`。缺少依赖时必须明确失败，禁止静默改用水平外接框近似。

上游 NMS 兼容入口继续过滤短边小于 `0.001` 的退化框，避免命名迁移改变既有推理行为。

## 7. 可检测性路由

对每个已匹配目标：

```text
effectiveshortedge = min(longedge, shortedge) / downsamplefactor
shortedgefactor = 1 - exp(-effectiveshortedge / shortedgethreshold)
blurfactor = 1 / (1 + psfsigma)
downsampleimpact = 1 / downsamplefactor
noisefactor = 1 / (1 + 10 × noiselevel)
sensorfactor = cubicroot(blurfactor × downsampleimpact × noisefactor)
survival = clamp(teacherconfidence × sensorfactor × shortedgefactor, 0, 1)
```

角度可靠性由教师 CSL 分布熵与 GT 长宽比共同决定：

```text
entropyreliability = 1 - entropy(teacherangleprobability) / log(180)
aspectratio = max(longedge / shortedge, shortedge / longedge)
aspectreliability = 1 - exp(-abs(log(aspectratio)))
anglereliability = entropyreliability × aspectreliability
```

四分量权重：

```text
classification = sqrt(survival)
center = survival
scale = survival²
angle = survival² × anglereliability
```

全部权重必须位于 `[0,1]`。模糊、噪声或降采样增强，以及有效短边减小时，任一权重都不得增加。近方形目标的角度权重应显著降低。

## 8. 四分量损失

```text
classification：带温度平方修正的类别 KL
center：按输入尺寸归一化的 Smooth L1
scale：log-space Smooth L1
angle：带温度平方修正的 180 维 CSL KL
```

教师分布始终 `detach`。空匹配通过 `student.values.sum() × 0` 产生零损失，保证设备、数据类型和计算图与学生一致。

## 9. 消融设计

| 编号 | 输入 | 清晰分支 | 蒸馏 | 路由 |
|---|---|---:|---:|---|
| E1 | 清晰 | 否 | 否 | 无，清晰上界 |
| E2 | 退化 | 否 | 否 | 无，鲁棒性基线 |
| E3 | 退化 | 是 | 否 | 无，计算控制组 |
| E4 | 退化 | 是 | 是 | 四分量统一为 1 |
| E5 | 退化 | 是 | 是 | 只用统一 survival |
| E6 | 退化 | 是 | 是 | 分量衰减但不使用角度可靠性 |
| E7 | 退化 | 是 | 是 | 完整路由 |
| E8 | 退化 | 是 | 是 | 只蒸馏分类 |

E2 与 E3 在清晰分支无损失且无副作用时，模型更新应相同；E3 用于量化额外计算和验证状态隔离，不预期精度提升。

DOTA 主指标使用旋转检测 mAP，并补充：

- 不同退化强度 mAP；
- 不同目标短边区间 AP；
- 不同长宽比区间 AP；
- 四个蒸馏分量的独立损失；
- 有效匹配数和平均路由权重；
- 峰值显存、吞吐量和缓存规模。

## 10. 显存约束

- 清晰分支与学生分支串行执行；
- 教师使用 no-grad；
- 教师稀疏结果提取后立即释放 dense output；
- 不保存 FPN 特征；
- 默认 `batchsize=2`、`imagesize=1024`；
- 本地 CPU 只验证逻辑闭环，真实峰值显存必须在云端 GPU 实测后报告。

任何显存数值在 GPU 实测前只能标为预算，不能标为结果。

## 11. 当前验收

P0-A.1 必须通过：

- 教师 eval tuple 正确解包；
- 自定义 xywh 与模型 eval 解码一致；
- 学生梯度非零；
- 教师无梯度且 BatchNorm 统计不变；
- 师生候选顺序不同时仍匹配同一 GT；
- 三类索引均无重复；
- 路由单调、有界且抑制近方形角度；
- 空匹配损失设备和类型正确；
- 缺少 Shapely 时明确失败。

下一阶段才进行：

1. 把独立组件接入 `MSDYOLOTrainer`；
2. 加入蒸馏总权重、预热和日志；
3. GPU 显存与速度验证；
4. 云端完整 DOTA 消融和统计分析。
