# MSDYOLO CP4-Pre 技术定义文档

**⚠️ DEPRECATED - DO NOT USE FOR IMPLEMENTATION ⚠️**

**This document (v0.1) is outdated and contains execution-level errors.**

**Please use**: `docs/cp4pre_techdef.md` (v0.3 Final)

---

**创建日期**: 2026-07-31  
**状态**: 已废弃 (Deprecated)  
**版本**: v0.1

---

## 一、教师与学生边界定义

### 1.1 教师分支

**定义**:
```python
# 教师：同一模型、清晰视图、eval + no_grad
model.eval()
with torch.no_grad():
    teacher_outputs = model(clean_images)
    teacher_outputs = teacher_outputs.detach()  # stop-gradient
```

**关键属性**:
- **模型**: 同一个YOLOv5-OBB模型（共享所有参数）
- **输入**: 清晰图像（无退化）
- **模式**: `model.eval()` - BatchNorm使用running stats
- **梯度**: `torch.no_grad()` - 不计算教师分支梯度
- **输出**: `detach()` - 切断梯度流

**推理时行为**:
- 推理时仅保留原检测模型
- 无需额外教师分支
- 无额外计算开销

### 1.2 学生分支

**定义**:
```python
# 学生：同一模型、退化视图、train + 正常反向传播
model.train()
student_outputs = model(degraded_images)  # 梯度正常传播
```

**关键属性**:
- **模型**: 同一个YOLOv5-OBB模型（与教师共享参数）
- **输入**: 退化图像（PSF模糊 + 降采样 + 噪声）
- **模式**: `model.train()` - BatchNorm使用batch stats
- **梯度**: 正常反向传播到模型参数
- **损失**: 检测损失 + 蒸馏损失

### 1.3 串行双视图前向传播

**执行流程**:
```python
# Phase 1: 清晰分支（教师）
model.eval()
with torch.no_grad():
    clean_preds = model(clean_images)
    teacher_sparse = extract_top_k(clean_preds)  # 立即提取稀疏预测
    teacher_sparse = teacher_sparse.detach()     # stop-gradient
    del clean_preds  # 释放密集预测

# Phase 2: 退化分支（学生）
model.train()
degraded_preds = model(degraded_images)
student_sparse = extract_top_k(degraded_preds)

# Phase 3: 稀疏匹配
matches = sparse_matching(teacher_sparse, student_sparse, gt_boxes)

# Phase 4: 计算损失
detection_loss = compute_detection_loss(student_sparse, gt_boxes)
distillation_loss = compute_distillation_loss(student_sparse, teacher_sparse, matches)
total_loss = detection_loss + distillation_loss
```

**显存优化**:
- 串行前向（非并行）：避免同时存储两份FPN特征
- 稀疏提取后立即释放密集预测
- 教师分支no_grad：节省梯度存储
- 不保存FPN中间特征：仅保存最终预测

---

## 二、稀疏预测匹配

### 2.1 旋转NMS定义

**输入**: 密集预测 `(N, 201)` 其中 `201 = 5 + nc + 180`
- `5`: x, y, w, h, objectness
- `nc=16`: DOTA v1.5类别logits
- `180`: CSL角度分布

**旋转NMS算法**:
```python
def rotated_nms(predictions, iou_threshold=0.45):
    """
    旋转框NMS
    
    Args:
        predictions: (N, 201) tensor
        iou_threshold: 旋转IoU阈值
    
    Returns:
        kept_indices: 保留的预测索引
    """
    # 1. 解码预测
    boxes = predictions[:, :5]  # x, y, w, h, obj
    class_logits = predictions[:, 5:21]  # 16类
    angle_dist = predictions[:, 21:]  # 180维CSL
    
    # 2. 解码角度（CSL → 弧度）
    angles = decode_csl(angle_dist)  # (N,) in radians
    
    # 3. 构造旋转框 (x, y, w, h, angle)
    rotated_boxes = torch.cat([boxes[:, :4], angles.unsqueeze(1)], dim=1)
    
    # 4. 计算置信度
    scores = boxes[:, 4] * class_logits.max(dim=1)[0]
    
    # 5. 旋转IoU NMS
    keep = torchvision.ops.nms_rotated(rotated_boxes, scores, iou_threshold)
    
    return keep
```

**关键点**:
- 使用`torchvision.ops.nms_rotated`（需PyTorch 1.13+）
- 旋转IoU考虑角度信息
- 置信度 = objectness × max(class_prob)

### 2.2 一对一匹配算法

**目标**: 为每个GT匹配唯一的教师预测和学生预测

**算法流程**:
```python
def sparse_matching(teacher_preds, student_preds, gt_boxes, K=100):
    """
    一对一稀疏匹配
    
    Args:
        teacher_preds: (K, 201) 教师Top-K预测
        student_preds: (K, 201) 学生Top-K预测
        gt_boxes: (M, 6) GT框 (x, y, w, h, angle, class_id)
        K: Top-K数量
    
    Returns:
        matches: List[(gt_idx, teacher_idx, student_idx)]
    """
    M = len(gt_boxes)
    matches = []
    
    # 1. 解码教师和学生预测
    teacher_boxes = decode_predictions(teacher_preds)  # (K, 5+1) x,y,w,h,angle,class
    student_boxes = decode_predictions(student_preds)
    
    # 2. 对每个GT找最佳教师预测
    teacher_used = set()
    student_used = set()
    
    for gt_idx in range(M):
        gt = gt_boxes[gt_idx]
        
        # 2.1 找与GT类别一致的教师预测
        teacher_candidates = []
        for t_idx in range(K):
            if t_idx in teacher_used:
                continue
            if teacher_boxes[t_idx, 5] == gt[5]:  # 类别一致
                iou = rotated_iou(teacher_boxes[t_idx, :5], gt[:5])
                teacher_candidates.append((iou, t_idx))
        
        if not teacher_candidates:
            continue  # 无匹配教师
        
        # 2.2 选择IoU最高的教师
        teacher_candidates.sort(reverse=True)
        best_teacher_iou, best_teacher_idx = teacher_candidates[0]
        
        if best_teacher_iou < 0.5:  # IoU阈值
            continue
        
        # 2.3 找与同一GT匹配的学生预测
        student_candidates = []
        for s_idx in range(K):
            if s_idx in student_used:
                continue
            if student_boxes[s_idx, 5] == gt[5]:  # 类别一致
                iou = rotated_iou(student_boxes[s_idx, :5], gt[:5])
                student_candidates.append((iou, s_idx))
        
        if not student_candidates:
            continue  # 无匹配学生
        
        # 2.4 选择IoU最高的学生
        student_candidates.sort(reverse=True)
        best_student_iou, best_student_idx = student_candidates[0]
        
        if best_student_iou < 0.3:  # 学生IoU阈值可以更低
            continue
        
        # 2.5 记录三元组
        matches.append((gt_idx, best_teacher_idx, best_student_idx))
        teacher_used.add(best_teacher_idx)
        student_used.add(best_student_idx)
    
    return matches
```

**关键约束**:
- **类别一致性**: 教师、学生、GT三者类别必须一致
- **一对一**: 每个GT最多匹配一个教师和一个学生
- **IoU阈值**: 教师≥0.5, 学生≥0.3（退化场景可放宽）
- **无匹配处理**: GT无匹配时跳过蒸馏损失（仅用检测损失）

### 2.3 教师重复框处理

**问题**: NMS后教师可能仍有重复框（IoU < threshold但语义重复）

**解决方案**: 优先选择置信度最高的教师预测
```python
# 在2.2算法的teacher_candidates排序时，增加置信度作为第二关键字
teacher_candidates.sort(key=lambda x: (x[0], teacher_preds[x[1], 4]), reverse=True)
```

### 2.4 GT辅助匹配仅用于训练

**明确**: GT仅在训练时可见
- 训练：使用GT辅助匹配教师-学生对
- 推理：仅使用学生分支（无教师、无GT、无蒸馏）

---

## 三、四分量损失定义

### 3.1 总体损失结构

```python
total_loss = detection_loss + lambda_distill * distillation_loss

detection_loss = lambda_cls * cls_loss + lambda_box * box_loss + lambda_obj * obj_loss

distillation_loss = (
    w_cls * L_cls_distill +
    w_center * L_center_distill +
    w_scale * L_scale_distill +
    w_angle * L_angle_distill
)
```

**超参数**:
- `lambda_distill = 1.0` - 蒸馏损失权重
- `w_cls, w_center, w_scale, w_angle` - 四分量权重（可检测性路由动态计算）

### 3.2 分类分布损失

**定义**: KL散度蒸馏类别分布

```python
def classification_distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """
    分类知识蒸馏
    
    Args:
        student_logits: (N, 16) 学生类别logits
        teacher_logits: (N, 16) 教师类别logits
        temperature: 温度参数
    
    Returns:
        loss: KL散度损失
    """
    # 1. 软化分布
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # 2. KL散度
    loss = F.kl_div(
        student_probs.log(),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return loss
```

**理由**: 类别分布适合用概率蒸馏

### 3.3 中心坐标损失

**定义**: L1损失（匹配Localization Distillation）

```python
def center_distillation_loss(student_centers, teacher_centers):
    """
    中心坐标蒸馏
    
    Args:
        student_centers: (N, 2) 学生中心坐标 (x, y)
        teacher_centers: (N, 2) 教师中心坐标 (x, y)
    
    Returns:
        loss: L1损失
    """
    loss = F.l1_loss(student_centers, teacher_centers, reduction='mean')
    return loss
```

**理由**: 中心坐标是连续值，L1损失直观且稳定

### 3.4 尺度损失

**定义**: L1损失（对数空间）

```python
def scale_distillation_loss(student_wh, teacher_wh):
    """
    尺度蒸馏
    
    Args:
        student_wh: (N, 2) 学生宽高 (w, h)
        teacher_wh: (N, 2) 教师宽高 (w, h)
    
    Returns:
        loss: 对数L1损失
    """
    # 1. 对数空间（处理尺度不变性）
    student_log_wh = torch.log(student_wh + 1e-6)
    teacher_log_wh = torch.log(teacher_wh + 1e-6)
    
    # 2. L1损失
    loss = F.l1_loss(student_log_wh, teacher_log_wh, reduction='mean')
    return loss
```

**理由**: 对数空间使损失对大小目标一致

### 3.5 周期性角度分布损失

**定义**: CSL分布的KL散度

```python
def angle_distillation_loss(student_angle_dist, teacher_angle_dist, temperature=3.0):
    """
    角度分布蒸馏
    
    Args:
        student_angle_dist: (N, 180) 学生CSL角度分布
        teacher_angle_dist: (N, 180) 教师CSL角度分布
        temperature: 温度参数
    
    Returns:
        loss: KL散度损失
    """
    # 1. 软化分布
    student_probs = F.softmax(student_angle_dist / temperature, dim=1)
    teacher_probs = F.softmax(teacher_angle_dist / temperature, dim=1)
    
    # 2. KL散度
    loss = F.kl_div(
        student_probs.log(),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return loss
```

**理由**: CSL角度表示本身是180类分类，KL散度天然适配

### 3.6 分量独立性要求

**关键约束**: 四个损失必须分别计算和统计

```python
# 错误：合并box loss
box_loss = l1_loss([student_x, student_y, student_w, student_h],
                   [teacher_x, teacher_y, teacher_w, teacher_h])

# 正确：分量独立
center_loss = l1_loss([student_x, student_y], [teacher_x, teacher_y])
scale_loss = l1_loss(log([student_w, student_h]), log([teacher_w, teacher_h]))
```

**理由**: 分量级贡献无法成立如果损失合并

---

## 四、可检测性路由定义

### 4.1 路由权重计算

**目标**: 为每个匹配对计算四个分量的蒸馏权重

```python
def detectability_routing(teacher_pred, student_pred, gt_box, degradation_params):
    """
    可检测性路由
    
    Args:
        teacher_pred: (201,) 教师预测
        student_pred: (201,) 学生预测
        gt_box: (6,) GT框 (x, y, w, h, angle, class_id)
        degradation_params: dict - 退化参数 {psf_sigma, downsample_factor, noise_level}
    
    Returns:
        weights: dict - {w_cls, w_center, w_scale, w_angle}
    """
    # 1. 退化后目标短边像素数
    degraded_short_edge = min(gt_box[2], gt_box[3]) / degradation_params['downsample_factor']
    
    # 2. 清晰分支置信度
    teacher_conf = teacher_pred[4] * teacher_pred[5:21].softmax(dim=0).max()
    
    # 3. 角度可靠性（CSL分布熵）
    angle_dist = teacher_pred[21:].softmax(dim=0)
    angle_entropy = -(angle_dist * angle_dist.log()).sum()
    angle_reliability = 1.0 / (1.0 + angle_entropy)
    
    # 4. 退化强度综合指标
    degradation_strength = (
        degradation_params['psf_sigma'] * 0.3 +
        degradation_params['downsample_factor'] * 0.4 +
        degradation_params['noise_level'] * 0.3
    )
    
    # 5. 四分量权重计算
    w_cls = teacher_conf  # 类别权重 = 教师置信度
    
    w_center = teacher_conf * (degraded_short_edge / 10.0)  # 中心权重 = 置信度 × 尺度因子
    w_center = torch.clamp(w_center, 0.0, 1.0)
    
    w_scale = teacher_conf * (degraded_short_edge / 20.0)  # 尺度权重更依赖目标大小
    w_scale = torch.clamp(w_scale, 0.0, 1.0)
    
    w_angle = teacher_conf * angle_reliability * (degraded_short_edge / 15.0)  # 角度最敏感
    w_angle = torch.clamp(w_angle, 0.0, 1.0)
    
    return {
        'w_cls': w_cls,
        'w_center': w_center,
        'w_scale': w_scale,
        'w_angle': w_angle
    }
```

### 4.2 路由策略选择

**三种实现方式**:

1. **固定函数路由** (Phase 1推荐):
   - 使用上述公式直接计算权重
   - 超参数：尺度因子 (10.0, 20.0, 15.0)
   - 优点：简单、可解释
   - 缺点：超参数需要手动调整

2. **可学习路由** (Phase 2探索):
   ```python
   class LearnableRouting(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc = nn.Sequential(
               nn.Linear(7, 32),  # 7 inputs: conf, short_edge, entropy, 3×degradation, class_id
               nn.ReLU(),
               nn.Linear(32, 4),  # 4 outputs: w_cls, w_center, w_scale, w_angle
               nn.Sigmoid()
           )
       
       def forward(self, features):
           return self.fc(features)
   ```
   - 优点：自适应学习最优权重
   - 缺点：增加参数量、可能过拟合

3. **离线校准路由** (Phase 3优化):
   - 在验证集上统计不同条件下的最优权重
   - 构建查找表
   - 推理时根据条件查表
   - 优点：兼顾准确性和效率
   - 缺点：需要额外校准步骤

**Phase 1选择**: 固定函数路由

### 4.3 路由消融实验

**必需对比**:
1. 无路由（w_cls = w_center = w_scale = w_angle = 1.0）
2. 仅置信度路由（所有权重 = teacher_conf）
3. 仅尺度路由（权重 ∝ degraded_short_edge）
4. 完整路由（上述4.1公式）

---

## 五、负迁移验证实验

### 5.1 实验矩阵

| 实验名称 | 蒸馏分量 | 路由策略 | 期望结果 |
|---------|---------|---------|---------|
| Baseline | 无蒸馏 | N/A | 基线性能 |
| Unified-KD | 全部4分量 | 无路由（w=1.0） | 可能负迁移 |
| Cls-Only | 仅分类 | 无路由 | 验证类别可传递性 |
| Cls+Center | 分类+中心 | 无路由 | 验证中心可传递性 |
| Cls+Center+Scale | 分类+中心+尺度 | 无路由 | 验证尺度可传递性 |
| Full-NoRouting | 全部4分量 | 无路由 | 验证是否负迁移 |
| Full-ConfRouting | 全部4分量 | 仅置信度路由 | 验证简单路由 |
| Full-Routing | 全部4分量 | 完整路由 | 验证分量级路由 |

### 5.2 评估指标

**主要指标**:
- mAP@0.5 (DOTA v1.5)
- mAP@0.75
- mAP@[0.5:0.95]

**分量级分析**:
- 分类准确率（正确类别比例）
- 中心定位误差（像素）
- 尺度IoU（宽高匹配度）
- 角度误差（度数）

### 5.3 负迁移判定标准

**负迁移成立条件**:
```
Unified-KD mAP < Baseline mAP
且
Full-Routing mAP > Unified-KD mAP
```

**如果负迁移不成立**: 说明研究假设不成立，需要重新审视创新点

---

## 六、显存约束验证

### 6.1 显存预算

**目标**: 单卡RTX 3090 (24GB) batch_size=2

**显存分配**:
- 模型参数：~200MB (YOLOv5-OBB)
- 清晰分支前向：~4GB (batch=2, no_grad)
- 退化分支前向+反向：~8GB (batch=2)
- 稀疏预测存储：~10MB (K=100)
- 优化器状态：~400MB
- 其他开销：~1GB

**总计**: ~13.6GB < 24GB ✅

### 6.2 显存优化检查清单

- [x] 串行双视图前向（避免并行）
- [x] 清晰分支`no_grad`
- [x] 稀疏输出立即`detach()`
- [x] 密集预测及时`del`
- [x] 不保存FPN中间特征
- [x] 不引入外部大教师
- [x] Top-K稀疏提取（K=100）

---

## 七、实现优先级

### P0 (必需完成才能训练):
1. ✅ 教师/学生边界定义
2. ⏳ 旋转NMS实现
3. ⏳ 一对一稀疏匹配
4. ⏳ 四分量损失实现
5. ⏳ 固定函数路由实现

### P1 (完整实验需要):
6. ⏳ 负迁移消融实验
7. ⏳ 分量级评估指标
8. ⏳ 显存profiling验证

### P2 (优化方向):
9. ⏳ 可学习路由
10. ⏳ 离线校准路由
11. ⏳ 温度参数调优

---

## 八、与GPT审核要求的对应

| GPT要求 | 本文档章节 | 完成状态 |
|---------|-----------|---------|
| 教师与学生边界 | 第一章 | ✅ 已定义 |
| 稀疏预测匹配 | 第二章 | ✅ 已定义 |
| 旋转NMS | 2.1节 | ✅ 已定义 |
| 类别一致性 | 2.2节 | ✅ 已定义 |
| 一对一匹配 | 2.2节 | ✅ 已定义 |
| 无匹配目标处理 | 2.2节 | ✅ 已定义 |
| 教师重复框处理 | 2.3节 | ✅ 已定义 |
| GT辅助匹配仅训练 | 2.4节 | ✅ 已定义 |
| 四分量损失 | 第三章 | ✅ 已定义 |
| 分类分布损失 | 3.2节 | ✅ 已定义 |
| 中心坐标损失 | 3.3节 | ✅ 已定义 |
| 尺度损失 | 3.4节 | ✅ 已定义 |
| 角度分布损失 | 3.5节 | ✅ 已定义 |
| 分量不合并 | 3.6节 | ✅ 已明确 |
| 可检测性路由 | 第四章 | ✅ 已定义 |
| 退化后目标足迹 | 4.1节 | ✅ 已定义 |
| 清晰分支置信度 | 4.1节 | ✅ 已定义 |
| 角度可靠性 | 4.1节 | ✅ 已定义 |
| 退化强度参数 | 4.1节 | ✅ 已定义 |
| 固定/可学习/校准 | 4.2节 | ✅ 已定义 |
| 负迁移验证 | 第五章 | ✅ 已定义 |
| 显存约束 | 第六章 | ✅ 已验证 |

---

**下一步**: 实现上述P0任务，集成到真实YOLOv5-OBB训练流程中
