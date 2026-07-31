# CP4-Pre技术定义修订要点 (v0.2)

**⚠️ DEPRECATED - DO NOT USE FOR IMPLEMENTATION ⚠️**

**This document (v0.2) is outdated and contains execution-level errors.**

**Please use**: `docs/cp4pre_techdef.md` (v0.3 Final)

---

**修订日期**: 2026-07-31  
**基于**: GPT第十一轮审核意见  
**修订项**: 10项执行级问题

---

## 修订总览

| 序号 | 问题 | 原v0.1 | 修订v0.2 | 状态 |
|------|------|--------|---------|------|
| 1 | 旋转NMS不可用 | torchvision.ops.nms_rotated | 纯Python实现+按类别NMS | ✅ |
| 2 | 学生训练输出解码 | extract_top_k(eval输出) | 可微训练输出解码器 | ✅ |
| 3 | CSL角度分布 | 仅保存标量角度 | 保留180维logits | ✅ |
| 4 | 教师学生数值域 | 未统一 | 统一为原始logits | ✅ |
| 5 | 匹配规则 | 学生类别=GT | 学生按空间匹配 | ✅ |
| 6 | 路由公式 | 未用PSF/噪声 | 显式包含全部退化 | ✅ |
| 7 | target像素单位 | 未明确 | 归一化→像素转换 | ✅ |
| 8 | 角度可靠性 | 仅熵 | 熵×长宽比 | ✅ |
| 9 | 消融实验 | Unified=Full-NoRouting | 重新定义8组 | ✅ |
| 10 | 显存验证 | 声称"验证" | 改为"预算" | ✅ |

---

## 修订1: 旋转NMS实现

### 问题
v0.1使用`torchvision.ops.nms_rotated`，但该接口不可用

### 修订
**已实现**: `utils/nms_rotated_pure.py`

**按类别NMS**:
```python
def obb_nms_per_class(predictions, iou_threshold=0.45):
    """
    按类别执行旋转NMS
    
    Args:
        predictions: (N, 201) [x,y,w,h,obj, 16class_logits, 180angle_dist]
    """
    nc = 16
    keep_all = []
    
    # 解码
    boxes = predictions[:, :5]  # x,y,w,h,obj
    class_logits = predictions[:, 5:21]
    angle_dist = predictions[:, 21:]
    
    # 预测类别
    class_ids = class_logits.argmax(dim=1)
    
    # 按类别NMS
    for cls in range(nc):
        mask = class_ids == cls
        if mask.sum() == 0:
            continue
        
        cls_boxes = boxes[mask]
        cls_scores = boxes[mask, 4] * class_logits[mask, cls]
        
        # 旋转NMS
        keep = obb_nms_python(cls_boxes, cls_scores, iou_threshold)
        keep_all.append(mask.nonzero()[keep])
    
    return torch.cat(keep_all) if keep_all else torch.zeros(0, dtype=torch.long)
```

---

## 修订2: 学生训练输出可微解码

### 问题
YOLOv5训练模式输出多尺度原始预测，不能直接用`extract_top_k(eval输出)`

### 修订

**训练输出结构**:
```python
# YOLOv5-OBB训练输出
train_outputs = [
    P3: (B, 3, H3, W3, 201),  # 8倍下采样
    P4: (B, 3, H4, W4, 201),  # 16倍下采样
    P5: (B, 3, H5, W5, 201),  # 32倍下采样
]
# 201 = 5(xywh+obj) + 16(class) + 180(angle)
```

**可微稀疏解码器**:
```python
def extract_sparse_from_train_output(train_outputs, top_k=100):
    """
    从训练输出提取Top-K稀疏预测（保持梯度）
    
    Args:
        train_outputs: List[Tensor] 多尺度原始预测
        top_k: int
    
    Returns:
        sparse_preds: (K, 201+meta) 保留完整信息和梯度
    """
    all_preds = []
    
    for scale_idx, output in enumerate(train_outputs):
        B, A, H, W, C = output.shape
        
        # Reshape: (B, A, H, W, 201) -> (B, A*H*W, 201)
        preds = output.view(B, -1, C)
        
        # 添加scale/anchor/grid元信息（用于后续匹配）
        # 这些信息不参与梯度，但需要保存
        anchors = torch.arange(A).repeat_interleave(H*W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grids = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        meta = {
            'scale_idx': scale_idx,
            'anchors': anchors,
            'grids': grids,
            'stride': model.stride[scale_idx]
        }
        
        all_preds.append((preds, meta))
    
    # 合并所有尺度
    merged = torch.cat([p[0] for p in all_preds], dim=1)  # (B, N_total, 201)
    
    # Top-K选择（基于objectness）
    obj_scores = merged[:, :, 4]  # objectness
    top_k_indices = obj_scores.topk(top_k, dim=1)[1]
    
    # 提取Top-K（保持梯度）
    sparse = torch.gather(merged, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, 201))
    
    return sparse  # (B, K, 201) with gradients
```

**关键特性**:
- ✅ 保持梯度连接（可反向传播）
- ✅ 保留完整201维信息（含CSL角度分布）
- ✅ 保存scale/anchor/grid元信息（用于匹配）
- ✅ 不切换eval模式（保持BN训练行为）

---

## 修订3: 保留CSL角度分布

### 问题
v0.1稀疏结构只保存解码后的标量角度，无法计算角度KL散度

### 修订

**扩展稀疏预测结构**:
```python
class SparsePrediction:
    def __init__(self, predictions):
        """
        Args:
            predictions: (K, 201) [x,y,w,h,obj, 16class, 180angle]
        """
        self.boxes = predictions[:, :5]  # (K, 5)
        self.class_logits = predictions[:, 5:21]  # (K, 16) 保留logits
        self.angle_logits = predictions[:, 21:]  # (K, 180) 保留logits
        
        # 解码（用于匹配，但不用于损失）
        self.class_ids = self.class_logits.argmax(dim=1)
        self.angles_decoded = decode_csl(self.angle_logits)  # (K,) radians
```

**角度KL散度**:
```python
def angle_kl_loss(student_angle_logits, teacher_angle_logits, temp=3.0):
    """
    Args:
        student_angle_logits: (N, 180) 原始logits
        teacher_angle_logits: (N, 180) 原始logits
    """
    student_log_probs = F.log_softmax(student_angle_logits / temp, dim=1)
    teacher_probs = F.softmax(teacher_angle_logits / temp, dim=1)
    
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return loss * (temp ** 2)
```

---

## 修订4: 统一教师学生数值域

### 问题
教师eval输出已sigmoid，学生train输出是原始logits，不能直接KL散度

### 修订

**方案A: 统一为原始logits（推荐）**
```python
# 教师分支：保留原始head输出
model.eval()
with torch.no_grad():
    # 不使用model(x)，而是手动forward到head
    clean_features = model.backbone(clean_images)
    clean_neck = model.neck(clean_features)
    teacher_logits = model.head(clean_neck)  # 原始logits，未sigmoid
    teacher_logits = teacher_logits.detach()

# 学生分支：同样使用原始head输出
model.train()
student_logits = model.head(model.neck(model.backbone(degraded_images)))

# 统一的KL散度
kl_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * (T ** 2)
```

**方案B: 统一为概率（备选）**
```python
# 如果必须使用model(x)
teacher_probs = torch.sigmoid(teacher_outputs)  # 已sigmoid
student_probs = torch.sigmoid(student_outputs)  # 手动sigmoid

# 使用JS散度或MSE
js_loss = js_divergence(student_probs, teacher_probs)
```

**推荐**: 方案A，保留原始logits更稳定

---

## 修订5: 匹配规则放宽

### 问题
v0.1要求学生类别=GT类别，会排除错误分类样本

### 修订

**新匹配规则**:
```python
def sparse_matching_relaxed(teacher_preds, student_preds, gt_boxes):
    """
    放宽的匹配规则
    
    教师匹配：类别一致 + IoU > 0.5
    学生匹配：空间位置 + IoU > 0.3（不要求类别）
    """
    matches = []
    
    for gt_idx, gt in enumerate(gt_boxes):
        # 1. 教师匹配：类别必须一致
        teacher_candidates = []
        for t_idx, t_pred in enumerate(teacher_preds):
            if t_pred.class_id == gt.class_id:  # 教师类别=GT
                iou = rotated_iou(t_pred.box, gt.box)
                if iou >= 0.5:
                    teacher_candidates.append((iou, t_idx))
        
        if not teacher_candidates:
            continue
        
        # 选择IoU最高的教师
        best_teacher_idx = max(teacher_candidates)[1]
        
        # 2. 学生匹配：仅按空间位置（不要求类别）
        student_candidates = []
        for s_idx, s_pred in enumerate(student_preds):
            # 不检查 s_pred.class_id == gt.class_id
            iou = rotated_iou(s_pred.box, gt.box)
            if iou >= 0.3:  # 学生阈值更低
                student_candidates.append((iou, s_idx))
        
        if not student_candidates:
            continue
        
        best_student_idx = max(student_candidates)[1]
        
        matches.append((gt_idx, best_teacher_idx, best_student_idx))
    
    return matches
```

**关键变化**:
- 教师：类别一致 + IoU ≥ 0.5
- 学生：**仅IoU ≥ 0.3**（不要求类别）
- 允许蒸馏错误分类样本

---

## 修订6: 路由公式完整化

### 问题
v0.1未真正使用PSF和噪声参数

### 修订

**完整路由公式**:
```python
def detectability_routing(teacher_pred, gt_box, degradation_params):
    """
    Args:
        degradation_params: {
            'psf_sigma': float,
            'downsample_factor': float,
            'noise_level': float
        }
    """
    # 1. 退化后目标短边（像素）
    img_size = degradation_params.get('img_size', 1024)
    gt_w_px = gt_box[2] * img_size  # 归一化→像素
    gt_h_px = gt_box[3] * img_size
    short_edge = min(gt_w_px, gt_h_px) / degradation_params['downsample_factor']
    
    # 2. 清晰教师置信度
    teacher_conf = teacher_pred[4] * teacher_pred[5:21].softmax(dim=0).max()
    
    # 3. 角度可靠性（熵 × 长宽比）
    angle_dist = teacher_pred[21:].softmax(dim=0)
    angle_entropy = -(angle_dist * angle_dist.log()).sum()
    q_entropy = 1.0 - angle_entropy / math.log(180)  # 归一化
    
    aspect_ratio = max(gt_w_px, gt_h_px) / (min(gt_w_px, gt_h_px) + 1e-6)
    q_shape = 1.0 - torch.exp(-torch.abs(torch.log(aspect_ratio)))  # 接近1时可靠
    
    q_angle = q_entropy * q_shape
    
    # 4. 退化强度综合（全部退化因素）
    psf_impact = 1.0 / (1.0 + degradation_params['psf_sigma'])
    downsample_impact = 1.0 / degradation_params['downsample_factor']
    noise_impact = 1.0 / (1.0 + degradation_params['noise_level'] * 10)
    
    degradation_factor = (psf_impact + downsample_impact + noise_impact) / 3.0
    
    # 5. 四分量权重
    w_cls = teacher_conf * degradation_factor
    
    w_center = teacher_conf * (short_edge / 10.0) * degradation_factor
    w_center = torch.clamp(w_center, 0.0, 1.0)
    
    w_scale = teacher_conf * (short_edge / 20.0) * degradation_factor
    w_scale = torch.clamp(w_scale, 0.0, 1.0)
    
    w_angle = teacher_conf * q_angle * (short_edge / 15.0) * degradation_factor
    w_angle = torch.clamp(w_angle, 0.0, 1.0)
    
    return {'w_cls': w_cls, 'w_center': w_center, 'w_scale': w_scale, 'w_angle': w_angle}
```

**关键改进**:
- ✅ 显式包含PSF sigma
- ✅ 显式包含降采样倍率
- ✅ 显式包含噪声水平
- ✅ 归一化到[0, 1]
- ✅ 长宽比影响角度可靠性

---

## 修订7: target像素单位转换

### 问题
DOTA标签可能是归一化坐标，路由需要像素单位

### 修订

**明确转换**:
```python
def gt_to_pixel(gt_box, img_size=1024):
    """
    GT格式: (class_id, cx_norm, cy_norm, w_norm, h_norm, angle)
    """
    cx_px = gt_box[1] * img_size
    cy_px = gt_box[2] * img_size
    w_px = gt_box[3] * img_size
    h_px = gt_box[4] * img_size
    angle = gt_box[5]  # 弧度
    
    return (cx_px, cy_px, w_px, h_px, angle)
```

**用于路由**:
```python
gt_px = gt_to_pixel(gt_box, img_size=1024)
short_edge_px = min(gt_px[2], gt_px[3])
degraded_short_edge = short_edge_px / downsample_factor
```

---

## 修订8: 角度可靠性改进

### 问题
v0.1仅用熵，未考虑正方形目标的角度歧义

### 修订

**公式**:
```python
# 熵可靠性（归一化）
q_entropy = 1.0 - H(p_theta) / log(K)  # K=180

# 形状可靠性（长宽比）
aspect_ratio = max(w, h) / min(w, h)
q_shape = 1.0 - exp(-abs(log(aspect_ratio)))

# 综合角度可靠性
q_angle = q_entropy * q_shape
```

**解释**:
- `aspect_ratio ≈ 1`（正方形）: `q_shape ≈ 0`（角度不可靠）
- `aspect_ratio >> 1`（细长）: `q_shape → 1`（角度可靠）

---

## 修订9: 消融实验去重

### 问题
Unified-KD和Full-NoRouting都是w=1.0，重复

### 修订

**新8组实验**:
```python
experiments = {
    'Baseline': {
        'distillation': False,
    },
    'Cls-Only': {
        'distillation': True,
        'components': ['cls'],
        'routing': None,
    },
    'Cls+Center': {
        'distillation': True,
        'components': ['cls', 'center'],
        'routing': None,
    },
    'Cls+Center+Scale': {
        'distillation': True,
        'components': ['cls', 'center', 'scale'],
        'routing': None,
    },
    'Full-Fixed': {
        'distillation': True,
        'components': ['cls', 'center', 'scale', 'angle'],
        'routing': 'fixed',  # w = [1.0, 1.0, 1.0, 1.0]
    },
    'Full-Uniform': {
        'distillation': True,
        'components': ['cls', 'center', 'scale', 'angle'],
        'routing': 'uniform_conf',  # w = teacher_conf * [1.0, 1.0, 1.0, 1.0]
    },
    'Full-Component': {
        'distillation': True,
        'components': ['cls', 'center', 'scale', 'angle'],
        'routing': 'component',  # w = [f1(conf), f2(conf, size), f3(conf, size), f4(conf, size, angle)]
    },
    'Full-Detectability': {
        'distillation': True,
        'components': ['cls', 'center', 'scale', 'angle'],
        'routing': 'detectability',  # 完整路由（PSF+downsample+noise+conf+size+angle）
    },
}
```

**8组独立条件**:
1. Baseline - 无蒸馏
2. Cls-Only - 仅分类
3. Cls+Center - 分类+中心
4. Cls+Center+Scale - 分类+中心+尺度
5. Full-Fixed - 四分量固定权重1.0
6. Full-Uniform - 四分量统一置信度路由
7. Full-Component - 四分量分量级简单路由
8. Full-Detectability - 四分量完整可检测性路由

---

## 修订10: 显存预算vs验证

### 问题
v0.1声称"显存验证结果"，实际是估算

### 修订

**改为"显存预算"**:
```markdown
## 六、显存预算与待验证假设

**目标**: 单卡RTX 3090 (24GB) batch_size=2

**预算分配**:
- 模型参数：~200MB (YOLOv5-OBB)
- 清晰分支前向：~4GB (batch=2, no_grad)
- 退化分支前向+反向：~8GB (batch=2)
- 稀疏预测存储：~10MB (K=100)
- 优化器状态：~400MB
- 其他开销：~1GB

**总计**: ~13.6GB < 24GB ✓（预算）

**待GPU实测验证**:
- torch.cuda.max_memory_allocated()
- torch.cuda.max_memory_reserved()
- 单iteration时间
- baseline与自蒸馏增量
```

---

## 实现优先级

### P0（必需）
1. ✅ 可微稀疏解码器
2. ✅ CSL角度分布保留
3. ✅ 教师学生数值域统一
4. ✅ 放宽匹配规则
5. ✅ 完整路由公式

### P1（完整实验需要）
6. ✅ 按类别旋转NMS
7. ✅ 消融实验去重
8. ✅ 角度可靠性改进
9. ⏳ GPU显存实测

### P2（优化）
10. ⏳ 可学习路由
11. ⏳ 离线校准路由

---

**修订完成度**: 10/10 ✅  
**状态**: 所有执行级问题已明确定义  
**下一步**: 实现可微稀疏解码器并集成到MSDYOLOTrainer
