# CP4-Pre Technical Definition (v0.3 Final)

**Status**: P0-A Revision  
**Date**: 2026-07-31  
**Based on**: GPT Round 12 Audit  
**Replaces**: cp4pretechdef.md (v0.1), cp4pre_revisions.md (v0.2)

---

## Document Purpose

This is the **single authoritative technical definition** for CP4-Pre (Phase 1 self-distillation baseline). All implementation must follow this document. Previous versions (v0.1, v0.2) are deprecated.

---

## 1. System Architecture

### 1.1 Training Pipeline

```
Input: Degraded Images (B, 3, H, W)
       ↓
   [Student Model]
       ↓
   Raw Predictions (multi-scale, training mode)
       ↓
   [Shared Differentiable Decoder] ← Key: preserves gradients
       ↓
   Sparse Predictions (B, K, 201)
       ↓
   [Four-Component Losses] ← cls/center/scale/angle
       ↓
   Total Loss → Backward
```

### 1.2 Teacher Signal (Self-Distillation)

```
Input: Clean Images (B, 3, H, W)
       ↓
   [Same Model in eval() + no_grad()]
       ↓
   Raw Predictions (multi-scale)
       ↓
   [Same Decoder] → Teacher Sparse (B, K, 201)
       ↓
   .detach() → Used as distillation targets
```

**Critical**: Teacher and student share the **same model weights** (self-distillation).

---

## 2. Data Format Specifications

### 2.1 YOLOv5-OBB Training Output

```python
train_outputs = [
    P3: (B, 3, H3, W3, 201),  # stride=8
    P4: (B, 3, H4, W4, 201),  # stride=16
    P5: (B, 3, H5, W5, 201),  # stride=32
]
```

**201 dimensions**:
- `[0:5]`: objectness + box (xywh offsets, NOT decoded)
- `[5:21]`: 16 class logits (raw, NOT softmax)
- `[21:201]`: 180-dim CSL angle distribution (raw, NOT softmax)

### 2.2 Ground Truth Format (Current Repository)

```python
targets: (N, 187)
# Schema:
[batch_index,           # 0
 class_id,              # 1
 cx_pixel,              # 2
 cy_pixel,              # 3
 long_edge_pixel,       # 4 (NOT normalized)
 short_edge_pixel,      # 5 (NOT normalized)
 theta_radian,          # 6 ∈[-π/2, π/2)
 180_csl_labels...]     # 7:187
```

**Important**: Coordinates are in **pixels**, NOT normalized.

---

## 3. Shared Differentiable Decoder

### 3.1 Purpose

Extract Top-K predictions from **training-mode** multi-scale outputs while **preserving gradients** for student branch.

### 3.2 Implementation

```python
def decode_train_output_to_sparse(train_outputs, model, top_k=100):
    """
    Decode multi-scale training predictions to sparse format
    
    Args:
        train_outputs: List[(B, 3, Hi, Wi, 201)] from model(x)
        model: YOLOv5-OBB model (for anchor/stride info)
        top_k: int
    
    Returns:
        sparse: (B, K, 201) with gradients
        meta: dict with decode metadata
    """
    B = train_outputs[0].shape[0]
    device = train_outputs[0].device
    
    all_decoded = []
    
    # Decode each scale
    for i, pred in enumerate(train_outputs):
        # Get anchor/stride info from model
        stride = model.model[-1].stride[i]  # e.g., 8, 16, 32
        anchors = model.model[-1].anchors[i]  # (3, 2)
        
        _, na, h, w, no = pred.shape  # na=3 anchors
        
        # Reshape: (B, na, h, w, 201) -> (B, na*h*w, 201)
        pred_flat = pred.view(B, -1, no)
        
        # Decode xywh (MUST apply sigmoid + grid offset)
        xy_raw = pred_flat[:, :, :2]  # raw offsets
        wh_raw = pred_flat[:, :, 2:4]
        obj_raw = pred_flat[:, :, 4:5]
        
        # Grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()  # (h, w, 2)
        grid = grid.view(1, 1, h, w, 2).expand(B, na, -1, -1, -1)
        grid_flat = grid.reshape(B, na*h*w, 2)
        
        # Anchor grid
        anchor_grid = anchors.view(1, na, 1, 1, 2).expand(B, -1, h, w, -1)
        anchor_flat = anchor_grid.reshape(B, na*h*w, 2)
        
        # Decode xy (pixel coordinates on feature map)
        xy_decoded = (xy_raw.sigmoid() * 2 - 0.5 + grid_flat) * stride
        
        # Decode wh (pixel dimensions)
        wh_decoded = (wh_raw.sigmoid() * 2) ** 2 * anchor_flat * stride
        
        # Keep class/angle logits as-is (NO softmax)
        cls_logits = pred_flat[:, :, 5:21]
        angle_logits = pred_flat[:, :, 21:201]
        
        # Concat: [xy_pixel, wh_pixel, obj_logit, cls_logits, angle_logits]
        decoded = torch.cat([
            xy_decoded,
            wh_decoded,
            obj_raw,  # Keep as logit
            cls_logits,
            angle_logits
        ], dim=-1)  # (B, na*h*w, 201)
        
        all_decoded.append(decoded)
    
    # Merge all scales
    merged = torch.cat(all_decoded, dim=1)  # (B, total_anchors, 201)
    
    # Top-K selection based on objectness logit
    obj_scores = merged[:, :, 4]  # (B, total_anchors)
    top_k_values, top_k_indices = obj_scores.topk(top_k, dim=1)
    
    # Gather Top-K (preserves gradients)
    top_k_indices_exp = top_k_indices.unsqueeze(-1).expand(-1, -1, 201)
    sparse = torch.gather(merged, 1, top_k_indices_exp)  # (B, K, 201)
    
    meta = {
        'num_scales': len(train_outputs),
        'total_anchors': merged.shape[1],
        'top_k': top_k
    }
    
    return sparse, meta
```

**Key Points**:
- Applies sigmoid + grid offset + stride to get **pixel coordinates**
- Keeps class/angle as **raw logits** (for KL divergence)
- Top-K selection on objectness logit (NOT sigmoid score)
- **All operations differentiable** → student gradients preserved

---

## 4. Teacher Branch Execution

### 4.1 Obtaining Teacher Predictions

```python
# Teacher forward (no gradient, eval mode)
model.eval()
with torch.no_grad():
    teacher_raw = model(clean_images)
    teacher_sparse, _ = decode_train_output_to_sparse(teacher_raw, model, top_k=100)
    teacher_sparse = teacher_sparse.detach()  # Extra safety

# Student forward (with gradient, train mode)
model.train()
student_raw = model(degraded_images)
student_sparse, meta = decode_train_output_to_sparse(student_raw, model, top_k=100)
```

**Critical**:
- Use `model.eval()` + `no_grad()` for teacher
- Use `model.train()` for student
- **Do NOT** manually call `model.backbone()` / `model.neck()` / `model.head()` (not exposed in YOLOv5-OBB)

### 4.2 Teacher Confidence Calculation

```python
def compute_teacher_confidence(teacher_sparse):
    """
    Compute teacher confidence for routing
    
    Args:
        teacher_sparse: (B, K, 201)
    
    Returns:
        confidence: (B, K) ∈[0, 1]
    """
    obj_logit = teacher_sparse[:, :, 4]
    cls_logits = teacher_sparse[:, :, 5:21]
    
    # Convert logits to probabilities
    obj_prob = torch.sigmoid(obj_logit)  # (B, K)
    cls_prob = torch.sigmoid(cls_logits)  # (B, K, 16)
    cls_max_prob = cls_prob.max(dim=-1)[0]  # (B, K)
    
    # Confidence = obj * cls_max
    confidence = obj_prob * cls_max_prob
    confidence = confidence.clamp(0, 1)
    
    return confidence.detach()
```

---

## 5. Detectability Routing

### 5.1 Complete Formula

```python
def compute_detectability_routing(degradation_params, teacher_sparse, teacher_confidence):
    """
    Route学生-教师匹配基于退化强度和教师置信度
    
    Args:
        degradation_params: dict with 'psf_sigma', 'downsample_factor', 'noise_level'
        teacher_sparse: (B, K, 201)
        teacher_confidence: (B, K) teacher置信度
    
    Returns:
        routing_weights: (B, K, 4) for [cls, center, scale, angle]
    """
    B, K = teacher_confidence.shape
    device = teacher_confidence.device
    
    # 1. Degradation strength (ALL factors)
    psf_impact = 1.0 / (1.0 + degradation_params['psf_sigma'])
    downsample_impact = 1.0 / degradation_params['downsample_factor']
    noise_impact = 1.0 / (1.0 + degradation_params['noise_level'] * 10)
    
    degradation_factor = (psf_impact + downsample_impact + noise_impact) / 3.0
    degradation_factor = torch.tensor(degradation_factor, device=device)
    
    # 2. Object size (from teacher predictions)
    wh = teacher_sparse[:, :, 2:4]  # (B, K, 2) pixel dimensions
    obj_area = wh[:, :, 0] * wh[:, :, 1]  # (B, K)
    obj_size_factor = torch.sqrt(obj_area) / 1024.0  # Normalize by image size
    obj_size_factor = obj_size_factor.clamp(0.01, 1.0)
    
    # 3. Base detectability
    detectability = teacher_confidence * degradation_factor * obj_size_factor
    detectability = detectability.clamp(0, 1)
    
    # 4. Component-specific transferability
    # Class: high transferability (semantic stable)
    w_cls = 0.9 + 0.1 * detectability
    
    # Center: medium transferability (affected by blur)
    w_center = 0.5 + 0.5 * detectability
    
    # Scale: lower transferability (PSF spreads edges)
    w_scale = 0.3 + 0.7 * detectability
    
    # Angle: lowest transferability (needs shape sharpness)
    # Also consider aspect ratio (square boxes → ambiguous angle)
    aspect_ratio = wh[:, :, 0] / (wh[:, :, 1] + 1e-6)
    aspect_ratio = torch.maximum(aspect_ratio, 1.0 / aspect_ratio)  # ≥1
    angle_shape_factor = 1.0 - torch.exp(-torch.abs(torch.log(aspect_ratio)))
    w_angle = (0.2 + 0.8 * detectability) * angle_shape_factor
    
    # Stack: (B, K, 4)
    routing_weights = torch.stack([w_cls, w_center, w_scale, w_angle], dim=-1)
    
    return routing_weights
```

**Key Changes from v0.1**:
- Explicitly includes PSF + downsample + noise (GPT要求)
- Angle reliability = detectability × aspect_ratio_factor (GPT要求)
- All weights in [0, 1]

---

## 6. Four-Component Losses

### 6.1 Matching Strategy (Relaxed for Student)

```python
def match_predictions_to_gt(student_sparse, teacher_sparse, gt_targets, routing_weights):
    """
    Match student predictions to GT (spatial only, no class constraint)
    
    Args:
        student_sparse: (B, K, 201)
        teacher_sparse: (B, K, 201)
        gt_targets: (N, 187)
        routing_weights: (B, K, 4)
    
    Returns:
        matched_indices: list of (student_idx, gt_idx) per batch
        valid_mask: (total_matches,)
    """
    matched_indices = []
    
    for b in range(B):
        batch_mask = gt_targets[:, 0] == b
        gt_batch = gt_targets[batch_mask]  # (n_gt, 187)
        
        if len(gt_batch) == 0:
            continue
        
        student_xy = student_sparse[b, :, :2]  # (K, 2)
        gt_xy = gt_batch[:, 2:4]  # (n_gt, 2) pixel coordinates
        
        # Spatial distance matrix
        dist = torch.cdist(student_xy, gt_xy, p=2)  # (K, n_gt)
        
        # Match each GT to closest student prediction (no class filter)
        min_dist, student_idx = dist.min(dim=0)  # (n_gt,)
        
        # Filter by distance threshold (e.g., 50 pixels)
        valid = min_dist < 50.0
        
        for gt_idx in torch.where(valid)[0]:
            s_idx = student_idx[gt_idx]
            matched_indices.append((b, s_idx.item(), gt_idx.item()))
    
    return matched_indices
```

**Key Change**: Student matches by **spatial proximity only**, allowing class mismatches (GPT requirement).

### 6.2 Loss Computation

```python
def compute_four_component_loss(student_sparse, teacher_sparse, gt_targets, 
                                 routing_weights, matched_indices):
    """
    Compute four-component self-distillation loss
    
    Returns:
        loss_dict: {
            'loss_cls': weighted class KL,
            'loss_center': weighted L1,
            'loss_scale': weighted L1,
            'loss_angle': weighted KL
        }
    """
    if len(matched_indices) == 0:
        return {k: torch.tensor(0.0) for k in ['loss_cls', 'loss_center', 'loss_scale', 'loss_angle']}
    
    losses = {'loss_cls': [], 'loss_center': [], 'loss_scale': [], 'loss_angle': []}
    
    for b, s_idx, gt_idx in matched_indices:
        student_pred = student_sparse[b, s_idx]  # (201,)
        teacher_pred = teacher_sparse[b, s_idx]  # (201,)
        weight = routing_weights[b, s_idx]  # (4,)
        
        # 1. Class loss (KL divergence on logits)
        student_cls_logits = student_pred[5:21]  # (16,)
        teacher_cls_logits = teacher_pred[5:21]  # (16,)
        
        student_cls_log_prob = F.log_softmax(student_cls_logits, dim=0)
        teacher_cls_prob = F.softmax(teacher_cls_logits, dim=0).detach()
        
        loss_cls_sample = F.kl_div(student_cls_log_prob, teacher_cls_prob, reduction='sum')
        losses['loss_cls'].append(weight[0] * loss_cls_sample)
        
        # 2. Center loss (L1 on pixel coordinates)
        student_xy = student_pred[:2]
        teacher_xy = teacher_pred[:2].detach()
        
        loss_center_sample = F.l1_loss(student_xy, teacher_xy, reduction='sum')
        losses['loss_center'].append(weight[1] * loss_center_sample)
        
        # 3. Scale loss (L1 on pixel dimensions)
        student_wh = student_pred[2:4]
        teacher_wh = teacher_pred[2:4].detach()
        
        loss_scale_sample = F.l1_loss(student_wh, teacher_wh, reduction='sum')
        losses['loss_scale'].append(weight[2] * loss_scale_sample)
        
        # 4. Angle loss (KL divergence on 180-dim CSL distribution)
        student_angle_logits = student_pred[21:201]  # (180,)
        teacher_angle_logits = teacher_pred[21:201]  # (180,)
        
        student_angle_log_prob = F.log_softmax(student_angle_logits, dim=0)
        teacher_angle_prob = F.softmax(teacher_angle_logits, dim=0).detach()
        
        loss_angle_sample = F.kl_div(student_angle_log_prob, teacher_angle_prob, reduction='sum')
        losses['loss_angle'].append(weight[3] * loss_angle_sample)
    
    # Average over all matches
    return {k: torch.stack(v).mean() if v else torch.tensor(0.0) for k, v in losses.items()}
```

**Key Points**:
- Class/Angle: KL divergence on **raw logits** (apply softmax inside KL)
- Center/Scale: L1 loss on **pixel values**
- All teacher targets use `.detach()`
- Routing weights scale each component independently

---

## 7. Ablation Experiments (Revised)

### 7.1 Experiment Groups (8 Configurations)

| ID | Name | Degradation | Clean Branch | Distillation | Routing |
|----|------|-------------|--------------|--------------|---------|
| E1 | Baseline | ❌ | ❌ | ❌ | N/A |
| E2 | Degrade-Only | ✅ | ❌ | ❌ | N/A |
| E3 | Clean-Branch | ✅ | ✅ | ❌ | N/A |
| E4 | Distill-NoRoute | ✅ | ✅ | ✅ | Uniform(1.0) |
| E5 | Distill-Detect | ✅ | ✅ | ✅ | Detectability |
| E6 | Distill-Comp | ✅ | ✅ | ✅ | Component-specific |
| E7 | **Full (CP4-Pre)** | ✅ | ✅ | ✅ | Detect + Comp |
| E8 | Oracle | ❌ | ❌ | ❌ | N/A (clean images) |

**Key Change**: E4 uses uniform routing (1.0 for all components), NOT "Full - NoRouting" (GPT correction).

### 7.2 Expected Results

- E1 < E2: Degradation hurts performance
- E2 < E3: Clean branch helps
- E4 < E5: Detectability routing matters
- E4 < E6: Component-specific routing matters
- E5, E6 < E7: Both routing factors needed
- E7 < E8: Oracle upper bound

---

## 8. Implementation Checklist

### Phase 1: Core Components (P0)
- [ ] Shared differentiable decoder (Section 3)
- [ ] Teacher branch execution (Section 4)
- [ ] Detectability routing (Section 5)
- [ ] Four-component losses (Section 6)
- [ ] Training loop integration

### Phase 2: Validation
- [ ] Gradient flow test (student sparse → loss → backward)
- [ ] Teacher detached test (no gradients in teacher branch)
- [ ] Routing weights in [0,1] test
- [ ] Loss values non-negative test

### Phase 3: Experiments
- [ ] Run 8 ablation configurations (Section 7)
- [ ] Compare mIoU on DOTA v1.5 test set
- [ ] Analyze component-wise improvements

---

## 9. GPU Memory Budget

**Estimated for CP4-Pre**:
- Baseline (E1): 11 GB
- Full (E7): 15 GB (+4 GB for clean branch + distillation)

**Target**: Fit in single RTX 4090 (24 GB) with batch_size=8

---

## 10. Known Limitations

1. **Teacher-student gap**: Self-distillation may have limited gains (teacher not stronger)
2. **Routing approximation**: Fixed formula, not learned
3. **Top-K selection**: May miss small objects if K too small
4. **CSL angle**: 180-class may be redundant (consider 90-class)

---

## 11. Document History

- **v0.1** (cp4pretechdef.md): Initial draft, many execution gaps
- **v0.2** (cp4pre_revisions.md): GPT Round 11 fixes, still incomplete
- **v0.3** (this doc): GPT Round 12 P0-A revision, executable definition

**Status**: Ready for implementation after P0-A闭环验证通过

---

**END OF CP4-PRE TECHNICAL DEFINITION**
