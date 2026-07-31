"""
Pure Python Rotated NMS Implementation
使用Shapely实现正确的旋转框IoU计算
"""

import torch
import numpy as np
import warnings

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    warnings.warn(
        "Shapely not available. Rotated IoU will use axis-aligned approximation. "
        "Install with: pip install shapely"
    )


def box_iou_rotated(box1, box2):
    """
    计算两个旋转框的IoU (使用Shapely精确计算)

    Args:
        box1: (cx, cy, w, h, angle) angle in radians [-pi/2, pi/2)
        box2: (cx, cy, w, h, angle)

    Returns:
        iou: float in [0, 1]
    """
    if not SHAPELY_AVAILABLE:
        # 回退到轴对齐近似
        return _box_iou_axis_aligned_approx(box1, box2)

    # 计算旋转后的4个顶点
    def get_corners(box):
        cx, cy, w, h, angle = box
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 4个顶点相对于中心的坐标
        dx = w / 2
        dy = h / 2

        corners = [
            (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a),
            (cx - dx * cos_a - dy * sin_a, cy - dx * sin_a + dy * cos_a),
            (cx - dx * cos_a + dy * sin_a, cy - dx * sin_a - dy * cos_a),
            (cx + dx * cos_a + dy * sin_a, cy + dx * sin_a - dy * cos_a),
        ]
        return corners

    corners1 = get_corners(box1)
    corners2 = get_corners(box2)

    # 创建多边形
    try:
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)

        # 确保多边形有效
        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)

        # 计算交集和并集
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union < 1e-6:
            return 0.0

        iou = intersection / union

        # 确保IoU在[0,1]范围内
        return max(0.0, min(1.0, iou))

    except Exception as e:
        warnings.warn(f"Shapely polygon operation failed: {e}. Using fallback.")
        return _box_iou_axis_aligned_approx(box1, box2)


def _box_iou_axis_aligned_approx(box1, box2):
    """
    使用轴对齐外接矩形近似IoU（仅作为回退方案）

    注意：这是近似值，不适用于大角度旋转的情况
    """
    def get_bounding_box(box):
        cx, cy, w, h, angle = box
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        dx = w / 2
        dy = h / 2

        corners = [
            (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a),
            (cx - dx * cos_a - dy * sin_a, cy - dx * sin_a + dy * cos_a),
            (cx - dx * cos_a + dy * sin_a, cy - dx * sin_a - dy * cos_a),
            (cx + dx * cos_a + dy * sin_a, cy + dx * sin_a - dy * cos_a),
        ]

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    x1_min, y1_min, x1_max, y1_max = get_bounding_box(box1)
    x2_min, y2_min, x2_max, y2_max = get_bounding_box(box2)

    # 计算外接矩形的IoU
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area

    if union_area < 1e-6:
        return 0.0

    iou = inter_area / union_area
    return max(0.0, min(1.0, iou))


def obb_nms_python(dets, scores, iou_thr):
    """
    纯Python旋转框NMS实现

    Args:
        dets: (N, 5) tensor or array, [cx, cy, w, h, angle]
        scores: (N,) tensor or array
        iou_thr: float

    Returns:
        keep_inds: (M,) tensor, indices of kept boxes
    """
    if isinstance(dets, np.ndarray):
        dets = torch.from_numpy(dets)
        scores = torch.from_numpy(scores)
        to_numpy = True
    else:
        to_numpy = False

    if dets.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    # 按分数降序排序
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        # 计算当前框与剩余框的IoU
        ious = []
        for j in order[1:]:
            iou = box_iou_rotated(dets[i].tolist(), dets[j].tolist())
            ious.append(iou)

        ious = torch.tensor(ious)

        # 保留IoU小于阈值的框
        inds = torch.where(ious <= iou_thr)[0]
        order = order[inds + 1]

    keep_inds = torch.tensor(keep, dtype=torch.long)

    if to_numpy:
        keep_inds = keep_inds.numpy()

    return keep_inds


def obb_nms_per_class(predictions, class_ids, iou_threshold=0.45, score_threshold=0.25):
    """
    按类别执行旋转NMS

    Args:
        predictions: (N, 5) [cx, cy, w, h, angle]
        class_ids: (N,) 预测类别ID
        iou_threshold: IoU阈值
        score_threshold: 分数阈值

    Returns:
        keep_indices: (M,) 保留的索引
    """
    if len(predictions) == 0:
        return torch.zeros(0, dtype=torch.long)

    nc = int(class_ids.max()) + 1
    keep_all = []

    for cls in range(nc):
        mask = class_ids == cls
        if mask.sum() == 0:
            continue

        cls_boxes = predictions[mask]
        cls_indices = torch.where(mask)[0]

        # 假设predictions包含置信度分数
        if cls_boxes.shape[1] > 5:
            cls_scores = cls_boxes[:, 5]
        else:
            cls_scores = torch.ones(len(cls_boxes))

        # 分数过滤
        score_mask = cls_scores > score_threshold
        if score_mask.sum() == 0:
            continue

        cls_boxes = cls_boxes[score_mask, :5]
        cls_scores = cls_scores[score_mask]
        cls_indices = cls_indices[score_mask]

        # 旋转NMS
        keep = obb_nms_python(cls_boxes, cls_scores, iou_threshold)
        keep_all.append(cls_indices[keep])

    if len(keep_all) == 0:
        return torch.zeros(0, dtype=torch.long)

    return torch.cat(keep_all)


def obb_nms(dets, scores, iou_thr, device_id=None):
    """
    旋转框NMS包装函数（兼容原接口）

    Args:
        dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        scores (tensor/array): (num)
        iou_thr (float): IoU threshold
        device_id: unused (for compatibility)

    Returns:
        dets (tensor/array): (n_nms, [cx cy w h θ])
        inds (tensor/array): (n_nms), nms index of dets
    """
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        device = dets.device
        dets_np = dets.cpu().numpy()
        scores_np = scores.cpu().numpy()
    else:
        is_tensor = False
        dets_np = dets
        scores_np = scores

    if len(dets_np) == 0:
        empty = np.array([], dtype=np.int64)
        return (torch.from_numpy(empty).to(device) if is_tensor else empty,
                torch.from_numpy(empty).to(device) if is_tensor else empty)

    # 过滤过小的框
    too_small = np.minimum(dets_np[:, 2], dets_np[:, 3]) < 0.001
    if too_small.all():
        empty = np.array([], dtype=np.int64)
        return (torch.from_numpy(empty).to(device) if is_tensor else empty,
                torch.from_numpy(empty).to(device) if is_tensor else empty)

    ori_inds = np.arange(len(dets_np))
    valid_inds = ori_inds[~too_small]
    valid_dets = dets_np[~too_small]
    valid_scores = scores_np[~too_small]

    # 执行NMS
    keep_inds = obb_nms_python(valid_dets, valid_scores, iou_thr)

    if isinstance(keep_inds, torch.Tensor):
        keep_inds = keep_inds.numpy()

    final_inds = valid_inds[keep_inds]

    if is_tensor:
        final_inds = torch.from_numpy(final_inds).to(device)
        return dets[final_inds], final_inds
    else:
        return dets_np[final_inds], final_inds


def poly_nms(dets, iou_thr, device_id=None):
    """
    多边形NMS（暂不实现，抛出NotImplementedError）
    """
    raise NotImplementedError("poly_nms requires C++ extension or external library")


__all__ = ['obb_nms', 'poly_nms', 'obb_nms_python', 'obb_nms_per_class', 'box_iou_rotated']
