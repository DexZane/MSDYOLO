"""
Pure Python Rotated NMS Implementation
无需C++扩展的旋转框NMS实现
"""

import torch
import numpy as np
import math


def box_iou_rotated(box1, box2):
    """
    计算两个旋转框的IoU (纯Python实现)

    Args:
        box1: (cx, cy, w, h, angle) angle in radians [-pi/2, pi/2)
        box2: (cx, cy, w, h, angle)

    Returns:
        iou: float
    """
    # 简化实现：使用外接矩形近似
    # 完整实现需要Shapely或cv2.rotatedRectangleIntersection

    # 计算旋转后的4个顶点
    def get_corners(box):
        cx, cy, w, h, angle = box
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

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

    # 获取外接矩形
    def get_bounding_box(corners):
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    corners1 = get_corners(box1)
    corners2 = get_corners(box2)

    x1_min, y1_min, x1_max, y1_max = get_bounding_box(corners1)
    x2_min, y2_min, x2_max, y2_max = get_bounding_box(corners2)

    # 计算外接矩形的IoU作为近似
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    area1 = box1[2] * box1[3]  # w * h
    area2 = box2[2] * box2[3]

    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


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


__all__ = ['obb_nms', 'poly_nms', 'obb_nms_python']
