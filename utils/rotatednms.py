"""基于 Shapely 的精确旋转框 IoU 与 NMS。"""

import math

import numpy as np
import torch

try:
    from shapely.geometry import Polygon as shapelypolygon
    from shapely.validation import make_valid as makevalid

    SHAPELYAVAILABLE = True
except ImportError:
    shapelypolygon = None
    makevalid = None
    SHAPELYAVAILABLE = False


def requireshapely():
    """科研计算禁止静默回退到不准确的水平框近似。"""
    if not SHAPELYAVAILABLE:
        raise RuntimeError(
            "Shapely>=2.0 is required for exact rotated IoU; "
            "install project requirements before matching"
        )


def boxcorners(box):
    """把 cx,cy,w,h,angle 转换为四个旋转顶点。"""
    centerx, centery, width, height, angle = [float(value) for value in box]
    cosine = math.cos(angle)
    sine = math.sin(angle)
    halfwidth = width / 2.0
    halfheight = height / 2.0
    return [
        (
            centerx + halfwidth * cosine - halfheight * sine,
            centery + halfwidth * sine + halfheight * cosine,
        ),
        (
            centerx - halfwidth * cosine - halfheight * sine,
            centery - halfwidth * sine + halfheight * cosine,
        ),
        (
            centerx - halfwidth * cosine + halfheight * sine,
            centery - halfwidth * sine - halfheight * cosine,
        ),
        (
            centerx + halfwidth * cosine + halfheight * sine,
            centery + halfwidth * sine - halfheight * cosine,
        ),
    ]


def rotatediou(firstbox, secondbox):
    """计算两个旋转矩形的精确 IoU。"""
    requireshapely()
    firstpolygon = shapelypolygon(boxcorners(firstbox))
    secondpolygon = shapelypolygon(boxcorners(secondbox))
    if not firstpolygon.is_valid:
        firstpolygon = makevalid(firstpolygon)
    if not secondpolygon.is_valid:
        secondpolygon = makevalid(secondpolygon)
    if firstpolygon.area <= 0 or secondpolygon.area <= 0:
        return 0.0
    intersection = firstpolygon.intersection(secondpolygon).area
    union = firstpolygon.union(secondpolygon).area
    if union <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, intersection / union)))


def rotatednms(boxes, scores, iouthreshold=0.45):
    """对旋转框执行按分数排序的贪心 NMS，并返回原始索引。"""
    requireshapely()
    numpyinput = isinstance(boxes, np.ndarray)
    if numpyinput:
        boxes = torch.as_tensor(boxes)
        scores = torch.as_tensor(scores)
    if boxes.numel() == 0:
        result = torch.empty(0, dtype=torch.long, device=boxes.device)
        return result.cpu().numpy() if numpyinput else result

    order = scores.argsort(descending=True)
    kept = []
    while order.numel():
        current = int(order[0].item())
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = []
        for candidate in order[1:].tolist():
            iou = rotatediou(
                boxes[current].detach().cpu().tolist(),
                boxes[candidate].detach().cpu().tolist(),
            )
            if iou <= iouthreshold:
                remaining.append(candidate)
        order = torch.tensor(remaining, dtype=torch.long, device=boxes.device)

    result = torch.tensor(kept, dtype=torch.long, device=boxes.device)
    return result.cpu().numpy() if numpyinput else result


def classwiserotatednms(boxes, scores, classids, iouthreshold=0.45):
    """分别对每个类别执行旋转 NMS。"""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    kept = []
    for classid in classids.unique(sorted=True):
        indices = torch.where(classids == classid)[0]
        selected = rotatednms(boxes[indices], scores[indices], iouthreshold)
        kept.append(indices[selected])
    return torch.cat(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)
