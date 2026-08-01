"""YOLOv5-OBB 多尺度 raw output 的可微稀疏解码。"""

from dataclasses import dataclass

import torch


@dataclass
class SparsePredictions:
    """稀疏候选及其原始尺度、anchor、grid 和全局索引。"""

    values: torch.Tensor
    scaleindex: torch.Tensor
    anchorindex: torch.Tensor
    gridx: torch.Tensor
    gridy: torch.Tensor
    rawindex: torch.Tensor

    def tensors(self):
        """按固定顺序返回全部张量。"""
        return (
            self.values,
            self.scaleindex,
            self.anchorindex,
            self.gridx,
            self.gridy,
            self.rawindex,
        )

def detecthead(model):
    """取得 YOLO 模型的 Detect 头。"""
    if not hasattr(model, "model") or not model.model:
        raise ValueError("Model does not expose a YOLO Detect head")
    head = model.model[-1]
    for attribute in ("stride", "anchors"):
        if not hasattr(head, attribute):
            raise ValueError(f"Detect head is missing {attribute}")
    return head


def gathercandidates(tensor, indices):
    """按 batch Top-K 索引收集候选。"""
    expanded = indices.unsqueeze(-1).expand(*indices.shape, tensor.shape[-1])
    return torch.gather(tensor, 1, expanded)


def gathermetadata(tensor, indices):
    """收集二维候选元数据。"""
    expanded = tensor.unsqueeze(0).expand(indices.shape[0], -1)
    return torch.gather(expanded, 1, indices)


def decodesparse(rawoutputs, model, topk=100):
    """按真实 Detect 公式解码 xywh，保留 obj/class/CSL 原始 logits。"""
    if not isinstance(rawoutputs, (list, tuple)) or not rawoutputs:
        raise TypeError("rawoutputs must be a non-empty list or tuple")
    if topk <= 0:
        raise ValueError("topk must be positive")

    head = detecthead(model)
    decodedscales = []
    scaleindices = []
    anchorindices = []
    gridxs = []
    gridys = []
    rawindices = []
    offset = 0

    for scaleindex, raw in enumerate(rawoutputs):
        if raw.ndim != 5:
            raise ValueError("Each raw output must have shape (B, A, H, W, O)")
        batch, anchors, height, width, outputs = raw.shape
        if outputs < 185:
            raise ValueError("YOLOv5-OBB output must include 180 CSL logits")

        stride = torch.as_tensor(head.stride[scaleindex], device=raw.device)
        anchorsizes = head.anchors[scaleindex].to(device=raw.device)
        if anchorsizes.shape[0] != anchors:
            raise ValueError("Raw anchor count does not match Detect metadata")

        ygrid, xgrid = torch.meshgrid(
            torch.arange(height, device=raw.device),
            torch.arange(width, device=raw.device),
            indexing="ij",
        )
        grid = (
            torch.stack((xgrid, ygrid), -1)
            .float()
            .view(1, 1, height, width, 2)
            .expand(batch, anchors, -1, -1, -1)
        )
        anchorgrid = (
            (anchorsizes * stride)
            .view(1, anchors, 1, 1, 2)
            .expand(batch, -1, height, width, -1)
            .float()
        )

        xy = (raw[..., :2].sigmoid() * 2 - 0.5 + grid) * stride
        wh = (raw[..., 2:4].sigmoid() * 2) ** 2 * anchorgrid
        if getattr(head, "inplace", True):
            xy = xy.to(raw.dtype)
            wh = wh.to(raw.dtype)
        decoded = torch.cat((xy, wh, raw[..., 4:]), -1).reshape(batch, -1, outputs)
        decodedscales.append(decoded)

        count = anchors * height * width
        scaleindices.append(torch.full((count,), scaleindex, device=raw.device, dtype=torch.long))
        anchorindices.append(
            torch.arange(anchors, device=raw.device)
            .view(anchors, 1, 1)
            .expand(-1, height, width)
            .reshape(-1)
        )
        gridxs.append(xgrid.view(1, height, width).expand(anchors, -1, -1).reshape(-1))
        gridys.append(ygrid.view(1, height, width).expand(anchors, -1, -1).reshape(-1))
        rawindices.append(torch.arange(offset, offset + count, device=raw.device))
        offset += count

    merged = torch.cat(decodedscales, 1)
    classend = merged.shape[-1] - 180
    if classend <= 5:
        raise ValueError("YOLOv5-OBB output does not contain class logits")
    confidence = merged[..., 4].sigmoid() * merged[..., 5:classend].sigmoid().amax(-1)
    selected = min(topk, merged.shape[1])
    indices = confidence.topk(selected, 1).indices

    return SparsePredictions(
        values=gathercandidates(merged, indices),
        scaleindex=gathermetadata(torch.cat(scaleindices), indices),
        anchorindex=gathermetadata(torch.cat(anchorindices), indices),
        gridx=gathermetadata(torch.cat(gridxs), indices),
        gridy=gathermetadata(torch.cat(gridys), indices),
        rawindex=gathermetadata(torch.cat(rawindices), indices),
    )
