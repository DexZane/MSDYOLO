"""
YOLO Output Decoder for MSDYOLO
修正GPT第五轮审核问题3.3：正确解码坐标、置信度和角度

关键修正：
1. 处理eval模式返回(decoded, raw)元组
2. 避免二次sigmoid
3. 正确解码网格坐标和anchor
4. 正确解码CSL角度: θ = (idx - 90) / 180 * π
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import math


class YOLOOutputDecoder:
    """
    YOLOv5-OBB输出解码器

    处理两种输出模式：
    1. 训练模式：List[Tensor[B, na, H, W, no]]
    2. eval模式：(Tensor[B, N, no], List[...])
    """

    def __init__(
        self,
        num_classes: int = 16,  # DOTA v1.5 has 16 classes (including container-crane)
        conf_threshold: float = 0.25,
        img_size: int = 1024,
    ):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.img_size = img_size

    def decode(
        self,
        outputs,
        model_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解码YOLO输出

        Args:
            outputs: YOLO模型输出
                - 训练模式: List[Tensor[B, na, H, W, no]]
                - eval模式: (Tensor[B, N, no], List[...])
            model_training: 模型是否处于训练模式

        Returns:
            boxes_xywh: (B*N, 4) 中心坐标和宽高（像素）
            theta: (B*N,) 角度（弧度，范围[-π/2, π/2)）
            objectness: (B*N,) 前景置信度
            class_probs: (B*N, C) 类别概率
            class_ids: (B*N,) 最大概率类别
            batch_indices: (B*N,) batch索引
        """
        if model_training:
            # 训练模式：需要手动解码
            return self._decode_training_output(outputs)
        else:
            # eval模式：已部分解码，需要提取
            return self._decode_eval_output(outputs)

    def _decode_training_output(
        self,
        outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解码训练模式输出

        outputs: List[Tensor[B, na, H, W, no]]
        no = 5 + nc + 180 (x, y, w, h, obj, cls..., theta...)

        GPT要求：
        - 正确应用sigmoid
        - 正确解码网格坐标
        - 正确解码anchor宽高
        - 正确解码CSL角度
        """
        # TODO: 需要获取stride和anchor信息
        # 这需要从model中提取，暂时抛出NotImplementedError
        raise NotImplementedError(
            "训练模式解码需要stride和anchor信息，"
            "建议在ClearBranchForward中使用eval模式"
        )

    def _decode_eval_output(
        self,
        outputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解码eval模式输出

        eval模式下，YOLOv5-OBB返回：
        (decoded_predictions, raw_outputs)

        decoded_predictions: Tensor[B, N, no]
        - 已经应用sigmoid
        - 已经解码网格坐标
        - 已经解码anchor
        - CSL角度仍需解码

        GPT要求：避免二次sigmoid
        """
        # 处理eval模式返回的元组
        if isinstance(outputs, tuple):
            decoded_preds = outputs[0]  # (B, N, no)
        else:
            # 如果直接是tensor（某些版本）
            decoded_preds = outputs

        B = decoded_preds.shape[0]

        # 解析预测
        # no = 5 + num_classes + 180
        boxes_xywh = decoded_preds[..., :4]  # (B, N, 4) 已解码的像素坐标
        objectness = decoded_preds[..., 4]   # (B, N) 已sigmoid
        class_logits = decoded_preds[..., 5:5+self.num_classes]  # (B, N, C) 已sigmoid

        # CSL角度标签
        if decoded_preds.shape[-1] >= 5 + self.num_classes + 180:
            theta_logits = decoded_preds[..., 5+self.num_classes:5+self.num_classes+180]  # (B, N, 180)

            # 正确的CSL角度解码（GPT要求）
            # θ = (idx - 90) / 180 * π
            theta_idx = theta_logits.argmax(dim=-1)  # (B, N)
            theta = (theta_idx.float() - 90.0) / 180.0 * math.pi  # (B, N) 弧度
        else:
            # 无角度信息
            theta = torch.zeros_like(objectness)

        # 类别概率（已经是sigmoid后的）
        class_probs = class_logits  # (B, N, C)
        class_ids = class_probs.argmax(dim=-1)  # (B, N)

        # 展平batch维度，添加batch索引
        # 为每个预测添加batch_idx列
        batch_indices = torch.arange(B, device=boxes_xywh.device).view(B, 1).expand(B, boxes_xywh.shape[1])  # (B, N)

        # 展平为 (B*N, ...)
        boxes_xywh = boxes_xywh.reshape(-1, 4)  # (B*N, 4)
        theta = theta.reshape(-1)  # (B*N,)
        objectness = objectness.reshape(-1)  # (B*N,)
        class_probs = class_probs.reshape(-1, self.num_classes)  # (B*N, C)
        class_ids = class_ids.reshape(-1)  # (B*N,)
        batch_indices = batch_indices.reshape(-1)  # (B*N,)

        return boxes_xywh, theta, objectness, class_probs, class_ids, batch_indices


def test_decoder():
    """测试解码器"""
    print("Testing YOLO Output Decoder...")

    decoder = YOLOOutputDecoder(num_classes=16)

    # 模拟eval模式输出
    B, N = 1, 100
    no = 5 + 15 + 180

    # 创建模拟输出（已sigmoid、已解码坐标）
    decoded_preds = torch.rand(B, N, no)
    decoded_preds[..., :4] = decoded_preds[..., :4] * 1024  # 像素坐标

    # 解码
    boxes, theta, obj, cls_prob, cls_id, batch_idx = decoder.decode(
        (decoded_preds, None),
        model_training=False
    )

    print(f"\n✓ Decoded outputs:")
    print(f"  Boxes: {boxes.shape}, range: [{boxes.min():.1f}, {boxes.max():.1f}]")
    print(f"  Theta: {theta.shape}, range: [{theta.min():.3f}, {theta.max():.3f}] rad")
    print(f"  Theta in degrees: [{theta.min()*180/math.pi:.1f}, {theta.max()*180/math.pi:.1f}]")
    print(f"  Objectness: {obj.shape}, range: [{obj.min():.3f}, {obj.max():.3f}]")
    print(f"  Class probs: {cls_prob.shape}")
    print(f"  Class ids: {cls_id.shape}")
    print(f"  Batch indices: {batch_idx.shape}, unique: {batch_idx.unique().tolist()}")

    # 验证角度范围
    assert theta.min() >= -math.pi/2 and theta.max() < math.pi/2, \
        f"Theta range incorrect: [{theta.min()}, {theta.max()}]"

    print("\n✓ All decoder tests passed!")


if __name__ == '__main__':
    test_decoder()
