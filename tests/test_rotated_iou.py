"""
测试旋转框IoU计算的正确性
"""

import pytest
import torch
import numpy as np
import math
from utils.nms_rotated_pure import box_iou_rotated, obb_nms_python, obb_nms_per_class


class TestRotatedIoU:
    """测试旋转框IoU计算"""

    def test_identical_boxes(self):
        """相同框的IoU应该为1.0"""
        box = (50, 50, 20, 10, 0.0)
        iou = box_iou_rotated(box, box)
        assert 0.99 < iou <= 1.0, f"相同框IoU应为1.0，实际为{iou}"

    def test_identical_rotated_boxes(self):
        """相同旋转框的IoU应该为1.0"""
        box = (50, 50, 20, 10, math.pi / 4)  # 45度旋转
        iou = box_iou_rotated(box, box)
        assert 0.99 < iou <= 1.0, f"相同旋转框IoU应为1.0，实际为{iou}"

    def test_separated_boxes(self):
        """完全分离的框IoU应该为0.0"""
        box1 = (10, 10, 5, 5, 0.0)
        box2 = (100, 100, 5, 5, 0.0)
        iou = box_iou_rotated(box1, box2)
        assert iou == 0.0, f"分离框IoU应为0.0，实际为{iou}"

    def test_iou_symmetry(self):
        """IoU(A,B) = IoU(B,A)"""
        box1 = (50, 50, 20, 10, 0.3)
        box2 = (55, 52, 18, 12, -0.2)
        iou1 = box_iou_rotated(box1, box2)
        iou2 = box_iou_rotated(box2, box1)
        assert abs(iou1 - iou2) < 1e-6, f"IoU不对称: {iou1} vs {iou2}"

    def test_iou_range(self):
        """所有IoU值应该在[0, 1]范围内"""
        test_cases = [
            ((50, 50, 20, 10, 0.0), (55, 52, 18, 12, 0.0)),
            ((50, 50, 20, 10, math.pi/4), (55, 52, 18, 12, -math.pi/4)),
            ((50, 50, 30, 5, 0.0), (55, 52, 5, 30, math.pi/2)),
            ((10, 10, 100, 50, 0.5), (15, 12, 90, 45, -0.3)),
        ]

        for box1, box2 in test_cases:
            iou = box_iou_rotated(box1, box2)
            assert 0.0 <= iou <= 1.0, f"IoU={iou}超出[0,1]范围，box1={box1}, box2={box2}"

    def test_axis_aligned_boxes(self):
        """轴对齐框的IoU应该与水平框IoU一致"""
        box1 = (50, 50, 20, 10, 0.0)
        box2 = (55, 52, 18, 12, 0.0)

        iou = box_iou_rotated(box1, box2)

        # 手动计算轴对齐框IoU
        x1_min, y1_min = 50 - 10, 50 - 5
        x1_max, y1_max = 50 + 10, 50 + 5
        x2_min, y2_min = 55 - 9, 52 - 6
        x2_max, y2_max = 55 + 9, 52 + 6

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = 20 * 10
            area2 = 18 * 12
            union_area = area1 + area2 - inter_area
            expected_iou = inter_area / union_area

            assert abs(iou - expected_iou) < 0.05, f"轴对齐IoU不一致: {iou} vs {expected_iou}"

    def test_rotated_90_degree(self):
        """90度旋转的细长框（不同尺寸）"""
        box1 = (50, 50, 40, 10, 0.0)  # 水平 40x10
        box2 = (50, 50, 20, 30, math.pi/2)  # 垂直 20x30（旋转90度后不同）

        iou = box_iou_rotated(box1, box2)

        # 两个框有重叠（中心相同）但不完全相同
        assert 0.0 < iou < 1.0, f"90度旋转框IoU应在(0,1)之间，实际为{iou}"
        assert 0.0 <= iou <= 1.0, f"IoU={iou}超出[0,1]范围"


class TestRotatedNMS:
    """测试旋转框NMS"""

    def test_nms_suppresses_duplicates(self):
        """NMS应该抑制重复框"""
        # 3个几乎相同的框
        dets = torch.tensor([
            [50, 50, 20, 10, 0.0],
            [51, 50, 20, 10, 0.0],
            [50, 51, 20, 10, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = obb_nms_python(dets, scores, iou_thr=0.5)

        # 应该只保留1个框（最高分）
        assert len(keep) == 1, f"应该只保留1个框，实际保留{len(keep)}个"
        assert keep[0] == 0, f"应该保留最高分框，实际保留索引{keep[0]}"

    def test_nms_keeps_separated_boxes(self):
        """NMS应该保留分离的框"""
        dets = torch.tensor([
            [10, 10, 5, 5, 0.0],
            [100, 100, 5, 5, 0.0],
            [200, 200, 5, 5, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = obb_nms_python(dets, scores, iou_thr=0.5)

        # 应该保留所有3个框
        assert len(keep) == 3, f"应该保留3个框，实际保留{len(keep)}个"

    def test_nms_score_order(self):
        """NMS应该按分数降序处理"""
        dets = torch.tensor([
            [50, 50, 20, 10, 0.0],
            [51, 50, 20, 10, 0.0],
        ])
        scores = torch.tensor([0.9, 0.95])  # 第二个分数更高

        keep = obb_nms_python(dets, scores, iou_thr=0.5)

        # 应该保留分数最高的框（索引1）
        assert keep[0] == 1, f"应该保留最高分框（索引1），实际保留{keep[0]}"

    def test_nms_per_class_no_cross_suppression(self):
        """不同类别的框不应该互相抑制"""
        predictions = torch.tensor([
            [50, 50, 20, 10, 0.0, 0.9],  # 类别0
            [51, 50, 20, 10, 0.0, 0.8],  # 类别1
        ])
        class_ids = torch.tensor([0, 1])

        keep = obb_nms_per_class(predictions, class_ids, iou_threshold=0.5)

        # 应该保留两个框（不同类别）
        assert len(keep) == 2, f"不同类别应该都保留，实际保留{len(keep)}个"

    def test_empty_input(self):
        """空输入应该返回空结果"""
        dets = torch.zeros((0, 5))
        scores = torch.zeros(0)

        keep = obb_nms_python(dets, scores, iou_thr=0.5)

        assert len(keep) == 0, f"空输入应返回空结果，实际返回{len(keep)}个"

    def test_single_box(self):
        """单个框应该被保留"""
        dets = torch.tensor([[50, 50, 20, 10, 0.0]])
        scores = torch.tensor([0.9])

        keep = obb_nms_python(dets, scores, iou_thr=0.5)

        assert len(keep) == 1, f"单个框应被保留"
        assert keep[0] == 0


class TestIoUEdgeCases:
    """测试IoU边界情况"""

    def test_zero_area_box(self):
        """零面积框"""
        box1 = (50, 50, 0, 0, 0.0)
        box2 = (50, 50, 20, 10, 0.0)

        iou = box_iou_rotated(box1, box2)
        assert iou == 0.0, f"零面积框IoU应为0.0，实际为{iou}"

    def test_very_thin_box(self):
        """极细的框"""
        box1 = (50, 50, 100, 0.1, 0.0)
        box2 = (50, 50, 100, 0.1, 0.0)

        iou = box_iou_rotated(box1, box2)
        assert 0.99 < iou <= 1.0, f"相同细框IoU应接近1.0，实际为{iou}"

    def test_large_angle_difference(self):
        """大角度差异"""
        box1 = (50, 50, 30, 10, 0.0)
        box2 = (50, 50, 30, 10, math.pi/2 - 0.01)  # 接近90度

        iou = box_iou_rotated(box1, box2)
        assert 0.0 <= iou <= 1.0, f"IoU应在[0,1]范围内，实际为{iou}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
