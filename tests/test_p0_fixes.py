"""
MSDYOLO Phase 1 P0修正测试套件
覆盖GPT第六轮审核要求
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.decoder import YOLOOutputDecoder
from utils.sparse import SparsePredictionExtractor, PredictionMatcher
from utils.degradation import ImageDegradation
from utils.config import MSDYOLOConfig
from utils.trainer import MSDYOLOTrainer
from utils.clearbranch import ClearBranchForward


class TestDecoder:
    """测试YOLO输出解码器"""

    def test_csl_angle_decode(self):
        """测试CSL角度解码正确性"""
        decoder = YOLOOutputDecoder(num_classes=16)
        B, N = 1, 100
        no = 5 + 15 + 180

        decoded_preds = torch.rand(B, N, no)
        decoded_preds[..., :4] = decoded_preds[..., :4] * 1024

        boxes, theta, obj, cls_prob, cls_id, batch_idx = decoder.decode(
            (decoded_preds, None),
            model_training=False
        )

        assert theta.min() >= -math.pi / 2, f"Min angle {theta.min()} < -π/2"
        assert theta.max() < math.pi / 2, f"Max angle {theta.max()} >= π/2"

    def test_batch_size_support(self):
        """GPT要求：测试batch size 1和2支持"""
        decoder = YOLOOutputDecoder(num_classes=16)
        no = 5 + 15 + 180

        # Batch size 1
        decoded_preds_1 = torch.rand(1, 50, no)
        decoded_preds_1[..., :4] = decoded_preds_1[..., :4] * 1024
        boxes1, theta1, obj1, cls_prob1, cls_id1, batch_idx1 = decoder.decode(
            (decoded_preds_1, None), model_training=False
        )
        assert boxes1.shape[0] == 50
        assert batch_idx1.unique().tolist() == [0]

        # Batch size 2
        decoded_preds_2 = torch.rand(2, 50, no)
        decoded_preds_2[..., :4] = decoded_preds_2[..., :4] * 1024
        boxes2, theta2, obj2, cls_prob2, cls_id2, batch_idx2 = decoder.decode(
            (decoded_preds_2, None), model_training=False
        )
        assert boxes2.shape[0] == 100
        assert set(batch_idx2.unique().tolist()) == {0, 1}

    def test_csl_boundary_values(self):
        """GPT要求：测试CSL边界值"""
        decoder = YOLOOutputDecoder(num_classes=16)
        B, N = 1, 3
        no = 5 + 16 + 180  # 修正：16类

        decoded_preds = torch.zeros(B, N, no)
        decoded_preds[..., :4] = 512.0

        # index 0 -> -π/2
        decoded_preds[0, 0, 5+16:5+16+180] = 0.0
        decoded_preds[0, 0, 5+16] = 100.0

        # index 90 -> 0
        decoded_preds[0, 1, 5+16:5+16+180] = 0.0
        decoded_preds[0, 1, 5+16+90] = 100.0

        # index 179 -> 89π/180
        decoded_preds[0, 2, 5+16:5+16+180] = 0.0
        decoded_preds[0, 2, 5+16+179] = 100.0

        boxes, theta, obj, cls_prob, cls_id, batch_idx = decoder.decode(
            (decoded_preds, None), model_training=False
        )

        assert abs(theta[0].item() - (-math.pi/2)) < 1e-5
        assert abs(theta[1].item() - 0.0) < 1e-5
        expected_179 = (179 - 90) / 180.0 * math.pi
        assert abs(theta[2].item() - expected_179) < 1e-5


class TestSparsePrediction:
    """测试稀疏预测提取和匹配"""

    def test_one_to_one_matching_nonempty(self):
        """GPT要求：非空确定性匹配测试"""
        nc = 15
        device = torch.device('cpu')

        clear_boxes = torch.tensor([
            [100.0, 100.0, 50.0, 30.0, 0.5],
            [200.0, 200.0, 60.0, 40.0, 0.3],
            [300.0, 300.0, 70.0, 50.0, 0.1],
        ], device=device)

        clear_preds = {
            'boxes': clear_boxes,
            'scores': torch.tensor([0.9, 0.8, 0.7], device=device),
            'classes': torch.zeros((3, nc), device=device),
            'class_ids': torch.tensor([0, 1, 2], device=device),
            'valid_mask': torch.ones(3, dtype=torch.bool, device=device),
        }

        degraded_boxes = torch.tensor([
            [102.0, 101.0, 51.0, 31.0, 0.52],
            [202.0, 201.0, 62.0, 41.0, 0.28],
            [500.0, 500.0, 80.0, 60.0, 0.15],
        ], device=device)

        degraded_preds = {
            'boxes': degraded_boxes,
            'scores': clear_preds['scores'],
            'classes': clear_preds['classes'],
            'class_ids': torch.tensor([0, 1, 0], device=device),
            'valid_mask': torch.ones(3, dtype=torch.bool, device=device),
        }

        matcher = PredictionMatcher(match_threshold=0.15, use_class_filter=True)
        clear_idx, degraded_idx = matcher.match_predictions(clear_preds, degraded_preds)

        assert len(clear_idx) > 0, "Should find at least one match"
        assert len(clear_idx) == len(set(clear_idx.tolist()))
        assert len(degraded_idx) == len(set(degraded_idx.tolist()))
        assert len(clear_idx) == 2
        assert 0 in clear_idx.tolist() and 0 in degraded_idx.tolist()
        assert 1 in clear_idx.tolist() and 1 in degraded_idx.tolist()

    def test_class_filtering(self):
        """GPT要求：类别过滤测试"""
        nc = 15
        device = torch.device('cpu')

        clear_preds = {
            'boxes': torch.tensor([[100, 100, 50, 30, 0.5]], device=device),
            'scores': torch.tensor([0.9], device=device),
            'classes': torch.zeros((1, nc), device=device),
            'class_ids': torch.tensor([0], device=device),
            'valid_mask': torch.ones(1, dtype=torch.bool, device=device),
        }

        degraded_preds = {
            'boxes': torch.tensor([[101, 101, 51, 31, 0.51]], device=device),
            'scores': torch.tensor([0.9], device=device),
            'classes': torch.zeros((1, nc), device=device),
            'class_ids': torch.tensor([1], device=device),
            'valid_mask': torch.ones(1, dtype=torch.bool, device=device),
        }

        matcher_with = PredictionMatcher(match_threshold=0.3, use_class_filter=True)
        clear_idx_with, degraded_idx_with = matcher_with.match_predictions(clear_preds, degraded_preds)
        assert len(clear_idx_with) == 0

        matcher_without = PredictionMatcher(match_threshold=0.3, use_class_filter=False)
        clear_idx_without, degraded_idx_without = matcher_without.match_predictions(clear_preds, degraded_preds)
        assert len(clear_idx_without) == 1

    def test_empty_input(self):
        """GPT要求：空输入测试"""
        nc = 15
        device = torch.device('cpu')

        extractor = SparsePredictionExtractor(num_classes=nc)
        empty_preds = extractor._empty_predictions(device)

        clear_preds = {
            'boxes': torch.tensor([[100, 100, 50, 30, 0.5]], device=device),
            'scores': torch.tensor([0.9], device=device),
            'classes': torch.zeros((1, nc), device=device),
            'class_ids': torch.tensor([0], device=device),
            'valid_mask': torch.ones(1, dtype=torch.bool, device=device),
        }

        matcher = PredictionMatcher()
        clear_idx, degraded_idx = matcher.match_predictions(empty_preds, clear_preds)
        assert len(clear_idx) == 0


class TestDegradation:
    """测试退化管线"""

    def test_identity_when_disabled(self):
        """测试恒等映射"""
        degradation = ImageDegradation(
            enable_psf=False,
            enable_downsample=False,
            enable_noise=False
        )

        # 使用[0,1]范围的输入
        x = torch.rand(2, 3, 64, 64)
        y = degradation(x)

        assert torch.allclose(x, y)

    def test_parameter_validation(self):
        """GPT要求：参数验证"""
        with pytest.raises(ValueError, match="positive odd"):
            ImageDegradation(enable_psf=True, psf_kernel_size=4)

        with pytest.raises(ValueError, match="positive"):
            ImageDegradation(enable_psf=True, psf_sigma=-1.0)

        with pytest.raises(ValueError, match="> 1"):
            ImageDegradation(enable_downsample=True, downsample_scale=0.5)

    def test_noise_reproducibility(self):
        """GPT要求：噪声可复现性（修正版）"""
        deg1 = ImageDegradation(enable_noise=True, noise_level=0.1, seed=42)
        deg2 = ImageDegradation(enable_noise=True, noise_level=0.1, seed=42)

        x = torch.randn(1, 3, 64, 64)

        # 相同种子应产生相同序列
        y1_batch1 = deg1(x)
        y2_batch1 = deg2(x)
        assert torch.allclose(y1_batch1, y2_batch1)

        # 连续batch应产生不同噪声
        y1_batch2 = deg1(x)
        assert not torch.allclose(y1_batch1, y1_batch2)


class TestClearBranch:
    """测试清晰分支"""



class TestConfig:
    """测试配置系统"""

    def test_baseline_mode(self):
        """测试baseline模式"""
        config = MSDYOLOConfig()
        config.apply_ablation_mode('baseline')

        assert not config.get('degradation.enabled')
        assert not config.get('clear_branch.enabled')
        assert not config.get('distillation.enabled')

    def test_phase1_distillation_validation(self):
        """测试Phase 1禁止蒸馏"""
        config = MSDYOLOConfig()
        config.set('distillation.enabled', True)

        assert not config.validate()


class TestTrainer:
    """测试训练包装器"""

    def test_baseline_consistency(self):
        """GPT要求：baseline模块禁用测试"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.nc = 15

            def forward(self, x):
                return [self.conv(x)]

        model = SimpleModel()
        config = MSDYOLOConfig()
        config.apply_ablation_mode('baseline')

        trainer = MSDYOLOTrainer(model, config, torch.device('cpu'))

        assert trainer.is_baseline_mode()
        assert trainer.degradation is None
        assert trainer.clear_branch is None

    def test_loss_tuple_handling(self):
        """GPT要求：真实loss tuple适配测试"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.nc = 15

            def forward(self, x):
                return [self.conv(x)]

        model = SimpleModel()
        config = MSDYOLOConfig()
        config.apply_ablation_mode('baseline')

        trainer = MSDYOLOTrainer(model, config, torch.device('cpu'))

        images = torch.randn(2, 3, 64, 64)
        targets = torch.zeros(10, 6)

        def tuple_loss(pred, tgt):
            if isinstance(pred, list):
                pred = pred[0]
            loss = pred.mean()
            loss_items = torch.tensor([loss.item(), 0.0, 0.0])
            return loss, loss_items

        result = trainer.process_batch(images, targets, tuple_loss)

        assert 'loss' in result
        assert 'loss_items' in result
        assert isinstance(result['loss'], torch.Tensor)
        assert result['loss'].dim() == 0
        assert result['loss'].requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
