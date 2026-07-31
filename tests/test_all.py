"""
MSDYOLO Unit Tests
GPT第五轮要求：正式测试框架
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.decoder import YOLOOutputDecoder
from utils.sparse import SparsePredictionExtractor, PredictionMatcher
from utils.degradation import ImageDegradation
from utils.config import MSDYOLOConfig
from utils.trainer import MSDYOLOTrainer


class TestDecoder:
    def test_csl_angle_decode(self):
        """测试CSL角度解码正确性"""
        decoder = YOLOOutputDecoder(num_classes=16)

        # 模拟eval输出
        B, N, no = 1, 10, 5 + 16 + 180  # 16类
        preds = torch.rand(B, N, no)

        boxes, theta, obj, cls_prob, cls_id, batch_indices = decoder.decode((preds, None), model_training=False)

        # 验证角度范围 [-π/2, π/2)
        assert theta.min() >= -torch.pi/2
        assert theta.max() < torch.pi/2


class TestSparsePrediction:
    def test_one_to_one_matching(self):
        """测试一对一匹配"""
        matcher = PredictionMatcher(match_threshold=0.5)

        # 创建测试预测
        clear = {
            'boxes': torch.rand(10, 5),
            'scores': torch.rand(10),
            'classes': torch.rand(10, 15),
            'class_ids': torch.randint(0, 15, (10,)),
            'valid_mask': torch.ones(10, dtype=torch.bool),
        }

        degraded = {
            'boxes': clear['boxes'] + torch.randn(10, 5) * 10,
            'scores': clear['scores'],
            'classes': clear['classes'],
            'class_ids': clear['class_ids'],
            'valid_mask': clear['valid_mask'],
        }

        c_idx, d_idx = matcher.match_predictions(clear, degraded)

        # 验证一对一
        assert len(c_idx) == len(set(c_idx.tolist()))
        assert len(d_idx) == len(set(d_idx.tolist()))


class TestDegradation:
    def test_identity_when_disabled(self):
        """测试关闭时恒等映射"""
        degrader = ImageDegradation(
            enable_psf=False,
            enable_downsample=False,
            enable_noise=False
        )

        x = torch.rand(1, 3, 64, 64)
        y = degrader(x)

        assert torch.allclose(x, y)

    def test_parameter_validation(self):
        """测试参数检查"""
        with pytest.raises(ValueError):
            ImageDegradation(psf_kernel_size=4)  # 必须为奇数

        with pytest.raises(ValueError):
            ImageDegradation(psf_sigma=-1)  # 必须>0


class TestConfig:
    def test_baseline_mode(self):
        """测试baseline模式"""
        config = MSDYOLOConfig()
        config.apply_ablation_mode('baseline')

        assert not config.get('degradation.enabled')
        assert not config.get('clear_branch.enabled')
        assert not config.get('distillation.enabled')

    def test_phase1_distillation_validation(self):
        """测试Phase 1不允许蒸馏"""
        config = MSDYOLOConfig()
        config.set('experiment.phase', 1)
        config.set('distillation.enabled', True)

        assert not config.validate()


class TestTrainer:
    def test_baseline_consistency(self):
        """测试baseline模式一致性"""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.nc = 16  # 16类

            def forward(self, x):
                return [self.conv(x)]

        model = DummyModel()
        config = MSDYOLOConfig()
        config.apply_ablation_mode('baseline')

        trainer = MSDYOLOTrainer(model, config, torch.device('cpu'))

        assert trainer.is_baseline_mode()
        assert trainer.degradation is None
        assert trainer.clear_branch is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
