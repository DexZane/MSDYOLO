"""
MSDYOLO Training Wrapper
将配置系统和基础设施模块集成到训练流程

GPT第五轮要求：
1. 读取YAML配置
2. 所有功能默认关闭
3. 接入train.py
4. 单batch集成测试
5. 默认关闭时与baseline完全一致
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from utils.config import MSDYOLOConfig
from utils.degradation import ImageDegradation
from utils.clearbranch import ClearBranchForward
from utils.sparse import SparsePredictionExtractor
from utils.profiler import MemoryProfiler

logger = logging.getLogger(__name__)


class MSDYOLOTrainer:
    """
    MSDYOLO训练包装器

    集成所有基础设施模块，通过配置控制启用/禁用
    """

    def __init__(
        self,
        model: nn.Module,
        config: MSDYOLOConfig,
        device: torch.device,
    ):
        """
        Args:
            model: YOLOv5模型
            config: MSDYOLO配置
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.device = device

        # 验证配置
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        # 初始化模块（仅在启用时）
        self.degradation = None
        self.clear_branch = None
        self.sparse_extractor = None
        self.profiler = None

        self._setup_modules()

    def _setup_modules(self):
        """根据配置初始化模块"""

        # 退化管线
        if self.config.get('degradation.enabled'):
            logger.info("Initializing degradation pipeline")
            self.degradation = ImageDegradation(
                enable_psf=self.config.get('degradation.psf.enabled'),
                enable_downsample=self.config.get('degradation.downsample.enabled'),
                enable_noise=self.config.get('degradation.noise.enabled'),
                psf_kernel_size=self.config.get('degradation.psf.kernel_size'),
                psf_sigma=self.config.get('degradation.psf.sigma'),
                downsample_scale=self.config.get('degradation.downsample.scale'),
                noise_type=self.config.get('degradation.noise.type'),
                noise_level=self.config.get('degradation.noise.level'),
                upsample_mode=self.config.get('degradation.upsample_mode'),
                seed=self.config.get('degradation.seed'),
            ).to(self.device)

        # 清晰分支
        if self.config.get('clear_branch.enabled'):
            logger.info("Initializing clear branch forward")
            self.clear_branch = ClearBranchForward(
                model=self.model,
                strategy=self.config.get('clear_branch.strategy'),
            )

            # 稀疏预测提取器
            self.sparse_extractor = SparsePredictionExtractor(
                conf_threshold=self.config.get('clear_branch.conf_threshold'),
                top_k=self.config.get('clear_branch.top_k'),
                num_classes=self.model.nc if hasattr(self.model, 'nc') else 16,  # DOTA v1.5
            )

        # Profiler
        if self.config.get('profiling.enabled'):
            logger.info("Initializing profiler")
            self.profiler = MemoryProfiler(self.device)

    def process_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        compute_loss_fn: callable,
    ) -> Dict:
        """
        处理单个batch

        Args:
            images: 输入图像 (B, C, H, W)
            targets: 标签
            compute_loss_fn: 损失函数，返回(loss, loss_items)或loss标量

        Returns:
            结果字典，包含loss、loss_items和预测
        """
        result = {}

        # 1. 退化管线（如果启用）
        if self.degradation is not None:
            images_degraded = self.degradation(images)
        else:
            images_degraded = images

        # 2. 清晰分支（如果启用）
        clear_preds = None
        if self.clear_branch is not None and self.sparse_extractor is not None:
            # 清晰分支无梯度前向
            clear_preds = self.clear_branch.forward_clear_branch(
                images,  # 原始清晰图像
                extract_sparse=True,
                sparse_extractor=self.sparse_extractor
            )
            result['clear_predictions'] = clear_preds

        # 3. 退化分支正常前向
        if self.clear_branch is not None:
            predictions, loss_output = self.clear_branch.forward_degraded_branch(
                images_degraded,
                targets=targets,
                compute_loss_fn=compute_loss_fn
            )
        else:
            # 标准训练（无清晰分支）
            predictions = self.model(images_degraded)
            loss_output = compute_loss_fn(predictions, targets)

        # 处理loss输出格式
        # YOLOv5的ComputeLoss返回(loss, loss_items)元组
        if isinstance(loss_output, tuple):
            loss, loss_items = loss_output
            result['loss'] = loss
            result['loss_items'] = loss_items
        else:
            # 简单标量loss
            result['loss'] = loss_output
            result['loss_items'] = None

        result['predictions'] = predictions

        # 4. 蒸馏损失（Phase 1禁止）
        if self.config.get('distillation.enabled'):
            raise RuntimeError("Phase 1 不允许启用distillation")

        return result

    def is_baseline_mode(self) -> bool:
        """检查是否为baseline模式（所有功能关闭）"""
        return (
            not self.config.get('degradation.enabled') and
            not self.config.get('clear_branch.enabled') and
            not self.config.get('distillation.enabled')
        )


def test_trainer():
    """测试训练包装器"""
    print("Testing MSDYOLO Trainer...")

    device = torch.device('cpu')

    # 创建简单测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.nc = 15  # 类别数

        def forward(self, x):
            return [self.conv(x)]

    model = SimpleModel()

    # 测试1: Baseline模式（所有功能关闭）
    print("\n" + "="*50)
    print("Test 1: Baseline mode (all features disabled)")
    print("="*50)

    config = MSDYOLOConfig()
    config.apply_ablation_mode('baseline')

    trainer = MSDYOLOTrainer(model, config, device)

    assert trainer.is_baseline_mode(), "Should be baseline mode"
    assert trainer.degradation is None, "Degradation should be None"
    assert trainer.clear_branch is None, "Clear branch should be None"
    print("✓ Baseline mode verified")

    # 测试2: 带退化管线
    print("\n" + "="*50)
    print("Test 2: With degradation pipeline")
    print("="*50)

    config2 = MSDYOLOConfig()
    config2.apply_ablation_mode('with_degradation')
    config2.set('degradation.psf.enabled', True)
    config2.set('degradation.downsample.enabled', True)

    trainer2 = MSDYOLOTrainer(model, config2, device)

    assert not trainer2.is_baseline_mode(), "Should not be baseline mode"
    assert trainer2.degradation is not None, "Degradation should be initialized"
    print("✓ Degradation pipeline initialized")

    # 测试3: 处理batch（baseline模式）
    print("\n" + "="*50)
    print("Test 3: Process batch (baseline)")
    print("="*50)

    images = torch.randn(2, 3, 64, 64)
    targets = torch.zeros(10, 6)  # 模拟标签

    def dummy_loss(pred, tgt):
        return torch.tensor(1.0)

    result = trainer.process_batch(images, targets, dummy_loss)

    assert 'loss' in result, "Should have loss"
    assert 'predictions' in result, "Should have predictions"
    assert 'clear_predictions' not in result, "Should not have clear predictions in baseline"
    print("✓ Batch processing works in baseline mode")

    # 测试4: 处理返回元组的loss函数
    print("\n" + "="*50)
    print("Test 4: Process batch with tuple loss")
    print("="*50)

    def tuple_loss(pred, tgt):
        loss = torch.tensor(1.5)
        loss_items = torch.tensor([0.5, 0.5, 0.5])
        return loss, loss_items

    result2 = trainer.process_batch(images, targets, tuple_loss)

    assert 'loss' in result2, "Should have loss"
    assert 'loss_items' in result2, "Should have loss_items"
    assert isinstance(result2['loss'], torch.Tensor), "Loss should be tensor"
    assert result2['loss'].dim() == 0, "Loss should be scalar"
    print(f"✓ Loss: {result2['loss'].item()}, Loss items: {result2['loss_items']}")

    # 测试5: 验证loss可以反向传播
    print("\n" + "="*50)
    print("Test 5: Loss backward")
    print("="*50)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    # 使用依赖模型输出的loss函数
    def real_loss(pred, tgt):
        # 简单的MSE loss
        if isinstance(pred, list):
            pred = pred[0]
        loss = pred.mean()  # 依赖模型输出
        loss_items = torch.tensor([loss.item(), 0.0, 0.0])
        return loss, loss_items

    result3 = trainer.process_batch(images, targets, real_loss)
    loss = result3['loss']

    try:
        loss.backward()
        print("✓ Loss backward successful")
        print(f"✓ Loss requires_grad: {loss.requires_grad}")
    except Exception as e:
        raise AssertionError(f"Loss backward failed: {e}")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == '__main__':
    test_trainer()
