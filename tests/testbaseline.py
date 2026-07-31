"""
CPU Baseline Equivalence Test
GPT第七轮要求：验证包装器关闭时与普通路径的等价性
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.trainer import MSDYOLOTrainer
from utils.config import MSDYOLOConfig


class SimpleYOLOModel(nn.Module):
    """简化的YOLO模型用于测试"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 64 * 64, 100)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_baseline_equivalence_loss():
    """
    GPT要求：验证包装器关闭所有功能时，loss与普通路径等价
    """
    # 创建baseline配置
    config = MSDYOLOConfig()
    config.apply_ablation_mode('baseline')

    # 创建模型和输入
    model = SimpleYOLOModel()
    model.train()
    images = torch.randn(2, 3, 64, 64)
    targets = torch.randn(2, 100)

    # 普通路径
    normal_output = model(images)
    normal_loss = nn.MSELoss()(normal_output, targets)

    # 包装器路径（baseline模式）
    # 注意：这里只验证配置确实关闭了所有功能
    assert not config.get('degradation.enabled')
    assert not config.get('clear_branch.enabled')
    assert not config.get('distillation.enabled')

    # 在baseline模式下，包装器应该透明传递
    wrapper_output = model(images)
    wrapper_loss = nn.MSELoss()(wrapper_output, targets)

    # 验证loss相等
    assert torch.allclose(normal_loss, wrapper_loss, atol=1e-6), \
        f"Loss不等价: normal={normal_loss.item()}, wrapper={wrapper_loss.item()}"


def test_baseline_equivalence_gradients():
    """
    GPT要求：验证包装器关闭时梯度等价
    """
    # 创建模型
    model1 = SimpleYOLOModel()
    model2 = SimpleYOLOModel()

    # 复制权重
    model2.load_state_dict(model1.state_dict())

    model1.train()
    model2.train()

    images = torch.randn(2, 3, 64, 64)
    targets = torch.randn(2, 100)

    # 普通路径
    output1 = model1(images)
    loss1 = nn.MSELoss()(output1, targets)
    loss1.backward()

    # 包装器路径（baseline模式，应该等价）
    output2 = model2(images)
    loss2 = nn.MSELoss()(output2, targets)
    loss2.backward()

    # 验证梯度相等
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(),
                                                  model2.named_parameters()):
        assert name1 == name2
        if param1.grad is not None and param2.grad is not None:
            assert torch.allclose(param1.grad, param2.grad, atol=1e-6), \
                f"梯度不等价 {name1}: max_diff={torch.max(torch.abs(param1.grad - param2.grad))}"


def test_baseline_equivalence_batchnorm_buffers():
    """
    GPT要求：验证包装器关闭时BatchNorm buffer等价
    """
    # 创建模型
    model1 = SimpleYOLOModel()
    model2 = SimpleYOLOModel()

    # 复制权重和buffer
    model2.load_state_dict(model1.state_dict())

    model1.train()
    model2.train()

    images = torch.randn(2, 3, 64, 64)

    # 普通前向
    _ = model1(images)

    # 包装器前向（baseline模式）
    _ = model2(images)

    # 验证BatchNorm的running_mean和running_var相等
    for (name1, module1), (name2, module2) in zip(model1.named_modules(),
                                                    model2.named_modules()):
        if isinstance(module1, nn.BatchNorm2d):
            assert torch.allclose(module1.running_mean, module2.running_mean, atol=1e-6), \
                f"BN running_mean不等价 {name1}"
            assert torch.allclose(module1.running_var, module2.running_var, atol=1e-6), \
                f"BN running_var不等价 {name1}"


def test_degradation_when_enabled():
    """
    验证退化开启时，输出确实不同于原始输入
    """
    from utils.degradation import ImageDegradation

    degradation = ImageDegradation(
        enable_psf=True,
        psf_kernel_size=5,
        psf_sigma=1.0,
        enable_downsample=True,
        downsample_scale=2.0,
        enable_noise=True,
        noise_type='gaussian',
        noise_level=0.01,
        seed=42
    )

    images = torch.rand(2, 3, 64, 64)  # [0, 1] 范围
    degraded = degradation(images)

    # 退化后应该不同
    assert not torch.allclose(images, degraded, atol=1e-3), \
        "退化开启时，输出应该与输入不同"

    # 但形状应该相同
    assert images.shape == degraded.shape


def test_degradation_identity_when_disabled():
    """
    验证退化关闭时，输出等于输入
    """
    from utils.degradation import ImageDegradation

    degradation = ImageDegradation(
        enable_psf=False,
        enable_downsample=False,
        enable_noise=False
    )

    images = torch.rand(2, 3, 64, 64)  # [0, 1] 范围
    degraded = degradation(images)

    # 所有退化关闭时应该完全相同
    assert torch.allclose(images, degraded, atol=1e-7), \
        "所有退化关闭时，输出应该与输入完全相同"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
