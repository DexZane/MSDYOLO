"""
CPU Baseline Equivalence Test
GPT第八轮要求：验证MSDYOLOTrainer包装器关闭时与普通路径的等价性
必须实际调用MSDYOLOTrainer.process_batch()
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
    def __init__(self, nc=16):
        super().__init__()
        self.nc = nc
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # 输出维度：5 + nc + 180
        self.fc = nn.Linear(32 * 64 * 64, 5 + nc + 180)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # 返回 (B, 5+nc+180) 作为单个锚点的输出
        return out.unsqueeze(1)  # (B, 1, 5+nc+180)


class SimpleLoss(nn.Module):
    """简化的损失函数"""
    def __call__(self, predictions, targets):
        # predictions: (B, N, 5+nc+180)
        # targets: list of tensors
        # 简单地返回预测的均值作为loss
        return predictions.abs().mean()


def test_baseline_equivalence_loss():
    """
    GPT要求：验证MSDYOLOTrainer包装器关闭所有功能时，loss与普通路径等价
    """
    device = torch.device('cpu')

    # 创建baseline配置
    config = MSDYOLOConfig()
    config.apply_ablation_mode('baseline')

    # 验证配置确实关闭了所有功能
    assert not config.get('degradation.enabled')
    assert not config.get('clear_branch.enabled')
    assert not config.get('distillation.enabled')

    # 创建模型和损失函数
    model = SimpleYOLOModel(nc=16)
    model.to(device)
    model.train()

    compute_loss = SimpleLoss()

    # 创建输入
    images = torch.randn(2, 3, 64, 64, device=device)
    targets = [torch.zeros(0, 6) for _ in range(2)]  # 空targets

    # 1. 普通路径：直接前向
    with torch.no_grad():
        normal_output = model(images)
        normal_loss = compute_loss(normal_output, targets)

    # 2. 包装器路径：通过MSDYOLOTrainer.process_batch
    trainer = MSDYOLOTrainer(model, config, device)

    with torch.no_grad():
        result = trainer.process_batch(images, targets, compute_loss)
        wrapper_loss = result['loss']

    # 3. 验证loss等价
    assert torch.allclose(normal_loss, wrapper_loss, atol=1e-6), \
        f"Loss不等价: normal={normal_loss.item():.6f}, wrapper={wrapper_loss.item():.6f}"

    print(f"✓ Baseline loss equivalence: {normal_loss.item():.6f} == {wrapper_loss.item():.6f}")


def test_baseline_equivalence_gradients():
    """
    GPT要求：验证MSDYOLOTrainer包装器关闭时梯度等价
    """
    device = torch.device('cpu')

    # 创建baseline配置
    config = MSDYOLOConfig()
    config.apply_ablation_mode('baseline')

    # 创建两个相同权重的模型
    model1 = SimpleYOLOModel(nc=16)
    model2 = SimpleYOLOModel(nc=16)
    model2.load_state_dict(model1.state_dict())

    model1.to(device)
    model2.to(device)
    model1.train()
    model2.train()

    compute_loss = SimpleLoss()

    # 相同输入
    images = torch.randn(2, 3, 64, 64, device=device)
    targets = [torch.zeros(0, 6) for _ in range(2)]

    # 1. 普通路径
    output1 = model1(images)
    loss1 = compute_loss(output1, targets)
    loss1.backward()

    # 2. 包装器路径
    trainer = MSDYOLOTrainer(model2, config, device)
    result = trainer.process_batch(images, targets, compute_loss)
    loss2 = result['loss']
    loss2.backward()

    # 3. 验证梯度等价
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(),
                                                  model2.named_parameters()):
        assert name1 == name2
        if param1.grad is not None and param2.grad is not None:
            max_diff = torch.max(torch.abs(param1.grad - param2.grad)).item()
            assert torch.allclose(param1.grad, param2.grad, atol=1e-5), \
                f"梯度不等价 {name1}: max_diff={max_diff:.2e}"

    print("✓ Baseline gradient equivalence verified")


def test_baseline_equivalence_batchnorm_buffers():
    """
    GPT要求：验证MSDYOLOTrainer包装器关闭时BatchNorm buffer等价
    """
    device = torch.device('cpu')

    # 创建baseline配置
    config = MSDYOLOConfig()
    config.apply_ablation_mode('baseline')

    # 创建两个相同权重的模型
    model1 = SimpleYOLOModel(nc=16)
    model2 = SimpleYOLOModel(nc=16)
    model2.load_state_dict(model1.state_dict())

    model1.to(device)
    model2.to(device)
    model1.train()
    model2.train()

    compute_loss = SimpleLoss()

    # 相同输入
    images = torch.randn(2, 3, 64, 64, device=device)
    targets = [torch.zeros(0, 6) for _ in range(2)]

    # 1. 普通路径
    _ = model1(images)

    # 2. 包装器路径
    trainer = MSDYOLOTrainer(model2, config, device)
    _ = trainer.process_batch(images, targets, compute_loss)

    # 3. 验证BatchNorm的running_mean和running_var等价
    for (name1, module1), (name2, module2) in zip(model1.named_modules(),
                                                    model2.named_modules()):
        if isinstance(module1, nn.BatchNorm2d):
            assert torch.allclose(module1.running_mean, module2.running_mean, atol=1e-6), \
                f"BN running_mean不等价 {name1}"
            assert torch.allclose(module1.running_var, module2.running_var, atol=1e-6), \
                f"BN running_var不等价 {name1}"

    print("✓ Baseline BatchNorm buffer equivalence verified")


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

    print("✓ Degradation changes input when enabled")


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

    print("✓ Degradation is identity when disabled")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
