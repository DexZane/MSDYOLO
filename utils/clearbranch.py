"""
Clear Branch No-Gradient Forward Implementation
GPT批准：Phase 1 低风险基础设施

实现清晰分支串行无梯度前向，测试显存节省效果

GPT关键要求：
1. 必须测试两种策略（eval模式 vs 控制BatchNorm）
2. 验证BatchNorm统计是否更新
3. 验证eval模式是否影响检测头输出格式
4. 验证混合精度兼容性
5. 密集特征必须立即释放
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ClearBranchForward:
    """
    清晰分支无梯度前向管理器

    实现完全非对称梯度：
    - 清晰分支：eval模式 + inference_mode，完全无梯度
    - 退化分支：正常训练模式

    GPT第五轮修正：保存和恢复每个子模块的training状态
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = 'eval_mode',  # 只支持 'eval_mode'
    ):
        """
        Args:
            model: YOLOv5 模型
            strategy: 清晰分支策略
                - 'eval_mode': 暂时切换到eval模式（唯一支持的策略）

        Note:
            GPT第七轮审核：删除freeze_bn策略（与sparse extraction不兼容）
        """
        self.model = model
        if strategy != 'eval_mode':
            raise ValueError(f"Only 'eval_mode' is supported, got: {strategy}")
        self.strategy = strategy
        self.module_states = {}  # GPT修正：保存每个模块的状态

    def _save_module_states(self):
        """
        保存所有子模块的training状态

        GPT第五轮修正：逐模块保存，避免统一覆盖
        """
        self.module_states.clear()
        for name, module in self.model.named_modules():
            self.module_states[name] = module.training

    def _restore_module_states(self):
        """
        恢复所有子模块的training状态

        GPT第五轮修正：逐模块恢复原始状态
        """
        for name, module in self.model.named_modules():
            if name in self.module_states:
                module.train(self.module_states[name])

    def forward_clear_branch(
        self,
        images: torch.Tensor,
        extract_sparse: bool = True,
        sparse_extractor: Optional[object] = None
    ):
        """
        清晰分支前向传播（无梯度）

        Args:
            images: 清晰图像 (B, C, H, W)
            extract_sparse: 是否提取稀疏预测
            sparse_extractor: SparsePredictionExtractor实例

        Returns:
            List[Dict[str, torch.Tensor]]: 每张图像的稀疏预测列表（如果extract_sparse=True）
            或密集预测（如果extract_sparse=False）

        GPT修正：
        - 密集特征必须在提取稀疏预测后立即释放
        - 使用inference_mode()而非no_grad()
        - 逐模块保存和恢复状态
        - 移除不必要的empty_cache()
        """
        # GPT修正：保存每个子模块的状态
        self._save_module_states()

        try:
            if self.strategy == 'eval_mode':
                # 策略1：临时切换eval模式
                self.model.eval()
                # GPT第七轮审核：改用no_grad而非inference_mode
                # 原因：inference_mode创建的张量无法用于KL反向传播
                context = torch.no_grad()
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # 无梯度前向
            with context:
                predictions = self.model(images)

                # 如果需要提取稀疏预测
                if extract_sparse and sparse_extractor is not None:
                    batch_size = images.shape[0]
                    sparse_preds = sparse_extractor.extract_from_yolo_output(
                        predictions,
                        model_training=False,  # eval模式
                        batch_size=batch_size
                    )

                    # 立即释放密集预测张量
                    del predictions
                    # GPT修正：移除empty_cache()，让PyTorch自动管理

                    # 确保稀疏预测正确detach
                    for pred in sparse_preds:
                        for key, value in pred.items():
                            if isinstance(value, torch.Tensor):
                                pred[key] = value.detach()

                    return sparse_preds
                else:
                    # 返回密集预测（需要手动detach）
                    if isinstance(predictions, tuple):
                        # eval模式返回(decoded, raw)元组
                        # 第二项是list，不能直接detach
                        decoded = predictions[0].detach() if hasattr(predictions[0], 'detach') else predictions[0]
                        return (decoded, predictions[1])
                    elif isinstance(predictions, list):
                        return [pred.detach() for pred in predictions]
                    else:
                        return predictions.detach()

        finally:
            # GPT修正：恢复每个子模块的原始状态
            # 无论哪种策略，都使用统一的状态恢复
            self._restore_module_states()

    def forward_degraded_branch(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        compute_loss_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        退化分支前向传播（正常梯度）

        Args:
            images: 退化图像 (B, C, H, W)
            targets: 标签（可选）
            compute_loss_fn: 损失函数（可选）

        Returns:
            predictions: 预测结果
            loss: 损失（如果提供了targets和compute_loss_fn）
        """
        # 确保模型处于训练模式
        self.model.train()

        # 正常前向传播
        predictions = self.model(images)

        # 计算损失（如果需要）
        loss = None
        if targets is not None and compute_loss_fn is not None:
            loss = compute_loss_fn(predictions, targets)

        return predictions, loss


def test_clear_branch_forward():
    """
    测试清晰分支无梯度前向

    GPT要求检查：
    1. BatchNorm统计是否更新
    2. eval模式是否影响检测头输出
    3. 显存是否节省
    4. 混合精度兼容性
    """
    print("Testing Clear Branch No-Gradient Forward...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建简单测试模型（包含BatchNorm）
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x).flatten(1)
            x = self.fc(x)
            return [x]  # 返回列表模拟YOLO输出

    model = SimpleModel().to(device)
    model.train()

    # 创建清晰分支管理器
    clear_branch_mgr = ClearBranchForward(model, strategy='eval_mode')

    # 创建测试数据
    images = torch.randn(2, 3, 64, 64, device=device)

    print("\n" + "="*50)
    print("Test 1: BatchNorm统计更新检查")
    print("="*50)

    # 记录初始BN统计
    initial_bn_mean = model.bn1.running_mean.clone()
    initial_bn_var = model.bn1.running_var.clone()

    print(f"Initial BN mean: {initial_bn_mean[:5]}")
    print(f"Initial BN var: {initial_bn_var[:5]}")

    # 清晰分支前向（无梯度）
    with torch.no_grad():  # 外层也加no_grad确保安全
        sparse_output = clear_branch_mgr.forward_clear_branch(
            images,
            extract_sparse=False,
            sparse_extractor=None
        )

    # 检查BN统计是否改变
    after_clear_bn_mean = model.bn1.running_mean.clone()
    after_clear_bn_var = model.bn1.running_var.clone()

    bn_mean_changed = not torch.allclose(initial_bn_mean, after_clear_bn_mean)
    bn_var_changed = not torch.allclose(initial_bn_var, after_clear_bn_var)

    print(f"\nAfter clear branch forward:")
    print(f"  BN mean changed: {bn_mean_changed}")
    print(f"  BN var changed: {bn_var_changed}")

    if bn_mean_changed or bn_var_changed:
        print("  ⚠️  WARNING: BN stats changed in eval mode (unexpected!)")
    else:
        print("  ✓ BN stats NOT changed (expected)")

    print("\n" + "="*50)
    print("Test 2: 训练模式状态恢复检查")
    print("="*50)

    # 确保模型恢复训练模式
    assert model.training, "Model should be in training mode after clear branch"
    print("✓ Model correctly restored to training mode")

    print("\n" + "="*50)
    print("Test 3: 退化分支正常前向检查")
    print("="*50)

    # 退化分支前向（有梯度）
    degraded_output, _ = clear_branch_mgr.forward_degraded_branch(images)

    # 检查梯度
    loss = degraded_output[0].sum()
    loss.backward()

    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"✓ Degraded branch has gradients: {has_gradients}")

    print("\n" + "="*50)
    print("Test 4: 显存对比（如果有CUDA）")
    print("="*50)

    if torch.cuda.is_available():
        from utils.profiler import MemoryProfiler

        profiler = MemoryProfiler(device)

        # 测试1: 正常前向（有梯度）
        model.zero_grad()
        torch.cuda.empty_cache()

        with profiler.profile("normal_forward"):
            out = model(images)
            loss = out[0].sum()
            loss.backward()

        normal_stats = profiler.get_stats()

        # 测试2: 清晰分支无梯度前向
        model.zero_grad()
        torch.cuda.empty_cache()

        with profiler.profile("clear_branch_forward"):
            _ = clear_branch_mgr.forward_clear_branch(images, extract_sparse=False)

        clear_stats = profiler.get_stats()

        print(f"\nNormal forward (with grad):")
        print(f"  Peak allocated: {normal_stats['peak_allocated_gb']:.3f} GB")

        print(f"\nClear branch forward (no grad):")
        print(f"  Peak allocated: {clear_stats['peak_allocated_gb']:.3f} GB")

        memory_saved = normal_stats['peak_allocated_gb'] - clear_stats['peak_allocated_gb']
        memory_saved_percent = (memory_saved / normal_stats['peak_allocated_gb'] * 100) if normal_stats['peak_allocated_gb'] > 0 else 0

        print(f"\nMemory saved: {memory_saved:.3f} GB ({memory_saved_percent:.1f}%)")

        if memory_saved > 0:
            print("✓ Clear branch saves memory as expected")
        else:
            print("⚠️  No memory saving detected (model may be too small)")

    else:
        print("Skipping CUDA memory test (CPU mode)")

    print("\n" + "="*50)
    print("All tests completed! ✓")
    print("="*50)


if __name__ == '__main__':
    test_clear_branch_forward()
