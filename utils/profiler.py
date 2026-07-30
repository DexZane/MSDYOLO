"""
Memory and Speed Profiler for MSDYOLO
GPT批准：Phase 1 低风险基础设施
"""

import torch
import time
from contextlib import contextmanager
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """峰值显存和速度测量工具"""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda_available = torch.cuda.is_available() and device.type == 'cuda'
        self.reset()

    def reset(self):
        """重置峰值统计"""
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.start_time = None
        self.end_time = None

        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

    @contextmanager
    def profile(self, name: str = "operation"):
        """
        上下文管理器：测量代码块的显存和时间

        用法:
            profiler = MemoryProfiler(device)
            with profiler.profile("forward"):
                output = model(input)
            stats = profiler.get_stats()
        """
        self.reset()
        self.start_time = time.time()

        if self.cuda_available:
            torch.cuda.synchronize(self.device)

        try:
            yield self
        finally:
            if self.cuda_available:
                torch.cuda.synchronize(self.device)

            self.end_time = time.time()

            if self.cuda_available:
                self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
                self.peak_reserved = torch.cuda.max_memory_reserved(self.device)

            elapsed = self.end_time - self.start_time
            logger.info(f"[{name}] Time: {elapsed:.3f}s, "
                       f"Peak Allocated: {self.peak_allocated / 1024**3:.3f}GB, "
                       f"Peak Reserved: {self.peak_reserved / 1024**3:.3f}GB")

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        elapsed = (self.end_time - self.start_time) if self.end_time else 0

        return {
            'time_seconds': elapsed,
            'peak_allocated_gb': self.peak_allocated / 1024**3 if self.cuda_available else 0,
            'peak_reserved_gb': self.peak_reserved / 1024**3 if self.cuda_available else 0,
            'peak_allocated_bytes': self.peak_allocated,
            'peak_reserved_bytes': self.peak_reserved,
        }

    def log_current_memory(self, tag: str = ""):
        """记录当前显存使用"""
        if not self.cuda_available:
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        logger.info(f"[{tag}] Current Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB")


class TrainingProfiler:
    """训练过程显存和速度测量"""

    def __init__(self, device: torch.device):
        self.device = device
        self.profiler = MemoryProfiler(device)
        self.batch_times = []
        self.baseline_stats = None
        self.degradation_stats = None
        self.clear_branch_stats = None
        self.distillation_stats = None

    def profile_batch(self, batch_idx: int, batch_fn, *args, **kwargs):
        """
        测量单个batch的训练

        Args:
            batch_idx: batch索引
            batch_fn: 训练函数
            *args, **kwargs: 传递给batch_fn的参数

        Returns:
            batch_fn的返回值
        """
        with self.profiler.profile(f"batch_{batch_idx}"):
            result = batch_fn(*args, **kwargs)

        stats = self.profiler.get_stats()
        self.batch_times.append(stats['time_seconds'])

        return result, stats

    def get_average_stats(self, skip_first: int = 5, batch_size: int = 1) -> Dict[str, float]:
        """
        获取平均统计（跳过前几个batch的warmup）

        Args:
            skip_first: 跳过前N个batch
            batch_size: batch大小（用于计算吞吐量）

        GPT第五轮修正：吞吐量 = batch_size / avg_time (samples/sec)
        """
        if len(self.batch_times) <= skip_first:
            return {'avg_time_seconds': 0, 'throughput_samples_per_sec': 0}

        valid_times = self.batch_times[skip_first:]
        avg_time = sum(valid_times) / len(valid_times)

        # 修正：吞吐量是样本数/秒，不是batch数/秒
        throughput = batch_size / avg_time if avg_time > 0 else 0

        return {
            'avg_time_seconds': avg_time,
            'throughput_samples_per_sec': throughput,  # 修正后的单位
            'batch_size': batch_size,
            'total_batches': len(self.batch_times),
            'warmup_batches_skipped': skip_first,
        }

    def compare_modes(self):
        """
        对比不同模式的显存和速度

        Returns:
            对比报告字典
        """
        report = {
            'baseline': self.baseline_stats,
            'with_degradation': self.degradation_stats,
            'with_clear_branch': self.clear_branch_stats,
            'with_distillation': self.distillation_stats,
        }

        # 计算增量
        if self.baseline_stats:
            base_mem = self.baseline_stats.get('peak_allocated_gb', 0)
            base_time = self.baseline_stats.get('time_seconds', 0)

            for mode_name, stats in report.items():
                if stats and mode_name != 'baseline':
                    stats['memory_overhead_gb'] = stats.get('peak_allocated_gb', 0) - base_mem
                    stats['time_overhead_seconds'] = stats.get('time_seconds', 0) - base_time
                    stats['memory_overhead_percent'] = (stats['memory_overhead_gb'] / base_mem * 100) if base_mem > 0 else 0
                    stats['time_overhead_percent'] = (stats['time_overhead_seconds'] / base_time * 100) if base_time > 0 else 0

        return report

    def save_stats(self, save_path: str):
        """保存统计信息到文件"""
        import json

        report = self.compare_modes()
        report['average_batch_stats'] = self.get_average_stats()

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Stats saved to {save_path}")


def measure_inference_speed(
    model,
    input_tensor,
    warmup: int = 10,
    repeat: int = 100,
    device: Optional[torch.device] = None
):
    """
    测量推理速度（FPS）

    Args:
        model: PyTorch模型
        input_tensor: 输入张量
        warmup: warmup次数
        repeat: 重复次数
        device: 设备

    Returns:
        dict: 包含平均时间(ms)、FPS、配置信息

    GPT第五轮修正：
    - 恢复模型原训练状态
    - 仅CUDA设备调用同步
    - 记录配置信息
    """
    if device is None:
        device = next(model.parameters()).device

    # 保存原训练状态
    original_training = model.training

    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # 仅CUDA设备同步
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    end_time = time.time()

    avg_time_ms = (end_time - start_time) / repeat * 1000
    fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

    # 恢复原训练状态
    model.train(original_training)

    return {
        'avg_time_ms': avg_time_ms,
        'fps': fps,
        'warmup': warmup,
        'repeat': repeat,
        'device': str(device),
        'input_shape': list(input_tensor.shape),
    }


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    profiler = MemoryProfiler(device)

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 测试基本功能
    with profiler.profile("test_allocation"):
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y

    stats = profiler.get_stats()
    print(f"Test stats: {stats}")
