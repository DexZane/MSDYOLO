"""显存、时间和吞吐量测量工具。"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class MemoryProfiler:
    """测量一个命名代码段的 CUDA 峰值显存和耗时。"""

    def __init__(self, device):
        self.device = torch.device(device)
        self.statistics = {}

    def reset(self):
        """清空统计并重置 CUDA 峰值计数。"""
        self.statistics = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def profile(self, name):
        """记录命名代码段。"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            peakallocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peakreserved = torch.cuda.max_memory_reserved(self.device) / 1024**3
        else:
            allocated = reserved = peakallocated = peakreserved = 0.0
        self.statistics[name] = {
            "elapsedseconds": time.perf_counter() - started,
            "allocatedgb": allocated,
            "reservedgb": reserved,
            "peakallocatedgb": peakallocated,
            "peakreservedgb": peakreserved,
        }

    def getstats(self, name=None):
        """读取全部统计或单个命名统计。"""
        if name is None:
            return self.statistics
        return self.statistics.get(name, {})

    def logcurrentmemory(self, tag):
        """即时记录当前 CUDA 显存。"""
        if self.device.type != "cuda":
            return {"tag": tag, "allocatedgb": 0.0, "reservedgb": 0.0}
        return {
            "tag": tag,
            "allocatedgb": torch.cuda.memory_allocated(self.device) / 1024**3,
            "reservedgb": torch.cuda.memory_reserved(self.device) / 1024**3,
        }


class TrainingProfiler:
    """累积训练 batch 的时间和峰值显存。"""

    def __init__(self, device):
        self.device = torch.device(device)
        self.records = []

    def profilebatch(self, batchindex, batchfunction):
        """执行并记录一个 batch。"""
        profiler = MemoryProfiler(self.device)
        with profiler.profile("batch"):
            result = batchfunction()
        record = {"batchindex": batchindex, **profiler.getstats("batch")}
        self.records.append(record)
        return result

    def getaveragestats(self, skipfirst=1, batchsize=1):
        """计算预热后的平均耗时、吞吐量和峰值显存。"""
        records = self.records[skipfirst:]
        if not records:
            return {}
        elapsed = sum(record["elapsedseconds"] for record in records) / len(records)
        peak = max(record["peakallocatedgb"] for record in records)
        return {
            "elapsedseconds": elapsed,
            "imagespersecond": batchsize / elapsed if elapsed else 0.0,
            "peakallocatedgb": peak,
        }

    def savestats(self, savepath):
        """把统计结果写入 JSON。"""
        path = Path(savepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


def measureinferencespeed(model, inputtensor, warmup=10, repeat=100, device=None):
    """测量单模型推理平均耗时。"""
    device = torch.device(device) if device is not None else inputtensor.device
    model.eval()
    with torch.no_grad():
        for warmupindex in range(warmup):
            model(inputtensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for repeatindex in range(repeat):
            model(inputtensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / repeat
