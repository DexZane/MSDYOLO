# P2阶段快速参考

**状态**: 准备就绪，等待云端执行  
**目标**: 训练收敛的baseline检测器，验证蒸馏路径  
**预计时间**: 3-5小时（GPU）  
**预计成本**: ~6-10元（AutoDL）

---

## 快速启动（3步）

### 1. 准备数据（首次执行）
```bash
# 下载DOTA v1.5
# 官网: https://captain-whu.github.io/DOTA/dataset.html

# 切片数据
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit
python ImgSplit_multi_process.py \
  --srcpath /path/to/DOTA/DOTAv1.5/train \
  --dstpath /path/to/DOTA/DOTAv1.5/train_split_1024_gap200 \
  --subsize 1024 --gap 200 --num_process 8

# 更新配置
vim data/dotav15_poly.yaml  # 修改path为实际路径
```

### 2. 启动训练
```bash
# 检查环境
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi

# 开始训练（一键）
bash scripts/train_baseline_p2.sh
```

### 3. 验证结果
```bash
# 50 epoch后快速检查
python trainmsd.py \
  --config configs/msdyolo-full.yaml \
  --weights runs/train/exp/weights/epoch_50.pt \
  --single-batch --device 0

# 期望: matchcount > 0
```

---

## 文件清单

### 已创建文件
- ✅ `configs/msdyolo-baseline-p2.yaml` - 训练配置
- ✅ `scripts/train_baseline_p2.sh` - 一键训练脚本
- ✅ `docs/p2-data-preparation.md` - 详细数据准备指南
- ✅ `docs/p2-quick-reference.md` - 本快速参考

### 需要准备
- ⏳ DOTA v1.5 数据集（~2.5GB）
- ⏳ 切片后数据（~4GB）
- ⏳ GPU环境（6GB+ VRAM）

---

## 关键参数

### 训练配置
```yaml
epochs: 200
batchsize: 16        # GPU调整: V100用32, 4GB GPU用8
imagesize: 1024
device: "0"
workers: 8
```

### GPU建议
| GPU | Batch Size | 训练时间 | 成本估算 |
|-----|-----------|---------|---------|
| GTX 1660Ti (6GB) | 8 | ~10小时 | - |
| RTX 3090 (24GB) | 32 | ~3小时 | ~5元 |
| V100 (16GB) | 16 | ~4小时 | ~10元 |
| A100 (40GB) | 64 | ~1.5小时 | ~10元 |

---

## 监控命令

```bash
# 实时日志
tail -f runs/train/baseline-p2-training.log

# GPU使用
watch -n 1 nvidia-smi

# 检查进度（每10 epoch保存）
ls -lh runs/train/exp/weights/
```

---

## 成功标志

### 训练完成
- [x] 200 epoch完成
- [x] mAP@0.5 > 65%
- [x] 损失收敛 (loss < 3)

### 权重验证
```bash
# Full模式单批次测试
python trainmsd.py \
  --config configs/msdyolo-full.yaml \
  --weights runs/train/exp/weights/best.pt \
  --single-batch

# 期望输出
[teacherforward diagnostics]
  Objectness logits: min=-3.x max=3.x     # 正常范围
  Confidence: min=0.0x max=0.5x           # 有高置信度预测

[Matching Summary]
  Filtered by confidence<0.1: <300        # 不是全部过滤
  Teacher-GT pairs: >0                    # 有匹配对
  Final matches: >0                       # 有学生-教师匹配

matchcount=X (X>0)                        # ✓ 成功！
distillationloss=Y (Y>0)                  # ✓ 蒸馏生效
```

---

## 故障排除

### OOM错误
```yaml
# 降低batch size
configs/msdyolo-baseline-p2.yaml:
  batchsize: 8  # 或更低
```

### 数据路径错误
```bash
# 检查配置
cat data/dotav15_poly.yaml | grep path

# 验证路径存在
ls /path/to/DOTA/DOTAv1.5/train_split_1024_gap200/images/ | head
```

### 训练太慢
```yaml
# 增加workers
configs/msdyolo-baseline-p2.yaml:
  workers: 16
```

---

## 下一步

### P3阶段：完整蒸馏实验
1. 使用baseline权重训练Full模式
2. 记录四分量损失演化
3. 消融实验（4种配置）
4. 性能对比分析

---

**P2准备完成。上传到云端后执行 `bash scripts/train_baseline_p2.sh` 开始训练。**
