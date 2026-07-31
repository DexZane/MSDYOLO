# P2阶段数据准备指南

**目标**：准备完整DOTA v1.5数据集用于baseline检测器训练

---

## 一、DOTA数据集概述

### 1.1 数据集信息

**DOTA (Dataset for Object deTection in Aerial images)**
- 官网：https://captain-whu.github.io/DOTA/dataset.html
- 版本：v1.5（推荐）
- 图像数量：
  - 训练集：1411张大图 → 切片后~15000张
  - 验证集：458张大图 → 切片后~5000张
  - 测试集：937张大图（无标注）
- 图像尺寸：800×800 到 4000×4000 像素
- 类别：16类（飞机、船舶、储罐、桥梁等）
- 标注格式：旋转框（Oriented Bounding Box）

### 1.2 数据集大小

- 原始图像：~2.5GB
- 标注文件：~50MB
- 切片后数据：~3-4GB
- **总需求空间：~10GB**（包括中间文件）

---

## 二、数据下载

### 2.1 官方下载

访问官网注册并下载：
```
https://captain-whu.github.io/DOTA/dataset.html
```

需要下载的文件：
- `train.zip` - 训练集图像
- `val.zip` - 验证集图像  
- `trainval_labelTxt.zip` - 训练和验证标注

### 2.2 百度网盘（中国用户）

官网提供百度网盘链接（速度更快）

### 2.3 Google Drive

```bash
# 使用gdown下载（需要安装：pip install gdown）
# 链接见官网
```

---

## 三、数据预处理

### 3.1 目录结构

下载后解压到以下结构：
```
/path/to/DOTA/DOTAv1.5/
├── train/
│   ├── images/          # 原始大图
│   └── labelTxt/        # 标注文件
├── val/
│   ├── images/
│   └── labelTxt/
└── test/
    └── images/
```

### 3.2 图像切片

DOTA图像太大（最大4000×4000），需要切成小块：

**推荐参数**：
- 切片尺寸：1024×1024
- 重叠（gap）：200像素
- 目的：避免目标跨边界被截断

**使用官方工具**：
```bash
# 克隆DOTA_devkit
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit

# 安装依赖
pip install shapely
pip install Pillow
pip install tqdm

# 切片训练集
python ImgSplit_multi_process.py \
  --srcpath /path/to/DOTA/DOTAv1.5/train \
  --dstpath /path/to/DOTA/DOTAv1.5/train_split_1024_gap200 \
  --subsize 1024 \
  --gap 200 \
  --num_process 8

# 切片验证集
python ImgSplit_multi_process.py \
  --srcpath /path/to/DOTA/DOTAv1.5/val \
  --dstpath /path/to/DOTA/DOTAv1.5/val_split_1024_gap200 \
  --subsize 1024 \
  --gap 200 \
  --num_process 8
```

**切片后目录结构**：
```
/path/to/DOTA/DOTAv1.5/
├── train_split_1024_gap200/
│   ├── images/          # 切片后的1024×1024图像
│   └── labelTxt/        # 对应的标注
└── val_split_1024_gap200/
    ├── images/
    └── labelTxt/
```

### 3.3 验证切片结果

```bash
# 统计切片数量
echo "Training images:"
ls /path/to/DOTA/DOTAv1.5/train_split_1024_gap200/images/*.png | wc -l

echo "Validation images:"
ls /path/to/DOTA/DOTAv1.5/val_split_1024_gap200/images/*.png | wc -l

# 预期：
# 训练集：~15000张
# 验证集：~5000张
```

---

## 四、配置文件更新

### 4.1 更新数据集路径

编辑 `data/dotav15_poly.yaml`：

```yaml
# 修改path为实际路径
path: /path/to/DOTA/DOTAv1.5  # 改为你的实际路径

train: train_split_1024_gap200/images
val: val_split_1024_gap200/images

nc: 16
names: ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
        'basketball-court', 'storage-tank', 'soccer-ball-field', 
        'roundabout', 'harbor', 'swimming-pool', 'helicopter', 
        'container-crane']
```

### 4.2 验证配置

```bash
python -c "
import yaml
with open('data/dotav15_poly.yaml') as f:
    data = yaml.safe_load(f)
    print('Dataset path:', data['path'])
    print('Train path:', data['train'])
    print('Val path:', data['val'])
    print('Classes:', data['nc'])
"
```

---

## 五、云端环境准备

### 5.1 推荐云平台

**选项A：AutoDL（推荐，国内速度快）**
- 网址：https://www.autodl.com/
- GPU：RTX 3090 / V100 / A100
- 费用：~1-3元/小时
- DOTA数据集可能已内置

**选项B：Google Colab Pro**
- 网址：https://colab.research.google.com/
- GPU：T4 / A100
- 费用：$10/月

**选项C：AWS / Azure / GCP**
- GPU实例：p3.2xlarge (V100)
- 费用：~$3/小时
- 需要手动上传数据

### 5.2 GPU要求

**最低配置**：
- GPU：GTX 1660 Ti (6GB VRAM)
- Batch size：8
- 训练时间：~8-12小时

**推荐配置**：
- GPU：RTX 3090 / V100 (16GB+ VRAM)
- Batch size：16-32
- 训练时间：~3-5小时

**高性能配置**：
- GPU：A100 (40GB VRAM)
- Batch size：32-64
- 训练时间：~1-2小时

### 5.3 软件环境

```bash
# Python 3.8+
python --version

# PyTorch 1.10+ with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖（从requirements.txt）
pip install -r requirements.txt

# 验证CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 六、快速启动检查清单

### 训练前检查

- [ ] DOTA v1.5数据集已下载
- [ ] 图像已切片（1024×1024，gap 200）
- [ ] `data/dotav15_poly.yaml`路径已更新
- [ ] YOLOv5s预训练权重已下载（`yolov5s.pt`）
- [ ] GPU可用且CUDA正常
- [ ] 配置文件 `configs/msdyolo-baseline-p2.yaml` 存在
- [ ] 训练脚本 `scripts/train_baseline_p2.sh` 可执行

### 启动命令

```bash
# 方式1：使用脚本（推荐）
bash scripts/train_baseline_p2.sh

# 方式2：直接命令
python trainmsd.py \
  --config configs/msdyolo-baseline-p2.yaml \
  --device 0
```

---

## 七、训练监控

### 7.1 实时监控

```bash
# 查看训练日志
tail -f runs/train/baseline-p2-training.log

# 监控GPU使用
watch -n 1 nvidia-smi
```

### 7.2 预期结果

**收敛指标**（200 epoch后）：
- mAP@0.5：~65-70%（DOTA v1.5 baseline）
- 训练损失：~2-3
- 检测损失（box）：~0.02-0.03
- 分类损失（cls）：~0.01-0.02
- Objectness损失（obj）：~0.01-0.02

**置信度分布**（关键验证）：
- Objectness logits：-3 to +3（正常范围）
- Max confidence on clean images：>0.25
- 预期匹配率：>10%

### 7.3 中间检查

**50 epoch后快速验证**：
```bash
# 使用50 epoch权重测试Full模式
python trainmsd.py \
  --config configs/msdyolo-full.yaml \
  --weights runs/train/exp/weights/epoch_50.pt \
  --single-batch \
  --device 0

# 查看matchcount（应该>0）
```

---

## 八、故障排除

### 8.1 常见问题

**问题1：OOM（Out of Memory）**
```
解决：降低batch size
configs/msdyolo-baseline-p2.yaml:
  batchsize: 8  # 从16降到8
```

**问题2：数据集路径错误**
```
错误：FileNotFoundError: train_split_1024_gap200/images
解决：检查data/dotav15_poly.yaml中的path设置
```

**问题3：标注格式错误**
```
错误：Invalid DOTA format
解决：确认使用DOTA_devkit切片，不是手动切的
```

### 8.2 性能优化

```bash
# 增加数据加载workers（如果CPU充足）
configs/msdyolo-baseline-p2.yaml:
  workers: 16  # 从8增加到16

# 启用混合精度训练（节省显存）
# 在trainmsd.py中添加torch.cuda.amp
```

---

## 九、预期时间和成本

### 9.1 时间估算

| 步骤 | 时间 |
|------|------|
| 数据下载 | 30分钟 - 2小时 |
| 数据切片 | 10-30分钟 |
| 环境配置 | 10-20分钟 |
| 训练200 epoch | 1-12小时（取决于GPU）|
| **总计** | **2-15小时** |

### 9.2 成本估算

**AutoDL（推荐）**：
- RTX 3090：1.5元/小时 × 4小时 = 6元
- V100：2.5元/小时 × 3小时 = 7.5元
- A100：5元/小时 × 2小时 = 10元

**Google Colab Pro**：
- 包月$10，无限训练

---

## 十、下一步

训练完成后：

1. **验证权重质量**：
   ```bash
   python trainmsd.py \
     --config configs/msdyolo-full.yaml \
     --weights runs/train/exp/weights/best.pt \
     --single-batch
   ```

2. **检查关键指标**：
   - matchcount > 0 ✓
   - distillationloss > 0 ✓
   - 四分量损失非零 ✓

3. **进入P3阶段**：完整蒸馏训练和消融实验

---

**数据准备文档完成。准备好后运行 `scripts/train_baseline_p2.sh` 开始训练。**
