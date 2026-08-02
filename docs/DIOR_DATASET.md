# DIOR 数据集使用指南

## 数据集信息

- **名称**: DIOR (Dataset for Object Detection in Aerial Images)
- **论文**: [Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges](https://arxiv.org/abs/1909.00133)
- **图像数量**: 23,463张
- **图像尺寸**: 800×800像素
- **类别数量**: 20类
- **标注格式**: Pascal VOC (XML)
- **边界框类型**: 水平框（Horizontal Bounding Box）

## 类别列表

1. airplane
2. airport
3. baseball-field
4. basketball-court
5. bridge
6. chimney
7. dam
8. expressway-service-area
9. expressway-toll-station
10. golf-course
11. ground-track-field
12. harbor
13. overpass
14. ship
15. stadium
16. storage-tank
17. tennis-court
18. train-station
19. vehicle
20. windmill

## 下载数据集

### 方法1：使用提供的下载脚本（推荐）

```bash
cd /Users/dexzane/Desktop/FindProject/MSDYOLO
python3 msdyolo/data/scripts/download_dior.py
```

默认下载到 `dataset/DIOR/` 目录。

### 方法2：指定下载目录

```bash
python3 msdyolo/data/scripts/download_dior.py /path/to/custom/directory
```

### 下载过程

脚本会自动完成：
1. 从Google Drive下载压缩包
2. 解压到目标目录
3. 验证数据集完整性
4. 清理临时文件

**注意**：
- 下载大小约 4-5 GB
- 需要安装 `gdown`：`pip install gdown`
- 支持断点续传（中断后重新运行脚本即可）

## 数据集目录结构

### 原始下载结构

```
dataset/DIOR/
├── Annotations/
│   ├── Horizontal Bounding Boxes/    # 水平框标注
│   │   ├── 00001.xml
│   │   └── ...
│   └── Oriented Bounding Boxes/      # 旋转框标注 ✓ MSDYOLO使用
│       ├── 00001.xml
│       └── ...
├── ImageSets/
│   └── Main/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
└── JPEGImages/
    ├── 00001.jpg
    └── ...
```

### 规范化后的结构（自动完成）

下载脚本会自动规范化目录结构，提取OBB标注：

```
dataset/DIOR/
├── Annotations/          # 旋转框标注（从OBB子目录提升）
│   ├── 00001.xml
│   ├── 00002.xml
│   └── ...
├── ImageSets/           # 数据集划分（从Main子目录提升）
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── JPEGImages/          # 原始图像
    ├── 00001.jpg
    ├── 00002.jpg
    └── ...
```

**注意**：
- 脚本默认使用 **Oriented Bounding Boxes (OBB)** 以匹配MSDYOLO的旋转框检测
- Horizontal Bounding Boxes 子目录会被自动删除
- 如需使用水平框，请手动修改脚本中的 `use_obb=False`

## 转换为YOLO格式

MSDYOLO需要将DIOR从VOC格式转换为YOLO格式：

```bash
# TODO: 转换脚本待实现
python3 msdyolo/data/scripts/convert_dior_to_yolo.py
```

转换后的目录结构：

```
dataset/DIOR/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 训练配置

使用DIOR数据集训练：

```bash
python3 train.py \
    --data msdyolo/data/dior.yaml \
    --cfg configs/models/yolov5s.yaml \
    --weights yolov5s.pt \
    --batch-size 16 \
    --epochs 150
```

## 与DOTA的区别

| 特性 | DOTA v1.5 | DIOR |
|------|-----------|------|
| 原始图像数量 | 2,806张 | 23,463张 |
| 切片后图像数量 | ~46,725张 | 23,463张（无需切片）|
| 图像尺寸 | 4000×4000 (需切片) | 800×800 (无需切片) |
| 切片大小 | 1024×1024 | N/A |
| 类别数量 | 16类 | 20类 |
| 边界框类型 | 旋转框(OBB) | 旋转框(OBB) ✓ |
| 标注格式 | txt (x1 y1 x2 y2 x3 y3 x4 y4) | XML (Pascal VOC) |
| 单轮训练时间 | 20分钟 | ~10分钟 (估算) |
| 200轮训练时间 | 38-41小时 | ~33小时 (估算) |
| 训练成本($2/小时) | ~$80 | ~$66 (节省约18%) |

**注意**：
- DIOR提供两种标注格式：Horizontal Bounding Boxes 和 Oriented Bounding Boxes
- MSDYOLO下载脚本默认使用OBB标注，与DOTA保持一致
- 训练时间基于DOTA切片后~46,725张图像计算

## 优势

1. **成本更低**：训练时间减少约60%
2. **图像尺寸适中**：无需切片，直接训练
3. **数据量充足**：23,463张图像，足够验证方法有效性
4. **标注规范**：Pascal VOC格式，工具链成熟

## 注意事项

⚠️ **重要**：DIOR使用水平框（Horizontal Bounding Box），与DOTA的旋转框（Oriented Bounding Box）不同。

如果你的MSDYOLO项目专门针对旋转框检测，需要：
1. 调整损失函数（从OBB损失改为HBB损失）
2. 调整后处理逻辑（NMS等）
3. 修改评估指标

或者，考虑使用DIOR的旋转框标注版本（如果可用）。

## 数据集引用

```bibtex
@article{li2019dior,
  title={Object Detection in Optical Remote Sensing Images: A Survey and A New Benchmark},
  author={Li, Ke and Wan, Gang and Cheng, Gong and Meng, Liqiu and Han, Junwei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={159},
  pages={296--307},
  year={2020},
  publisher={Elsevier}
}
```

## 参考链接

- 论文: https://arxiv.org/abs/1909.00133
- 官方网站: http://www.escience.cn/people/gongcheng/DIOR.html
- Google Drive下载: https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC
