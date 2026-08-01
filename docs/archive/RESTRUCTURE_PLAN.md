# MSDYOLO Project Restructuring Plan

## Current Issues
- Root directory cluttered with log files
- Configs scattered
- Scripts mixed with data
- No clear entry points

## Target Structure (YOLOv5-style)

```
MSDYOLO/
├── README.md
├── requirements.txt
├── setup.py                 # Package installation
│
├── msdyolo/                 # Main package
│   ├── __init__.py
│   ├── train.py            # Training entry (from trainmsd.py)
│   ├── detect.py
│   ├── export.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo.py
│   │   ├── common.py
│   │   └── experimental.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── general.py
│   │   ├── loss.py
│   │   ├── metrics.py
│   │   └── rboxs_utils.py
│   └── data/
│       ├── scripts/
│       │   ├── download_dota.py
│       │   └── split_dota.py  # from imgsplit.py
│       └── hyps/
│
├── configs/
│   ├── train/
│   │   ├── baseline.yaml
│   │   ├── degradation.yaml
│   │   └── distillation.yaml
│   └── models/
│       ├── yolov5s.yaml
│       └── yolov5m.yaml
│
├── data/
│   ├── dota.yaml
│   └── examples/
│
├── scripts/
│   ├── setup.sh            # Unified setup
│   ├── train.sh            # Training shortcuts
│   └── eval.sh
│
├── tests/
│   └── test_*.py
│
├── docs/
│   ├── INSTALL.md
│   ├── QUICKSTART.md
│   └── TRAINING.md
│
└── .gitignore              # Ignore logs/, runs/, dataset/
```

## Benefits
1. **Clean root** - Only README, requirements, setup.py
2. **Importable package** - `from msdyolo import train`
3. **Clear separation** - models/utils/data/configs
4. **Standard tools** - pip install -e .
5. **Professional** - Like ultralytics/yolov5

## Migration Steps
1. Create msdyolo/ package structure
2. Move and rename files
3. Update imports
4. Create setup.py
5. Clean root directory
6. Update documentation

Proceed with restructuring?
