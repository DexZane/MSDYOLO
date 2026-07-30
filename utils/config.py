"""
MSDYOLO Configuration System
GPT批准：Phase 1 低风险基础设施

消融实验配置框架，支持：
- Baseline模式（所有新功能关闭）
- 退化管线配置
- 清晰分支配置
- 蒸馏损失配置（预留，Phase 1不实现损失）
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MSDYOLOConfig:
    """
    MSDYOLO配置管理

    设计原则：
    - 默认关闭所有新功能（与baseline严格一致）
    - 每个功能可独立开关
    - 所有参数可序列化
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 配置文件路径（YAML格式）
        """
        # 默认配置（baseline模式）
        self.config = self._get_default_config()

        # 从文件加载配置
        if config_path is not None:
            self.load_from_file(config_path)

    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置（baseline模式）

        GPT要求：默认关闭时必须与原始baseline严格一致
        """
        return {
            # ===== 实验元信息 =====
            'experiment': {
                'name': 'baseline',
                'description': 'Original YOLOv5-OBB baseline',
                'phase': 1,  # Phase 1: 基础设施，Phase 2: 实验验证
            },

            # ===== Baseline训练配置 =====
            'training': {
                'data': 'data/dotav15_poly.yaml',
                'cfg': 'models/yolov5s.yaml',
                'weights': '',  # 预训练权重路径
                'epochs': 300,
                'batch_size': 16,
                'img_size': 1024,
                'device': '0',  # GPU ID
                'workers': 8,
                'hyp': 'data/hyps/obb/hyp.finetune_dota.yaml',  # 修正：使用实际存在的路径
            },

            # ===== 退化管线配置 =====
            'degradation': {
                'enabled': False,  # 默认关闭
                'psf': {
                    'enabled': False,
                    'kernel_size': 5,
                    'sigma': 1.0,
                },
                'downsample': {
                    'enabled': False,
                    'scale': 2.0,  # 降采样倍率
                },
                'noise': {
                    'enabled': False,
                    'type': 'gaussian',  # 'gaussian' or 'salt_pepper'
                    'level': 0.01,
                },
                'upsample_mode': 'bilinear',
                'seed': 42,  # 随机种子，用于可复现
            },

            # ===== 清晰分支配置 =====
            'clear_branch': {
                'enabled': False,  # 默认关闭
                'strategy': 'eval_mode',  # 'eval_mode' or 'freeze_bn'
                'extract_sparse': True,  # 是否提取稀疏预测
                'top_k': 300,  # 保留Top-K预测
                'conf_threshold': 0.25,  # 置信度阈值
            },

            # ===== 知识蒸馏配置（预留，Phase 1不实现） =====
            'distillation': {
                'enabled': False,  # Phase 1必须关闭
                'alpha': 0.1,  # 蒸馏损失权重（预留）
                'match_iou_threshold': 0.5,  # 预测匹配IoU阈值
                'loss_type': 'l1',  # 'l1', 'l2', 'kl'（预留）
                # 以下参数Phase 1不使用，等待DPKD/ADMD/DMR分析后确定
                'angle_weight': None,  # 角度分量权重（待定）
                'detectability_weight': None,  # 可检测性权重（待定）
            },

            # ===== 显存和速度profiling =====
            'profiling': {
                'enabled': False,  # GPT第五轮修正：默认关闭
                'log_interval': 100,  # 每N个batch记录一次
                'save_stats': True,  # 是否保存统计信息
                'output_dir': 'runs/profiling',
            },

            # ===== 消融实验模式 =====
            'ablation_mode': 'baseline',  # 'baseline', 'with_degradation', 'with_clear_branch', 'full'
        }

    def load_from_file(self, config_path: str):
        """从YAML文件加载配置"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        # 递归更新配置
        self._recursive_update(self.config, user_config)

        logger.info(f"Loaded config from {config_path}")

    def _recursive_update(self, base: Dict, update: Dict):
        """递归更新嵌套字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._recursive_update(base[key], value)
            else:
                base[key] = value

    def save_to_file(self, output_path: str):
        """保存配置到YAML文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.safe_dump(self.config, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Saved config to {output_path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点分隔路径）

        Example:
            config.get('degradation.psf.sigma')  # 返回 1.0
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        设置配置值（支持点分隔路径）

        Example:
            config.set('degradation.psf.enabled', True)
        """
        keys = key_path.split('.')
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def apply_ablation_mode(self, mode: str):
        """
        应用预定义的消融模式

        Args:
            mode: 'baseline', 'with_degradation', 'with_clear_branch', 'full'
        """
        self.config['ablation_mode'] = mode

        if mode == 'baseline':
            # 所有新功能关闭
            self.set('degradation.enabled', False)
            self.set('clear_branch.enabled', False)
            self.set('distillation.enabled', False)

        elif mode == 'with_degradation':
            # 仅启用退化管线
            self.set('degradation.enabled', True)
            self.set('clear_branch.enabled', False)
            self.set('distillation.enabled', False)

        elif mode == 'with_clear_branch':
            # 启用退化管线和清晰分支
            self.set('degradation.enabled', True)
            self.set('clear_branch.enabled', True)
            self.set('distillation.enabled', False)

        elif mode == 'full':
            # 启用所有功能（Phase 1不允许，distillation必须关闭）
            self.set('degradation.enabled', True)
            self.set('clear_branch.enabled', True)
            # self.set('distillation.enabled', True)  # Phase 1禁止
            logger.warning("'full' mode in Phase 1: distillation remains disabled")

        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

        logger.info(f"Applied ablation mode: {mode}")

    def validate(self) -> bool:
        """
        验证配置合法性

        GPT要求检查：
        - Phase 1不能启用distillation
        - 清晰分支依赖退化管线
        """
        errors = []

        # 检查1: Phase 1不能启用蒸馏
        if self.get('experiment.phase') == 1 and self.get('distillation.enabled'):
            errors.append("Phase 1 不允许启用distillation")

        # 检查2: 清晰分支需要退化管线
        if self.get('clear_branch.enabled') and not self.get('degradation.enabled'):
            errors.append("清晰分支需要启用退化管线")

        # 检查3: 降采样scale必须>1
        if self.get('degradation.downsample.enabled'):
            scale = self.get('degradation.downsample.scale')
            if scale <= 1.0:
                errors.append(f"降采样scale必须>1，当前: {scale}")

        if errors:
            for error in errors:
                logger.error(f"配置验证失败: {error}")
            return False

        logger.info("配置验证通过 ✓")
        return True

    def __repr__(self):
        mode = self.get('ablation_mode')
        phase = self.get('experiment.phase')
        return f"MSDYOLOConfig(phase={phase}, mode={mode})"


def create_example_configs():
    """创建示例配置文件"""
    output_dir = Path('configs')
    output_dir.mkdir(exist_ok=True)

    # 1. Baseline配置
    baseline_config = MSDYOLOConfig()
    baseline_config.save_to_file('configs/baseline.yaml')

    # 2. 退化管线配置
    degradation_config = MSDYOLOConfig()
    degradation_config.apply_ablation_mode('with_degradation')
    degradation_config.set('degradation.psf.enabled', True)
    degradation_config.set('degradation.downsample.enabled', True)
    degradation_config.set('degradation.downsample.scale', 2.0)
    degradation_config.set('experiment.name', 'with_degradation')
    degradation_config.save_to_file('configs/with_degradation.yaml')

    # 3. 清晰分支配置
    clear_branch_config = MSDYOLOConfig()
    clear_branch_config.apply_ablation_mode('with_clear_branch')
    clear_branch_config.set('degradation.psf.enabled', True)
    clear_branch_config.set('degradation.downsample.enabled', True)
    clear_branch_config.set('experiment.name', 'with_clear_branch')
    clear_branch_config.save_to_file('configs/with_clear_branch.yaml')

    print("✓ Created example configs:")
    print("  - configs/baseline.yaml")
    print("  - configs/with_degradation.yaml")
    print("  - configs/with_clear_branch.yaml")


if __name__ == '__main__':
    # 测试配置系统
    print("Testing MSDYOLO Config System...\n")

    # 测试1: 默认配置
    config = MSDYOLOConfig()
    print(f"Default config: {config}")
    print(f"Ablation mode: {config.get('ablation_mode')}")
    print(f"Degradation enabled: {config.get('degradation.enabled')}")
    print(f"Validation: {config.validate()}\n")

    # 测试2: 应用消融模式
    config.apply_ablation_mode('with_degradation')
    print(f"After applying 'with_degradation':")
    print(f"  Degradation enabled: {config.get('degradation.enabled')}")
    print(f"  Clear branch enabled: {config.get('clear_branch.enabled')}\n")

    # 测试3: 验证失败场景
    config.set('clear_branch.enabled', True)
    config.set('degradation.enabled', False)
    print("Testing validation (should fail):")
    print(f"  Validation: {config.validate()}\n")

    # 测试4: 创建示例配置
    create_example_configs()
