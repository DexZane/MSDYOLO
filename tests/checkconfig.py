#!/usr/bin/env python
"""配置验证单元测试。"""

import pytest
from utils.config import MSDYOLOConfig


class CheckConfigValidation:
    """配置验证规则测试。"""

    def checkphase0rejectsdistillation(self):
        """Phase 0 必须拒绝蒸馏。"""
        config = MSDYOLOConfig()
        config.set("experiment.phase", 0)
        config.set("distillation.enabled", True)
        config.set("degradation.enabled", True)
        config.set("clearbranch.enabled", True)

        assert not config.validate(), "Phase 0 should reject distillation"

    def checkphase1rejectsdistillation(self):
        """Phase 1 必须拒绝蒸馏。"""
        config = MSDYOLOConfig()
        config.set("experiment.phase", 1)
        config.set("distillation.enabled", True)
        config.set("degradation.enabled", True)
        config.set("clearbranch.enabled", True)

        assert not config.validate(), "Phase 1 should reject distillation"

    def checkphase2allowsdistillation(self):
        """Phase 2 允许蒸馏。"""
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.set("distillation.enabled", True)
        config.set("degradation.enabled", True)
        config.set("clearbranch.enabled", True)

        assert config.validate(), "Phase 2 should allow distillation"

    def checkdistillationrequiresdegradation(self):
        """蒸馏必须启用退化。"""
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.set("distillation.enabled", True)
        config.set("degradation.enabled", False)
        config.set("clearbranch.enabled", True)

        assert not config.validate(), "Distillation requires degradation"

    def checkdistillationrequiresclearbranch(self):
        """蒸馏必须启用清晰分支。"""
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.set("distillation.enabled", True)
        config.set("degradation.enabled", True)
        config.set("clearbranch.enabled", False)

        assert not config.validate(), "Distillation requires clearbranch"
