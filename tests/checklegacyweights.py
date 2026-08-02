"""Regression tests for loading checkpoints serialized with YOLOv5 module names."""


class CheckLegacyWeights:

    def checkregisterlegacycheckpointmodulealiases(self):
        """Legacy ``models.*`` globals resolve to canonical ``msdyolo.*`` modules."""

        from msdyolo.compat import register_legacy_module_aliases

        register_legacy_module_aliases()

        import models.common as legacycommon
        import models.yolo as legacyyolo
        from msdyolo.models import common, yolo

        assert legacycommon.Conv is common.Conv
        assert legacyyolo.Model is yolo.Model
