"""训练 epoch 健康警告测试。"""

from msdyolo.train import training_health_message


class CheckTrainingHealth:
    """验证仅完整蒸馏训练报告零匹配。"""

    def checkbaselinezeromatchesremainssilent(self):
        """关闭蒸馏的 baseline 固定零匹配不应触发警告。"""
        assert training_health_message(False, 0, 10) is None

    def checkdistillationzeromatcheswithtargetsreportswarning(self):
        """蒸馏训练有目标却没有匹配时应报告可操作的警告。"""
        message = training_health_message(True, 0, 10)
        assert message is not None
        assert "targets=10" in message
        assert "match=0" in message

    def checkdistillationzeromatcheswithouttargetsremainssilent(self):
        """没有 epoch 目标时，零匹配不表示蒸馏健康问题。"""
        assert training_health_message(True, 0, 0) is None

    def checkdistillationzeromatcheswithnegativetargetsremainssilent(self):
        """无效的负目标计数不应触发仅适用于正目标的健康警告。"""
        assert training_health_message(True, 0, -1) is None

    def checkdistillationmatchesremainssilent(self):
        """只要 epoch 中存在匹配，就不应发出零匹配警告。"""
        assert training_health_message(True, 1, 10) is None
