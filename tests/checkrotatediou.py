"""精确旋转 IoU、NMS 和 Shapely 依赖测试。"""

import math

import pytest
import torch

import utils.rotatednms as rotatedmodule
from utils.rotatednms import classwiserotatednms, rotatediou, rotatednms


class CheckRotatedIoU:

    def checkidenticalboxesreturnone(self):
        box = (50, 50, 20, 10, 0.0)
        assert rotatediou(box, box) == pytest.approx(1.0)

    def checkidenticalrotatedboxesreturnone(self):
        box = (50, 50, 20, 10, math.pi / 4)
        assert rotatediou(box, box) == pytest.approx(1.0)

    def checkseparatedboxesreturnzero(self):
        assert rotatediou((10, 10, 5, 5, 0), (100, 100, 5, 5, 0)) == 0

    def checkiousymmetry(self):
        first = (50, 50, 20, 10, 0.3)
        second = (55, 52, 18, 12, -0.2)
        assert rotatediou(first, second) == pytest.approx(rotatediou(second, first))

    @pytest.mark.parametrize(
        "first,second",
        [
            ((50, 50, 20, 10, 0), (55, 52, 18, 12, 0)),
            ((50, 50, 20, 10, math.pi / 4), (55, 52, 18, 12, -math.pi / 4)),
            ((50, 50, 30, 5, 0), (55, 52, 5, 30, math.pi / 2)),
            ((10, 10, 100, 50, 0.5), (15, 12, 90, 45, -0.3)),
        ],
    )
    def checkiouisboundsafe(self, first, second):
        assert 0 <= rotatediou(first, second) <= 1

    def checkaxisalignedvalue(self):
        value = rotatediou((50, 50, 20, 10, 0), (55, 52, 18, 12, 0))
        expected = (14 * 9) / (20 * 10 + 18 * 12 - 14 * 9)
        assert value == pytest.approx(expected)

    def checkperpendicularboxespartiallyoverlap(self):
        value = rotatediou((50, 50, 40, 10, 0), (50, 50, 20, 30, math.pi / 2))
        assert 0 < value < 1


class CheckRotatedNMS:

    def checkupstreamfallbackinterfaceusesstrictimplementation(self):
        from utils.nms_rotated import obb_nms

        boxes = torch.tensor([[50, 50, 20, 10, 0.0], [51, 50, 20, 10, 0.0]])
        selectedboxes, kept = obb_nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
        assert kept.tolist() == [0]
        assert torch.equal(selectedboxes, boxes[kept])

    def checkupstreamfallbackfilterstinyboxes(self):
        from utils.nms_rotated import obb_nms

        boxes = torch.tensor([[10, 10, 0.0001, 5, 0.0], [30, 30, 5, 5, 0.0]])
        selectedboxes, kept = obb_nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
        assert kept.tolist() == [1]
        assert torch.equal(selectedboxes, boxes[kept])

    def checkduplicatedboxesaresuppressed(self):
        boxes = torch.tensor(
            [[50, 50, 20, 10, 0.0], [51, 50, 20, 10, 0.0], [50, 51, 20, 10, 0.0]]
        )
        kept = rotatednms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5)
        assert kept.tolist() == [0]

    def checkseparatedboxesarekept(self):
        boxes = torch.tensor([[10, 10, 5, 5, 0], [100, 100, 5, 5, 0], [200, 200, 5, 5, 0]])
        kept = rotatednms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5)
        assert kept.tolist() == [0, 1, 2]

    def checkhighestscorewins(self):
        boxes = torch.tensor([[50, 50, 20, 10, 0], [51, 50, 20, 10, 0]])
        kept = rotatednms(boxes, torch.tensor([0.9, 0.95]), 0.5)
        assert kept.tolist() == [1]

    def checkclassesdonotcrosssuppress(self):
        boxes = torch.tensor([[50, 50, 20, 10, 0], [51, 50, 20, 10, 0]])
        kept = classwiserotatednms(
            boxes,
            torch.tensor([0.9, 0.8]),
            torch.tensor([0, 1]),
            0.5,
        )
        assert sorted(kept.tolist()) == [0, 1]

    def checkemptyinputstaysempty(self):
        kept = rotatednms(torch.zeros(0, 5), torch.zeros(0), 0.5)
        assert kept.numel() == 0

    def checksingleboxiskept(self):
        kept = rotatednms(torch.tensor([[50, 50, 20, 10, 0.0]]), torch.tensor([0.9]))
        assert kept.tolist() == [0]


class CheckIoUEdges:

    def checkzeroareaboxreturnszero(self):
        assert rotatediou((50, 50, 0, 0, 0), (50, 50, 20, 10, 0)) == 0

    def checkverythinidenticalboxreturnsone(self):
        box = (50, 50, 100, 0.1, 0)
        assert rotatediou(box, box) == pytest.approx(1.0)

    def checklargeangledifferenceisboundsafe(self):
        value = rotatediou((50, 50, 30, 10, 0), (50, 50, 30, 10, math.pi / 2 - 0.01))
        assert 0 <= value <= 1

    def checkmissingshapelyfailsclearly(self, monkeypatch):
        monkeypatch.setattr(rotatedmodule, "SHAPELYAVAILABLE", False)
        with pytest.raises(RuntimeError, match="Shapely>=2.0"):
            rotatedmodule.rotatediou((0, 0, 1, 1, 0), (0, 0, 1, 1, 0))
