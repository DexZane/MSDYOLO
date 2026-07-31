"""P0-A.1 独立蒸馏组件的行为测试。"""

import math

import pytest
import torch
import torch.nn as nn

from utils.clearbranch import teacherforward
from utils.decoder import SparsePredictions, decodesparse
from utils.distillation import computefourcomponentloss
from utils.matching import DistillationMatch, matchpredictions
from utils.routing import computerouting


class TinyHead:
    """提供与 YOLO Detect 一致的 anchor 和 stride 元数据。"""

    def __init__(self):
        self.stride = torch.tensor([8.0])
        self.anchors = torch.tensor([[[2.0, 3.0]]])


class TinyModel(nn.Module):
    """产生可控 raw output，并模拟 YOLO eval tuple。"""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bn = nn.BatchNorm2d(3)
        self.model = [TinyHead()]

    def forward(self, images):
        images = self.bn(images)
        batch = images.shape[0]
        raw = torch.zeros(batch, 1, 1, 2, 201, device=images.device)
        raw = raw + self.scale
        raw[..., 4] = torch.tensor([2.0, 1.0], device=images.device)
        raw[..., 5] = 3.0
        raw[..., 21 + 90] = 4.0
        rawoutputs = [raw]
        if self.training:
            return rawoutputs
        decoded = expecteddecode(rawoutputs, self.model[-1])
        return decoded, rawoutputs


def expecteddecode(rawoutputs, head):
    """独立计算 YOLO Detect 的 eval 坐标结果。"""
    decoded = []
    for scaleindex, raw in enumerate(rawoutputs):
        batch, anchors, height, width, outputs = raw.shape
        ygrid, xgrid = torch.meshgrid(
            torch.arange(height, device=raw.device),
            torch.arange(width, device=raw.device),
            indexing="ij",
        )
        grid = torch.stack((xgrid, ygrid), -1).view(1, 1, height, width, 2)
        anchorgrid = (
            head.anchors[scaleindex]
            .view(1, anchors, 1, 1, 2)
            .to(raw.device)
            * head.stride[scaleindex].to(raw.device)
        )
        activated = raw.sigmoid()
        xy = (activated[..., :2] * 2 - 0.5 + grid) * head.stride[scaleindex]
        wh = (activated[..., 2:4] * 2) ** 2 * anchorgrid
        decoded.append(torch.cat((xy, wh, activated[..., 4:]), -1).view(batch, -1, outputs))
    return torch.cat(decoded, 1)


def makesparse(values):
    """建立匹配和损失测试所需的稀疏预测。"""
    batch, candidates, outputs = values.shape
    assert outputs == 201
    shape = (batch, candidates)
    zeros = torch.zeros(shape, dtype=torch.long, device=values.device)
    return SparsePredictions(
        values=values,
        scaleindex=zeros,
        anchorindex=zeros,
        gridx=zeros,
        gridy=zeros,
        rawindex=torch.arange(candidates, device=values.device).expand(batch, -1),
    )


def makeprediction(boxes, classes, angles, requiresgrad=False):
    """由旋转框、类别和角度构造 201 维候选。"""
    values = torch.full((1, len(boxes), 201), -5.0)
    values[..., :4] = torch.tensor(boxes)
    values[..., 4] = 5.0
    for index, classid in enumerate(classes):
        values[0, index, 5 + classid] = 5.0
        angleindex = int(round(angles[index] * 180.0 / math.pi + 90.0)) % 180
        values[0, index, 21 + angleindex] = 5.0
    return makesparse(values.requires_grad_(requiresgrad))


def maketargets():
    """构造两个 DOTA 187 维像素坐标目标。"""
    targets = torch.zeros(2, 187)
    targets[0, :7] = torch.tensor([0, 2, 20, 20, 16, 8, 0.0])
    targets[1, :7] = torch.tensor([0, 4, 60, 30, 20, 10, math.pi / 6])
    targets[0, 7 + 90] = 1.0
    targets[1, 7 + 120] = 1.0
    return targets


class CheckDecoder:

    def checkdecodedxywhmatchesrealyoloevaloutput(self):
        from models.yolo import Model

        model = Model("models/yolov5n.yaml", nc=16).eval()
        images = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            decoded, rawoutputs = model(images)
        sparse = decodesparse(rawoutputs, model, topk=20)
        expected = decoded[0, sparse.rawindex[0], :4]
        assert torch.allclose(sparse.values[0, :, :4], expected, atol=1e-5)

    def checknoninplacehalfdecodermatchesrealyolo(self):
        from models.yolo import Model

        model = Model("models/yolov5n.yaml", nc=16).eval().half()
        model.model[-1].inplace = False
        images = torch.zeros(1, 3, 64, 64).half()
        with torch.no_grad():
            decoded, rawoutputs = model(images)
        sparse = decodesparse(rawoutputs, model, topk=20)
        expected = decoded[0, sparse.rawindex[0], :4]
        assert sparse.values.dtype == decoded.dtype
        assert torch.allclose(sparse.values[0, :, :4], expected, atol=1e-5)

    def checkevaltupleisunpackedandteacherisdetached(self):
        model = TinyModel().train()
        images = torch.randn(2, 3, 8, 8)
        initialmean = model.bn.running_mean.clone()

        sparse = teacherforward(model, images, topk=2)

        assert not sparse.values.requires_grad
        assert all(not tensor.requires_grad for tensor in sparse.tensors())
        assert torch.equal(initialmean, model.bn.running_mean)
        assert model.training

    def checkdecodedxywhmatchesmodelevaloutput(self):
        model = TinyModel().eval()
        images = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            decoded, rawoutputs = model(images)

        sparse = decodesparse(rawoutputs, model, topk=2)
        expected = decoded[0, sparse.rawindex[0], :4]

        assert torch.allclose(sparse.values[0, :, :4], expected, atol=1e-6)

    def checkstudentdecoderpreservesgradient(self):
        model = TinyModel().train()
        rawoutputs = model(torch.randn(1, 3, 8, 8))
        sparse = decodesparse(rawoutputs, model, topk=2)

        sparse.values.sum().backward()

        assert model.scale.grad is not None
        assert model.scale.grad.abs().item() > 0


class CheckMatching:

    def checkdifferentcandidateordersmatchthesamegroundtruth(self):
        targets = maketargets()
        teacher = makeprediction(
            [[60, 30, 20, 10], [20, 20, 16, 8]],
            [4, 2],
            [math.pi / 6, 0.0],
        )
        student = makeprediction(
            [[20.5, 20.0, 16, 8], [60.5, 30.0, 20, 10]],
            [9, 9],
            [0.0, math.pi / 6],
        )

        matches = matchpredictions(
            student,
            teacher,
            targets,
            confidencethreshold=0.25,
            iouthreshold=0.5,
            distancethreshold=1.0,
        )

        mapping = {
            int(target): (int(studentindex), int(teacherindex))
            for target, studentindex, teacherindex in zip(
                matches.targetindex,
                matches.studentindex,
                matches.teacherindex,
            )
        }
        assert mapping == {0: (0, 1), 1: (1, 0)}

    def checkmatchingindicesareonetoone(self):
        targets = maketargets()
        teacher = makeprediction(
            [[20, 20, 16, 8], [60, 30, 20, 10]],
            [2, 4],
            [0.0, math.pi / 6],
        )
        student = makeprediction(
            [[20, 20, 16, 8], [60, 30, 20, 10]],
            [0, 0],
            [0.0, math.pi / 6],
        )
        matches = matchpredictions(student, teacher, targets)

        assert isinstance(matches, DistillationMatch)
        assert len(matches.studentindex.unique()) == len(matches.studentindex)
        assert len(matches.teacherindex.unique()) == len(matches.teacherindex)
        assert len(matches.targetindex.unique()) == len(matches.targetindex)


class CheckRouting:

    def checkweightsstayboundedandfollowcomponentdecay(self):
        targets = maketargets()
        teacher = makeprediction([[20, 20, 16, 8]], [2], [0.0])
        matches = DistillationMatch(
            batchindex=torch.tensor([0]),
            studentindex=torch.tensor([0]),
            teacherindex=torch.tensor([0]),
            targetindex=torch.tensor([0]),
        )

        routing = computerouting(targets, teacher, matches)

        assert torch.all(routing.weights >= 0)
        assert torch.all(routing.weights <= 1)
        assert routing.classification >= routing.center
        assert routing.center >= routing.scale
        assert routing.scale >= routing.angle

    @pytest.mark.parametrize(
        "stronger",
        [
            {"psfsigma": 2.0},
            {"noiselevel": 0.2},
            {"downsamplefactor": 4.0},
        ],
    )
    def checkstrongerdegradationdoesnotincreaseweights(self, stronger):
        targets = maketargets()[:1]
        teacher = makeprediction([[20, 20, 16, 8]], [2], [0.0])
        matches = DistillationMatch(
            batchindex=torch.tensor([0]),
            studentindex=torch.tensor([0]),
            teacherindex=torch.tensor([0]),
            targetindex=torch.tensor([0]),
        )
        mild = computerouting(targets, teacher, matches)
        strong = computerouting(targets, teacher, matches, **stronger)

        assert torch.all(strong.weights <= mild.weights + 1e-7)

    def checkshorterobjectdoesnotincreaseweights(self):
        targets = maketargets()[:1]
        shorttargets = targets.clone()
        shorttargets[:, 4:6] *= 0.25
        teacher = makeprediction([[20, 20, 16, 8]], [2], [0.0])
        matches = DistillationMatch(
            batchindex=torch.tensor([0]),
            studentindex=torch.tensor([0]),
            teacherindex=torch.tensor([0]),
            targetindex=torch.tensor([0]),
        )

        regular = computerouting(targets, teacher, matches)
        short = computerouting(shorttargets, teacher, matches)

        assert torch.all(short.weights <= regular.weights + 1e-7)

    def checksquareobjectreducesangleweight(self):
        targets = maketargets()[:1]
        squaretargets = targets.clone()
        squaretargets[:, 4:6] = 12
        teacher = makeprediction([[20, 20, 16, 8]], [2], [0.0])
        matches = DistillationMatch(
            batchindex=torch.tensor([0]),
            studentindex=torch.tensor([0]),
            teacherindex=torch.tensor([0]),
            targetindex=torch.tensor([0]),
        )

        elongated = computerouting(targets, teacher, matches)
        square = computerouting(squaretargets, teacher, matches)

        assert square.angle.item() < elongated.angle.item()


class CheckLoss:

    def checkfourcomponentlossbackpropagatesonlytostudent(self):
        targets = maketargets()[:1]
        teacher = makeprediction([[20, 20, 16, 8]], [2], [0.0])
        student = makeprediction([[21, 19, 15, 9]], [3], [math.pi / 12], requiresgrad=True)
        matches = DistillationMatch(
            batchindex=torch.tensor([0]),
            studentindex=torch.tensor([0]),
            teacherindex=torch.tensor([0]),
            targetindex=torch.tensor([0]),
        )
        routing = computerouting(targets, teacher, matches)

        losses = computefourcomponentloss(student, teacher, matches, routing, imagesize=64)
        losses["total"].backward()

        assert student.values.grad is not None
        assert student.values.grad.abs().sum().item() > 0
        assert teacher.values.grad is None
        assert all(torch.isfinite(value) for value in losses.values())

    def checkemptymatchreturnsstudenttypedzero(self):
        student = makeprediction([[20, 20, 16, 8]], [2], [0.0], requiresgrad=True)
        teacher = makeprediction([[20, 20, 16, 8]], [2], [0.0])
        empty = torch.empty(0, dtype=torch.long)
        matches = DistillationMatch(empty, empty, empty, empty)
        routing = computerouting(torch.zeros(0, 187), teacher, matches)

        losses = computefourcomponentloss(student, teacher, matches, routing, imagesize=64)

        assert losses["total"].device == student.values.device
        assert losses["total"].dtype == student.values.dtype
        assert losses["total"].item() == 0.0
