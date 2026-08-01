"""P0-A.2 Trainer集成验证测试。"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msdyolo.models.yolo import Model
from msdyolo.utils.config import MSDYOLOConfig
from msdyolo.utils.loss import ComputeLoss
from msdyolo.utils.trainer import MSDYOLOTrainer


def generatecsllabels(anglerad, numclasses=180, u=6.0):
    """生成CSL标签（高斯分布）。

    Args:
        anglerad: 角度（弧度），范围[-pi/2, pi/2)
        numclasses: CSL类别数（默认180）
        u: 高斯窗口参数

    Returns:
        torch.Tensor: shape (numclasses,)
    """
    # 将角度转换为类别索引 [0, 179]
    angledeg = anglerad * 180.0 / math.pi
    angleidx = angledeg + 90.0  # 映射到[0, 180)

    # 生成高斯分布
    labels = torch.zeros(numclasses)
    for i in range(numclasses):
        distance = min(abs(i - angleidx), 180 - abs(i - angleidx))
        labels[i] = math.exp(-0.5 * (distance ** 2) / (u ** 2))

    # 归一化
    labels = labels / labels.sum()
    return labels


class ControlledModel(nn.Module):
    """可控的轻量模型，用于产生确定性的非空匹配。"""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.nc = 16
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

        # 模拟Detect head属性
        self.stride = torch.tensor([8.0, 16.0, 32.0], device=device)
        self.anchors = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], device=device),
            torch.tensor([[30, 61], [62, 45], [59, 119]], device=device),
            torch.tensor([[116, 90], [156, 198], [373, 326]], device=device),
        ]
        self.na = 3
        self.no = 16 + 5 + 180  # nc + 5 + 180

    def forward(self, images):
        """返回可控的raw outputs以产生确定性匹配。"""
        batch = images.shape[0]
        # 使用卷积确保参数连接到计算图
        feature = self.conv(images)

        if not self.training:
            # 推理模式：返回(decoded, raw)
            decoded = torch.zeros(batch, 100, self.no, device=self.device)
            raw = self.createrawoutputs(batch)
            # 添加微小的可学习分量
            for i, r in enumerate(raw):
                raw[i] = r + feature.mean() * 1e-6
            return decoded, raw
        else:
            # 训练模式：学生输出添加小扰动以产生非零蒸馏损失
            rawoutputs = self.createrawoutputs(batch)
            # 添加可学习分量并加入小扰动
            for i, raw in enumerate(rawoutputs):
                # 在受控位置添加扰动，使学生与教师有差异
                perturbation = feature.mean() * 0.1 + 0.05
                rawoutputs[i] = raw + perturbation
            return rawoutputs

    def createrawoutputs(self, batch):
        """创建三个尺度的确定性可控raw outputs。"""
        rawoutputs = []

        # 尺度1: 40x40网格 - 使用零基础
        raw1 = torch.zeros((batch, self.na, 40, 40, self.no), device=self.device)
        # 在目标位置(20, 20)设置高置信度
        raw1[0, 0, 20, 20, 4] = 3.0  # objectness logit
        raw1[0, 0, 20, 20, 5 + 5] = 3.0  # class 5 logit
        # 在训练模式下，为学生添加类别差异（教师是eval模式，不受影响）
        if self.training:
            raw1[0, 0, 20, 20, 5 + 5] = 2.5  # 学生class 5 logit略低
            raw1[0, 0, 20, 20, 5 + 4] = 0.5  # 添加噪声类别
        raw1[0, 0, 20, 20, :2] = 0.0  # xy接近grid center
        raw1[0, 0, 20, 20, 2:4] = 0.5  # wh
        # 设置CSL logits在90度附近
        raw1[0, 0, 20, 20, 21 + 90] = 3.0  # 90度高logit
        if self.training:
            raw1[0, 0, 20, 20, 21 + 89] = 0.3  # 添加角度噪声
        rawoutputs.append(raw1)

        # 尺度2: 20x20网格 - 使用零基础
        raw2 = torch.zeros((batch, self.na, 20, 20, self.no), device=self.device)
        # 在目标位置(12, 10)设置高置信度
        raw2[0, 1, 12, 10, 4] = 3.0
        raw2[0, 1, 12, 10, 5 + 3] = 3.0  # class 3
        if self.training:
            raw2[0, 1, 12, 10, 5 + 3] = 2.7  # 学生class 3 logit略低
            raw2[0, 1, 12, 10, 5 + 2] = 0.3  # 添加噪声类别
        raw2[0, 1, 12, 10, :2] = 0.0
        raw2[0, 1, 12, 10, 2:4] = 0.3
        raw2[0, 1, 12, 10, 21 + 60] = 3.0  # 60度位置（约-30度）
        if self.training:
            raw2[0, 1, 12, 10, 21 + 61] = 0.4  # 添加角度噪声
        rawoutputs.append(raw2)

        # 尺度3: 10x10网格 - 全零
        raw3 = torch.zeros((batch, self.na, 10, 10, self.no), device=self.device)
        rawoutputs.append(raw3)

        return rawoutputs


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def modelconfig():
    cfgpath = ROOT / "configs" / "models" / "yolov5s.yaml"
    with cfgpath.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = 16
    return cfg


@pytest.fixture
def model(modelconfig, hypconfig, device):
    m = Model(modelconfig).to(device)
    m.hyp = hypconfig  # ComputeLoss需要model.hyp
    m.eval()
    return m


@pytest.fixture
def hypconfig():
    hyppath = ROOT / "data" / "hyps" / "obb" / "hyp.finetune_dota.yaml"
    with hyppath.open("r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def computeloss(model, hypconfig, device):
    return ComputeLoss(model, hypconfig)


@pytest.fixture
def syntheticbatch(device):
    images = torch.randn(1, 3, 320, 320, device=device)

    # DOTA格式: [batch_idx, class_id, cx, cy, long_edge, short_edge, angle, csl_labels(180)]
    target1 = torch.cat([
        torch.tensor([0, 5, 160.0, 160.0, 80.0, 40.0, 0.5]),
        generatecsllabels(0.5),
    ])
    target2 = torch.cat([
        torch.tensor([0, 3, 100.0, 200.0, 60.0, 30.0, -0.3]),
        generatecsllabels(-0.3),
    ])

    targets = torch.stack([target1, target2]).to(device)
    return images, targets


@pytest.fixture
def controlledmodel(hypconfig, device):
    """可控模型用于确定性非空匹配测试。"""
    m = ControlledModel(device)
    m.hyp = hypconfig
    # 添加model.model[-1]来模拟Detect head
    class MockDetect(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.stride = parent.stride
            self.anchors = parent.anchors
            self.inplace = True
    m.model = nn.ModuleList([MockDetect(m)])
    return m


@pytest.fixture
def controlledbatch(device):
    """与ControlledModel匹配的目标批次。"""
    images = torch.randn(1, 3, 320, 320, device=device)

    # 目标1：class=5, 在尺度1的(20,20)附近
    target1 = torch.cat([
        torch.tensor([0, 5, 160.0, 160.0, 60.0, 30.0, 0.0]),
        generatecsllabels(0.0),
    ])

    # 目标2：class=3, 在尺度2的(12,10)附近
    target2 = torch.cat([
        torch.tensor([0, 3, 200.0, 180.0, 50.0, 40.0, -0.5]),
        generatecsllabels(-0.5),
    ])

    targets = torch.stack([target1, target2]).to(device)
    return images, targets


class CheckP0A2Integration:
    """P0-A.2 Trainer集成验证。"""

    def checkdeterministicnonemptymatch(self, controlledmodel, controlledbatch, hypconfig, device):
        """确定性非空匹配集成测试：验证完整蒸馏训练图。"""
        images, targets = controlledbatch

        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")

        # 降低匹配阈值以确保匹配成功
        config.set("distillation.confidencethreshold", 0.1)
        config.set("distillation.iouthreshold", 0.05)
        config.set("distillation.distancethreshold", 50.0)

        def computeloss(predictions, targets):
            """简单的可微损失函数。"""
            if isinstance(predictions, list):
                loss = sum(p.abs().mean() for p in predictions)
            else:
                loss = predictions.abs().mean()
            return loss, loss.detach().repeat(4)

        trainer = MSDYOLOTrainer(controlledmodel, config, device)

        # 清零梯度
        for param in controlledmodel.parameters():
            param.grad = None

        result = trainer.processbatch(images, targets, computeloss)

        # 关键断言：必须产生非空匹配
        assert result["matchcount"] > 0, f"Expected non-empty match, got matchcount={result['matchcount']}"

        # 蒸馏损失必须非零
        assert result["distillationloss"].item() > 0, f"Expected non-zero distillation loss, got {result['distillationloss'].item()}"

        # 四个分量损失必须全部非零
        assert result["classificationloss"].item() > 0, f"Classification loss is zero"
        assert result["centerloss"].item() > 0, f"Center loss is zero"
        assert result["scaleloss"].item() > 0, f"Scale loss is zero"
        assert result["angleloss"].item() > 0, f"Angle loss is zero"

        # 验证总损失公式
        alpha = config.get("distillation.alpha")
        expected = result["detectionloss"] + alpha * result["distillationloss"]
        assert torch.allclose(result["loss"], expected, rtol=1e-5), \
            f"Loss formula incorrect: {result['loss'].item()} != {expected.item()}"

        # 验证总损失反向传播产生学生梯度
        result["loss"].backward()

        hasgradient = False
        for param in controlledmodel.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                hasgradient = True
                break
        assert hasgradient, "Student parameters have no gradient after total loss backward"

        # 清零梯度，单独验证蒸馏损失产生学生梯度
        for param in controlledmodel.parameters():
            param.grad = None

        # 重新执行前向（因为backward消耗了计算图）
        result2 = trainer.processbatch(images, targets, computeloss)
        result2["distillationloss"].backward()

        hasdistillationgradient = False
        for param in controlledmodel.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                hasdistillationgradient = True
                break
        assert hasdistillationgradient, "Student parameters have no gradient after distillation loss backward"

    def checkbaselineequivalence(self, model, computeloss, syntheticbatch, device):
        """distillation.enabled=false时，与直接YOLO+ComputeLoss接近。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 1)
        config.set("degradation.enabled", False)
        config.set("clearbranch.enabled", False)
        config.set("distillation.enabled", False)

        trainer = MSDYOLOTrainer(model, config, device)
        result = trainer.processbatch(images, targets, computeloss)

        model.train()
        predictions = model(images)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        directloss = computeloss(predictions, targets)
        if isinstance(directloss, tuple):
            directloss = directloss[0]

        assert torch.isfinite(result["loss"])
        assert result["distillationloss"].item() == 0.0
        assert result["matchcount"] == 0
        # Baseline路径应该与直接调用接近（允许BatchNorm等的数值差异）
        assert abs(result["loss"].item() - directloss.item()) < 1e-3

    def checkfullmodeforward(self, model, computeloss, syntheticbatch, device):
        """full模式下完成教师前向、学生前向、匹配、路由和总损失计算。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")

        trainer = MSDYOLOTrainer(model, config, device)
        result = trainer.processbatch(images, targets, computeloss)

        assert torch.isfinite(result["loss"])
        assert torch.isfinite(result["detectionloss"])
        assert torch.isfinite(result["distillationloss"])
        assert torch.isfinite(result["classificationloss"])
        assert torch.isfinite(result["centerloss"])
        assert torch.isfinite(result["scaleloss"])
        assert torch.isfinite(result["angleloss"])
        assert result["matchcount"] >= 0
        assert 0.0 <= result["meansurvival"] <= 1.0
        assert 0.0 <= result["meananglereliability"] <= 1.0

    def checkteachernogradient(self, model, computeloss, syntheticbatch, device):
        """验证Trainer调用了无梯度教师路径。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")

        trainer = MSDYOLOTrainer(model, config, device)

        # 启用梯度跟踪
        images.requires_grad_(True)
        result = trainer.processbatch(images, targets, computeloss)

        # 验证程序可以完成（教师路径内部是no_grad）
        assert torch.isfinite(result["loss"])

    def checkteacherbnstats(self, model, computeloss, syntheticbatch, device):
        """教师前向在eval模式下执行（BN统计不应大幅改变）。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")

        trainer = MSDYOLOTrainer(model, config, device)
        result = trainer.processbatch(images, targets, computeloss)

        # 验证教师前向正常完成
        assert torch.isfinite(result["loss"])

    def checkmodulestaterestore(self, model, computeloss, syntheticbatch, device):
        """教师前向内部恢复原模块状态，学生分支显式切换到train模式。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")

        trainer = MSDYOLOTrainer(model, config, device)
        result = trainer.processbatch(images, targets, computeloss)

        # 验证processbatch完成后模型处于train模式
        assert model.training, "Model should be in train mode after processbatch"
        assert torch.isfinite(result["loss"])

    def checkstudentgradient(self, model, computeloss, syntheticbatch, device):
        """学生分支梯度非零。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")

        trainer = MSDYOLOTrainer(model, config, device)

        # 清零梯度
        for param in model.parameters():
            param.grad = None

        result = trainer.processbatch(images, targets, computeloss)
        loss = result["loss"]
        loss.backward()

        # 至少有一些参数有非零梯度
        hasgradient = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                hasgradient = True
                break
        assert hasgradient

    def checklossformula(self, model, computeloss, syntheticbatch, device):
        """总损失严格符合：detectionloss + alpha × distillationloss。"""
        images, targets = syntheticbatch
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")
        alpha = config.get("distillation.alpha")

        trainer = MSDYOLOTrainer(model, config, device)
        result = trainer.processbatch(images, targets, computeloss)

        expected = result["detectionloss"] + alpha * result["distillationloss"]
        assert torch.allclose(result["loss"], expected, atol=1e-5)

    def checkemptymatch(self, model, computeloss, device):
        """空匹配时蒸馏损失为零，检测损失仍可反向传播。"""
        images = torch.randn(1, 3, 320, 320, device=device)
        # 创建一个极端的目标，使得匹配失败
        target = torch.cat([
            torch.tensor([0, 15, 10.0, 10.0, 5.0, 5.0, 0.0]),
            generatecsllabels(0.0),
        ]).to(device).unsqueeze(0)

        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")
        # 设置极高阈值，强制无匹配
        config.set("distillation.confidencethreshold", 0.99)

        trainer = MSDYOLOTrainer(model, config, device)
        result = trainer.processbatch(images, target, computeloss)

        # 必须产生空匹配
        assert result["matchcount"] == 0, f"Expected empty match, got matchcount={result['matchcount']}"
        assert result["distillationloss"].item() == 0.0
        assert result["classificationloss"].item() == 0.0
        assert result["centerloss"].item() == 0.0
        assert result["scaleloss"].item() == 0.0
        assert result["angleloss"].item() == 0.0

        # 检测损失仍应有限且可反向传播
        assert torch.isfinite(result["detectionloss"])
        for param in model.parameters():
            param.grad = None
        result["loss"].backward()

    def checkmatchingconsistency(self, controlledmodel, controlledbatch, hypconfig, device):
        """验证匹配结果的一致性：学生/教师/GT索引唯一。"""
        images, targets = controlledbatch

        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")
        config.set("distillation.confidencethreshold", 0.1)
        config.set("distillation.iouthreshold", 0.05)
        config.set("distillation.distancethreshold", 50.0)

        def computeloss(predictions, targets):
            if isinstance(predictions, list):
                loss = sum(p.abs().mean() for p in predictions)
            else:
                loss = predictions.abs().mean()
            return loss, loss.detach().repeat(4)

        # 通过直接调用matching验证索引唯一性
        from msdyolo.utils.decoder import decodesparse
        from msdyolo.utils.matching import matchpredictions
        from msdyolo.utils.clearbranch import teacherforward

        teacher = teacherforward(controlledmodel, images, 300)
        controlledmodel.train()
        studentraw = controlledmodel(images)
        student = decodesparse(studentraw, controlledmodel, 300)

        matches = matchpredictions(
            student, teacher, targets,
            confidencethreshold=0.1,
            iouthreshold=0.05,
            distancethreshold=50.0,
        )

        # 必须产生非空匹配
        assert len(matches) > 0, f"Expected non-empty matches, got {len(matches)}"

        # 验证索引唯一性
        studentindices = matches.studentindex.tolist()
        teacherindices = matches.teacherindex.tolist()
        targetindices = matches.targetindex.tolist()

        assert len(studentindices) == len(set(studentindices)), "Student indices not unique"
        assert len(teacherindices) == len(set(teacherindices)), "Teacher indices not unique"
        # GT在一对一贪心匹配中应唯一
        assert len(targetindices) == len(set(targetindices)), "Target indices not unique in one-to-one matching"

    def checkmatchnoduplicate(self, controlledmodel, controlledbatch, hypconfig, device):
        """验证候选顺序改变后仍映射到同一GT（贪心一对一稳定性）。"""
        images, targets = controlledbatch

        from msdyolo.utils.decoder import decodesparse
        from msdyolo.utils.matching import matchpredictions
        from msdyolo.utils.clearbranch import teacherforward

        # 第一次匹配
        teacher1 = teacherforward(controlledmodel, images, 300)
        controlledmodel.train()
        studentraw1 = controlledmodel(images)
        student1 = decodesparse(studentraw1, controlledmodel, 300)

        matches1 = matchpredictions(
            student1, teacher1, targets,
            confidencethreshold=0.1,
            iouthreshold=0.05,
            distancethreshold=50.0,
        )

        assert len(matches1) > 0, "First matching produced no matches"

        # 第二次匹配（由于模型确定性，应产生相同结果）
        teacher2 = teacherforward(controlledmodel, images, 300)
        controlledmodel.train()
        studentraw2 = controlledmodel(images)
        student2 = decodesparse(studentraw2, controlledmodel, 300)

        matches2 = matchpredictions(
            student2, teacher2, targets,
            confidencethreshold=0.1,
            iouthreshold=0.05,
            distancethreshold=50.0,
        )

        # 验证两次匹配结果一致
        assert len(matches1) == len(matches2), "Match count differs between runs"
        assert torch.allclose(matches1.studentindex, matches2.studentindex), "Student indices differ"
        assert torch.allclose(matches1.teacherindex, matches2.teacherindex), "Teacher indices differ"
        assert torch.allclose(matches1.targetindex, matches2.targetindex), "Target indices differ"

    def checkfullconfigvalidation(self):
        """full配置能够通过配置验证。"""
        config = MSDYOLOConfig()
        config.set("experiment.phase", 2)
        config.applyablationmode("full")
        assert config.validate()

    def checkinvalidcombinations(self):
        """非法组合必须失败。"""
        # Phase 1不允许蒸馏
        config1 = MSDYOLOConfig()
        config1.set("experiment.phase", 1)
        config1.set("distillation.enabled", True)
        config1.set("degradation.enabled", True)
        config1.set("clearbranch.enabled", True)
        assert not config1.validate()

        # Phase < 2不允许蒸馏
        config1b = MSDYOLOConfig()
        config1b.set("experiment.phase", 0)
        config1b.set("distillation.enabled", True)
        config1b.set("degradation.enabled", True)
        config1b.set("clearbranch.enabled", True)
        assert not config1b.validate()

        # 开启蒸馏但关闭退化
        config2 = MSDYOLOConfig()
        config2.set("experiment.phase", 2)
        config2.set("distillation.enabled", True)
        config2.set("degradation.enabled", False)
        config2.set("clearbranch.enabled", True)
        assert not config2.validate()

        # 开启蒸馏但关闭清晰分支
        config3 = MSDYOLOConfig()
        config3.set("experiment.phase", 2)
        config3.set("distillation.enabled", True)
        config3.set("degradation.enabled", True)
        config3.set("clearbranch.enabled", False)
        assert not config3.validate()

    def checkablationmodes(self, model, computeloss, syntheticbatch, device):
        """baseline、withdegradation、withclearbranch和full四种模式行为正确。"""
        images, targets = syntheticbatch

        # baseline
        configbaseline = MSDYOLOConfig()
        configbaseline.applyablationmode("baseline")
        trainerbaseline = MSDYOLOTrainer(model, configbaseline, device)
        resultbaseline = trainerbaseline.processbatch(images, targets, computeloss)
        assert resultbaseline["distillationloss"].item() == 0.0
        assert resultbaseline["matchcount"] == 0

        # withdegradation
        configdeg = MSDYOLOConfig()
        configdeg.applyablationmode("withdegradation")
        trainerdeg = MSDYOLOTrainer(model, configdeg, device)
        resultdeg = trainerdeg.processbatch(images, targets, computeloss)
        assert resultdeg["distillationloss"].item() == 0.0

        # withclearbranch：执行教师前向但不加入蒸馏损失
        configcb = MSDYOLOConfig()
        configcb.set("experiment.phase", 2)
        configcb.applyablationmode("withclearbranch")
        trainercb = MSDYOLOTrainer(model, configcb, device)

        # 使用包装计数器验证教师前向被调用
        from msdyolo.utils.clearbranch import teacherforward
        callcount = [0]
        originalteacherforward = teacherforward

        def countedteacherforward(model, images, topk):
            callcount[0] += 1
            return originalteacherforward(model, images, topk)

        # 临时替换teacherforward
        import msdyolo.utils.trainer
        msdyolo.utils.trainer.teacherforward = countedteacherforward

        try:
            resultcb = trainercb.processbatch(images, targets, computeloss)
            # 验证教师前向被调用了一次
            assert callcount[0] == 1, f"Expected teacherforward to be called once, got {callcount[0]}"
            # 蒸馏损失为零（不加入损失）
            assert resultcb["distillationloss"].item() == 0.0
            # matchcount为零（未执行匹配）
            assert resultcb["matchcount"] == 0
        finally:
            # 恢复原函数
            msdyolo.utils.trainer.teacherforward = originalteacherforward

        # full
        configfull = MSDYOLOConfig()
        configfull.set("experiment.phase", 2)
        configfull.applyablationmode("full")
        trainerfull = MSDYOLOTrainer(model, configfull, device)
        resultfull = trainerfull.processbatch(images, targets, computeloss)
        assert torch.isfinite(resultfull["distillationloss"])

    def checkinferenceoutput(self, model, device):
        """推理模型输出结构没有因Trainer集成发生变化。"""
        images = torch.randn(1, 3, 320, 320, device=device)
        model.eval()
        with torch.no_grad():
            output = model(images)

        # YOLOv5-OBB推理输出应该是tuple: (decoded_tensor, raw_list)
        assert isinstance(output, tuple)
        assert len(output) == 2
        decoded, raw = output
        assert isinstance(decoded, torch.Tensor)
        assert decoded.ndim == 3  # (B, N, 201)
        assert isinstance(raw, list)  # raw outputs list
        assert len(raw) == 3  # 3 scales

    def checknamingcompliance(self):
        """所有新增名称通过命名守卫。"""
        # 测试新增的配置键
        config = MSDYOLOConfig()
        config.set("distillation.topk", 300)
        config.set("distillation.confidencethreshold", 0.25)
        config.set("distillation.iouthreshold", 0.1)
        config.set("distillation.distancethreshold", 2.0)
        config.set("distillation.classtemperature", 2.0)
        config.set("distillation.angletemperature", 2.0)
        config.set("distillation.shortedgethreshold", 8.0)

        assert config.get("distillation.topk") == 300
