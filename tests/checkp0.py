"""Phase 1 基础设施回归测试。"""

import pytest
import torch
import torch.nn as nn

from utils.clearbranch import ClearBranchForward
from utils.config import MSDYOLOConfig
from utils.degradation import ImageDegradation
from utils.sparse import PredictionMatcher, SparsePredictionExtractor
from utils.trainer import MSDYOLOTrainer


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv2d(3, 16, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(16)
        self.nc = 16

    def forward(self, images):
        values = self.batchnorm(self.convolution(images))
        if self.training:
            return [values]
        batch = images.shape[0]
        decoded = torch.zeros(batch, 2, 201, device=images.device)
        decoded[..., :4] = 8
        decoded[..., 4:21] = 0.9
        decoded[..., 21 + 90] = 1
        return decoded, [values]


class CheckDegradation:

    def checkdisabledpipelineisidentity(self):
        degradation = ImageDegradation()
        images = torch.rand(2, 3, 32, 32)
        assert torch.equal(images, degradation(images))

    def checkenabledpipelinechangesinputandkeepsshape(self):
        degradation = ImageDegradation(
            enablepsf=True,
            enabledownsample=True,
            enablenoise=True,
            downsamplescale=2,
            noiselevel=0.01,
            seed=42,
        )
        images = torch.rand(2, 3, 32, 32)
        degraded = degradation(images)
        assert degraded.shape == images.shape
        assert not torch.allclose(images, degraded)
        assert degraded.min() >= 0
        assert degraded.max() <= 1

    @pytest.mark.parametrize(
        "arguments",
        [
            {"psfkernelsize": 4},
            {"psfsigma": -1},
            {"enabledownsample": True, "downsamplescale": 1},
            {"noiselevel": 2},
        ],
    )
    def checkinvalidparametersfail(self, arguments):
        with pytest.raises(ValueError):
            ImageDegradation(enablepsf=True, **arguments)

    def checkseedreproducessequencewithoutrepeatingbatches(self):
        first = ImageDegradation(enablenoise=True, noiselevel=0.1, seed=7)
        second = ImageDegradation(enablenoise=True, noiselevel=0.1, seed=7)
        images = torch.full((1, 3, 16, 16), 0.5)
        firstbatch = first(images)
        assert torch.equal(firstbatch, second(images))
        assert not torch.equal(firstbatch, first(images))


class CheckClearBranch:

    def checkclearbranchkeepsbatchnormandrestoresmode(self):
        model = SimpleModel().train()
        manager = ClearBranchForward(model)
        initialmean = model.batchnorm.running_mean.clone()
        output = manager.forwardclearbranch(torch.rand(2, 3, 16, 16), extractsparse=False)
        assert isinstance(output, tuple)
        assert torch.equal(initialmean, model.batchnorm.running_mean)
        assert model.training

    def checkdegradedbranchhasgradient(self):
        model = SimpleModel()
        manager = ClearBranchForward(model)
        predictions, loss = manager.forwarddegradedbranch(torch.rand(2, 3, 16, 16))
        assert loss is None
        predictions[0].mean().backward()
        assert model.convolution.weight.grad is not None


class CheckSparse:

    def checkevaltupleextractstopk(self):
        model = SimpleModel().eval()
        predictions = model(torch.rand(2, 3, 16, 16))
        extractor = SparsePredictionExtractor(confidencethreshold=0.1, topk=1)
        results = extractor.extractfromyolooutput(predictions, batchsize=2)
        assert len(results) == 2
        assert all(result["boxes"].shape == (1, 5) for result in results)

    def checkmatcherisonetoone(self):
        predictions = {
            "boxes": torch.tensor([[10, 10, 5, 3, 0.0], [30, 30, 5, 3, 0.0]]),
            "classids": torch.tensor([1, 2]),
        }
        degraded = {
            "boxes": torch.tensor([[10.1, 10, 5, 3, 0.0], [30.1, 30, 5, 3, 0.0]]),
            "classids": torch.tensor([1, 2]),
        }
        matcher = PredictionMatcher(matchthreshold=0.5)
        clearindices, degradedindices = matcher.matchpredictions(predictions, degraded)
        assert clearindices.tolist() == [0, 1]
        assert degradedindices.tolist() == [0, 1]


class CheckTrainer:

    def checkbaselinemodedisablesnewmodules(self):
        config = MSDYOLOConfig()
        trainer = MSDYOLOTrainer(SimpleModel(), config, torch.device("cpu"))
        assert trainer.isbaselinemode()
        assert trainer.degradation is None
        assert not trainer.distillationenabled

    def checktuplelossisretained(self):
        model = SimpleModel()
        trainer = MSDYOLOTrainer(model, MSDYOLOConfig(), torch.device("cpu"))

        def computeloss(predictions, targets):
            loss = predictions[0].mean()
            return loss, loss.detach().repeat(4)

        result = trainer.processbatch(
            torch.rand(2, 3, 16, 16),
            torch.zeros(0, 187),
            computeloss,
        )
        assert result["loss"].requires_grad
        assert result["lossitems"].shape == (4,)

    def checkphaseoneblockscompletedistillation(self):
        config = MSDYOLOConfig()
        config.set("distillation.enabled", True)
        assert not config.validate()
