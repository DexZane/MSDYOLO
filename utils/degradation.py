"""PSF 模糊、降采样、噪声和尺寸恢复组成的传感器退化链。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSFBlur(nn.Module):
    """使用逐通道高斯核近似传感器点扩散函数。"""

    def __init__(self, kernelsize=5, sigma=1.0):
        super().__init__()
        if kernelsize <= 0 or kernelsize % 2 == 0:
            raise ValueError("PSF kernel size must be a positive odd number")
        if sigma <= 0:
            raise ValueError("PSF sigma must be positive")
        self.kernelsize = kernelsize
        self.sigma = sigma
        self.register_buffer("kernel", self.creategaussiankernel())

    def creategaussiankernel(self):
        """构造归一化二维高斯核。"""
        coordinate = torch.arange(self.kernelsize, dtype=torch.float32)
        coordinate = coordinate - self.kernelsize // 2
        ycoordinate, xcoordinate = torch.meshgrid(coordinate, coordinate, indexing="ij")
        kernel = torch.exp(-(xcoordinate.square() + ycoordinate.square()) / (2 * self.sigma**2))
        return (kernel / kernel.sum()).view(1, 1, self.kernelsize, self.kernelsize)

    def forward(self, images):
        """逐通道执行 PSF 卷积。"""
        channels = images.shape[1]
        kernel = self.kernel.to(device=images.device, dtype=images.dtype).repeat(channels, 1, 1, 1)
        return F.conv2d(images, kernel, padding=self.kernelsize // 2, groups=channels)


class ImageDegradation(nn.Module):
    """执行 PSF → 降采样 → 噪声 → 恢复尺寸。"""

    def __init__(
        self,
        enablepsf=False,
        enabledownsample=False,
        enablenoise=False,
        psfkernelsize=5,
        psfsigma=1.0,
        downsamplescale=2.0,
        noisetype="gaussian",
        noiselevel=0.01,
        upsamplemode="bilinear",
        seed=None,
    ):
        super().__init__()
        if enabledownsample and downsamplescale <= 1:
            raise ValueError("Downsample scale must be > 1")
        if noiselevel < 0 or noiselevel > 1:
            raise ValueError("Noise level must be in [0, 1]")
        if noisetype not in {"gaussian", "saltpepper"}:
            raise ValueError(f"Unknown noise type: {noisetype}")
        if upsamplemode not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError(f"Invalid upsample mode: {upsamplemode}")

        self.enablepsf = enablepsf
        self.enabledownsample = enabledownsample
        self.enablenoise = enablenoise
        self.psfkernelsize = psfkernelsize
        self.psfsigma = psfsigma
        self.downsamplescale = downsamplescale
        self.noisetype = noisetype
        self.noiselevel = noiselevel
        self.upsamplemode = upsamplemode
        self.seed = seed
        self.noisegenerator = torch.Generator()
        if seed is not None:
            self.noisegenerator.manual_seed(seed)
        self.psfblur = PSFBlur(psfkernelsize, psfsigma) if enablepsf else None

    def getconfig(self):
        """返回可记录和复现的连续小写配置。"""
        return {
            "enablepsf": self.enablepsf,
            "enabledownsample": self.enabledownsample,
            "enablenoise": self.enablenoise,
            "psfkernelsize": self.psfkernelsize,
            "psfsigma": self.psfsigma,
            "downsamplescale": self.downsamplescale,
            "noisetype": self.noisetype,
            "noiselevel": self.noiselevel,
            "upsamplemode": self.upsamplemode,
            "seed": self.seed,
        }

    def applynoise(self, images):
        """使用持久化随机数生成器产生可复现但不重复的噪声。"""
        if self.noiselevel == 0:
            return images
        generator = self.noisegenerator if self.seed is not None else None
        if self.noisetype == "gaussian":
            noise = torch.randn(
                images.shape,
                generator=generator,
                device="cpu",
                dtype=images.dtype,
            ).to(images.device)
            return (images + noise * self.noiselevel).clamp(0, 1)
        mask = torch.rand(
            images.shape,
            generator=generator,
            device="cpu",
            dtype=images.dtype,
        ).to(images.device)
        result = images.clone()
        result[mask < self.noiselevel / 2] = 1
        result[mask > 1 - self.noiselevel / 2] = 0
        return result

    def interpolate(self, images, size, mode):
        """统一处理 nearest 不接受 align_corners 的差异。"""
        arguments = {"size": size, "mode": mode}
        if mode != "nearest":
            arguments["align_corners"] = False
        return F.interpolate(images, **arguments)

    def forward(self, images):
        """应用退化链并保持输入尺寸和值域。"""
        if not (self.enablepsf or self.enabledownsample or self.enablenoise):
            return images
        originalsize = images.shape[-2:]
        degraded = self.psfblur(images) if self.psfblur is not None else images
        if self.enabledownsample:
            reducedsize = tuple(max(1, int(size / self.downsamplescale)) for size in originalsize)
            degraded = self.interpolate(degraded, reducedsize, "bilinear")
        if self.enablenoise:
            degraded = self.applynoise(degraded)
        if self.enabledownsample:
            degraded = self.interpolate(degraded, originalsize, self.upsamplemode)
        return degraded.clamp(0, 1)

    def __repr__(self):
        return (
            f"ImageDegradation(psf={self.enablepsf}, "
            f"down={self.enabledownsample}, noise={self.enablenoise}, "
            f"scale={self.downsamplescale})"
        )
