"""
Rotated NMS module with automatic fallback
优先使用C++扩展，失败时回退到纯Python实现
"""

try:
    # 尝试导入C++扩展版本
    from .nms_rotated_wrapper import obb_nms, poly_nms
    _USE_CPP_EXT = True
except (ImportError, ModuleNotFoundError) as e:
    # 回退到纯Python实现
    import warnings
    warnings.warn(
        f"C++ extension 'nms_rotated_ext' not available: {e}\n"
        "Falling back to pure Python implementation (slower but functional).\n"
        "To use C++ version, run: cd utils/nms_rotated && python setup.py install",
        RuntimeWarning
    )
    from ..rotatednms import rotatednms

    def obb_nms(dets, scores, iou_thr, device_id=None):
        """Preserve the upstream extension API while using strict Shapely NMS."""
        import numpy as np
        import torch

        isnumpy = isinstance(dets, np.ndarray)
        workingdets = torch.as_tensor(dets)
        workingscores = torch.as_tensor(scores, device=workingdets.device)
        valid = workingdets[:, 2:4].amin(1) >= 0.001
        validindices = torch.where(valid)[0]
        if validindices.numel() == 0:
            keep = torch.empty(0, dtype=torch.long, device=workingdets.device)
        else:
            selected = rotatednms(
                workingdets[validindices],
                workingscores[validindices],
                iou_thr,
            )
            keep = validindices[selected]
        if isnumpy:
            keep = keep.cpu().numpy()
        return dets[keep], keep

    def poly_nms(dets, iou_thr, device_id=None):
        """The strict Python fallback currently supports rotated boxes only."""
        raise NotImplementedError("poly_nms requires the compiled upstream extension")

    _USE_CPP_EXT = False

__all__ = ['obb_nms', 'poly_nms']
