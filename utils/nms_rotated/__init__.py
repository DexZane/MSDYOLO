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
    from ..nms_rotated_pure import obb_nms, poly_nms
    _USE_CPP_EXT = False

__all__ = ['obb_nms', 'poly_nms']

