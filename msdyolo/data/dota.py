"""DOTA v1.5 label parsing and patch geometry."""

from dataclasses import dataclass
import math
from pathlib import Path


DOTA15_CLASSES: tuple[str, ...] = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
)


@dataclass(frozen=True)
class DotaObject:
    coordinates: tuple[float, float, float, float, float, float, float, float]
    classname: str
    difficult: int


def is_degenerate_polygon(
    coordinates: tuple[float, float, float, float, float, float, float, float]
) -> bool:
    """Return whether a quadrilateral has zero signed area."""
    points = tuple(zip(coordinates[::2], coordinates[1::2]))
    signed_area = sum(
        xcurrent * ynext - xnext * ycurrent
        for (xcurrent, ycurrent), (xnext, ynext) in zip(points, points[1:] + points[:1])
    )
    scale = max(1.0, *(abs(value) for value in coordinates))
    return math.isclose(signed_area, 0.0, abs_tol=1e-12 * scale * scale)


def parse_dota_label(path: Path) -> list[DotaObject]:
    """Parse strict ten-column DOTA v1.5 labels from *path*."""
    objects = []
    for linenumber, rawline in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = rawline.strip()
        if not line:
            continue
        if line.startswith(("imagesource:", "gsd:")):
            continue

        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"{path}:{linenumber}: expected 10 label columns")

        try:
            coordinates = tuple(float(value) for value in parts[:8])
        except ValueError as error:
            raise ValueError(f"{path}:{linenumber}: coordinates must be numeric") from error
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{path}:{linenumber}: coordinates must be finite")
        if all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{linenumber}: normalized coordinates are not supported")

        classname = parts[8]
        if classname not in DOTA15_CLASSES:
            raise ValueError(f"{path}:{linenumber}: unknown class {classname!r}")

        try:
            difficult = int(parts[9])
        except ValueError as error:
            raise ValueError(f"{path}:{linenumber}: difficult must be 0, 1, or 2") from error
        if difficult not in {0, 1, 2}:
            raise ValueError(f"{path}:{linenumber}: difficult must be 0, 1, or 2")
        if is_degenerate_polygon(coordinates):
            raise ValueError(f"{path}:{linenumber}: degenerate polygon")

        objects.append(DotaObject(coordinates, classname, difficult))
    return objects


def clip_object_to_patch(obj: DotaObject, xstart: int, ystart: int, subsize: int) -> DotaObject:
    """Translate *obj* into a patch and clamp its vertices to patch bounds."""
    coordinates = []
    for index, value in enumerate(obj.coordinates):
        start = xstart if index % 2 == 0 else ystart
        coordinates.append(max(0.0, min(float(subsize), value - start)))
    return DotaObject(tuple(coordinates), obj.classname, obj.difficult)


def format_dota_object(obj: DotaObject) -> str:
    """Format an object as one ten-column DOTA v1.5 label line."""
    coordinates = " ".join(f"{value:.1f}" for value in obj.coordinates)
    return f"{coordinates} {obj.classname} {obj.difficult}"
