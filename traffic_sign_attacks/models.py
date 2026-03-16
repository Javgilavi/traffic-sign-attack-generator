from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    split: str
    class_id: int
    label: str
    shape: str
    width: int
    height: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    sample_id: str

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)
