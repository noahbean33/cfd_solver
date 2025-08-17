"""
Geometry helpers for potential_flow: rectangular obstacles and masks.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Rect:
    imin: int
    imax: int
    jmin: int
    jmax: int
    color: str = "gray"


def rect_mask(nx: int, ny: int, rect: Rect) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    mask[rect.imin:rect.imax, rect.jmin:rect.jmax] = True
    return mask


def combine_masks(masks: list[np.ndarray], ny: int, nx: int) -> np.ndarray:
    if not masks:
        return np.zeros((ny, nx), dtype=bool)
    m = np.zeros((ny, nx), dtype=bool)
    for mk in masks:
        m |= mk
    return m
