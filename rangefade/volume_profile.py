"""Volume profile over an arbitrary bar window: POC, VAH, VAL.

Bar volume is spread uniformly across the price bins its high-low span covers —
the standard approximation when only OHLCV (not tick) data is available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Profile:
    poc: float
    vah: float
    val: float
    bin_edges: np.ndarray
    bin_volume: np.ndarray


def volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    bin_points: float = 1.0,
    value_area: float = 0.70,
) -> Profile | None:
    if len(high) == 0:
        return None
    lo = float(np.min(low))
    hi = float(np.max(high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    start = np.floor(lo / bin_points) * bin_points
    n_bins = max(1, int(np.ceil((hi - start) / bin_points)) + 1)
    edges = start + np.arange(n_bins + 1) * bin_points
    vol_bins = np.zeros(n_bins)

    vol = volume.astype(float).copy()
    if vol.sum() <= 0:  # no volume data: fall back to time-at-price (each bar weight 1)
        vol = np.ones(len(high))

    for h, l, v in zip(high, low, vol):
        if v <= 0:
            continue
        i0 = int(np.clip((l - start) // bin_points, 0, n_bins - 1))
        i1 = int(np.clip((h - start) // bin_points, 0, n_bins - 1))
        if i1 <= i0:
            vol_bins[i0] += v
            continue
        span = h - l
        for i in range(i0, i1 + 1):
            seg_lo = max(l, edges[i])
            seg_hi = min(h, edges[i + 1])
            if seg_hi > seg_lo:
                vol_bins[i] += v * (seg_hi - seg_lo) / span

    poc_i = int(np.argmax(vol_bins))
    total = vol_bins.sum()
    target = value_area * total
    lo_i = hi_i = poc_i
    acc = vol_bins[poc_i]
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        below = vol_bins[lo_i - 1] if lo_i > 0 else -1.0
        above = vol_bins[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += vol_bins[hi_i]
        else:
            lo_i -= 1
            acc += vol_bins[lo_i]

    centers = (edges[:-1] + edges[1:]) / 2.0
    return Profile(
        poc=float(centers[poc_i]),
        vah=float(edges[hi_i + 1]),
        val=float(edges[lo_i]),
        bin_edges=edges,
        bin_volume=vol_bins,
    )
