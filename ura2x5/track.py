"""CFAR detection, bearing tracking and the bearing-time record (Track B, G11).

Track A takes the global maximum of each azimuth scan: one target, no threshold, no track
continuity. This module adds a CFAR threshold across azimuth (so the number of detections
per scan is data-driven), nearest-neighbour association with gating, and an alpha-beta
tracker in bearing / bearing-rate.
"""
import numpy as np

from .geometry import wrap180

__all__ = ["cfar_peaks", "AlphaBetaTracker", "track_scans", "btr_image"]


def cfar_peaks(phi_grid, P, guard_deg=30.0, train_deg=40.0, alpha_db=8.0,
                min_separation_deg=15.0):
    """
    Cell-averaging CFAR across a circular azimuth scan, returning refined peak azimuths.

    A cell is a detection when it exceeds its local background (mean of the training cells,
    excluding guard cells on both sides) by `alpha_db`, and is a local maximum. Close
    detections are thinned to the strongest within `min_separation_deg`.

    The windows are specified in **degrees**, not cells, because they must be sized against
    the beamwidth rather than the scan grid: this ring's HPBW is ~41 deg, so a guard window
    narrower than about half the beamwidth puts the target's own main lobe into its own
    background estimate and the threshold rises above the target. Default `guard_deg` is
    therefore wide enough to clear the main lobe at f0.
    """
    from .bearing import refine_peak_deg

    K = len(P)
    step = phi_grid[1] - phi_grid[0]
    n_guard = max(1, int(round(guard_deg / step)))
    n_train = max(2, int(round(train_deg / step)))
    if 2 * (n_guard + n_train) >= K:
        raise ValueError("guard + training windows exceed the scan; reduce guard_deg/train_deg")

    idx = np.arange(K)
    off = np.concatenate([np.arange(-n_guard - n_train, -n_guard),
                          np.arange(n_guard + 1, n_guard + 1 + n_train)])
    bg = np.array([P[(i + off) % K].mean() for i in range(K)])
    thr = bg * 10 ** (alpha_db / 10)

    is_peak = (P > np.roll(P, 1)) & (P >= np.roll(P, -1)) & (P > thr)
    cand = idx[is_peak]
    if cand.size == 0:
        return [], thr

    order = cand[np.argsort(P[cand])[::-1]]
    chosen = []
    for i in order:
        if all(abs(wrap180(phi_grid[i] - phi_grid[j])) >= min_separation_deg for j in chosen):
            chosen.append(i)
    chosen.sort()

    peaks = []
    for i in chosen:
        local = np.array([(i - 1) % K, i, (i + 1) % K])
        peaks.append(refine_peak_deg(phi_grid, np.where(np.isin(idx, local), P, -np.inf)))
    return peaks, thr


class AlphaBetaTracker:
    """
    Alpha-beta tracker on (bearing, bearing rate), with circular-wrapped innovations.
    `alpha` smooths position, `beta` smooths rate; `gate_deg` rejects associations further
    than that from the prediction.
    """

    def __init__(self, bearing_deg, rate_dps=0.0, alpha=0.5, beta=0.1, gate_deg=12.0,
                 track_id=0):
        self.b, self.v = float(bearing_deg), float(rate_dps)
        self.alpha, self.beta, self.gate = alpha, beta, gate_deg
        self.id = track_id
        self.history = [(0, self.b)]
        self.misses = 0

    def predict(self, dt):
        return (self.b + self.v * dt) % 360.0

    def update(self, measurement_deg, dt, scan_index):
        pred = self.predict(dt)
        if measurement_deg is None:
            self.b, self.misses = pred, self.misses + 1
        else:
            innov = wrap180(measurement_deg - pred)
            self.b = (pred + self.alpha * innov) % 360.0
            self.v = self.v + (self.beta / dt) * innov
            self.misses = 0
        self.history.append((scan_index, self.b))
        return self.b


def track_scans(scans, phi_grid, dt=1.0, max_misses=3, alpha=0.5, beta=0.1, gate_deg=12.0,
                 **cfar_kw):
    """
    Run CFAR + association + alpha-beta tracking over a sequence of azimuth scans.

    `scans` is (n_scans, len(phi_grid)) of scanned power. Returns (tracks, detections) where
    `tracks` is a list of AlphaBetaTracker and `detections` the per-scan CFAR peak lists.
    """
    tracks, detections, next_id = [], [], 0
    for k, P in enumerate(scans):
        peaks, _ = cfar_peaks(phi_grid, P, **cfar_kw)
        detections.append(peaks)
        unused = list(peaks)

        for t in tracks:
            if t.misses > max_misses:
                continue
            pred = t.predict(dt)
            if unused:
                d = [abs(wrap180(p - pred)) for p in unused]
                j = int(np.argmin(d))
                if d[j] <= gate_deg:
                    t.update(unused.pop(j), dt, k)
                    continue
            t.update(None, dt, k)

        for p in unused:                                  # unassociated detections start tracks
            tracks.append(AlphaBetaTracker(p, 0.0, alpha, beta, gate_deg, next_id))
            tracks[-1].history = [(k, p)]
            next_id += 1

    return [t for t in tracks if len(t.history) > 2], detections


def btr_image(scans, normalise='per_scan'):
    """Bearing-time record in dB: (n_scans, n_azimuth), normalised per scan or globally."""
    S = np.asarray(scans, dtype=float)
    if normalise == 'per_scan':
        S = S / (S.max(axis=1, keepdims=True) + 1e-20)
    else:
        S = S / (S.max() + 1e-20)
    return 10 * np.log10(S + 1e-20)
