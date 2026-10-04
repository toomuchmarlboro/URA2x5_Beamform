"""Bearing estimation and the Cramer-Rao bound (Track B, G4).

Track A estimates bearing by taking the global maximum of a 1-degree scan at a single SNR.
This module adds sub-grid peak refinement, a MUSIC estimator, the deterministic (conditional)
CRB for azimuth on an arbitrary array, and an RMSE-vs-SNR sweep that reports the threshold
SNR and outlier rate.
"""
import numpy as np

from .beamform import steering_vector, load_diag, spatial_spectrum
from .geometry import wrap180

__all__ = [
    "steering_derivative_az", "crb_azimuth_deg", "music_spectrum",
    "refine_peak_deg", "estimate_bearing_deg", "rmse_vs_snr",
]


def steering_derivative_az(theta_deg, phi_deg, freq, *, positions, c):
    """d a / d(phi in radians) for a single direction — the CRB's sensitivity term."""
    a = steering_vector(theta_deg, phi_deg, freq, positions=positions, c=c)
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    du_dphi = np.array([-np.cos(th) * np.sin(ph), np.cos(th) * np.cos(ph), 0.0])
    return a * (-1j * 2 * np.pi * freq / c * (positions @ du_dphi))


def crb_azimuth_deg(phi_deg, freq, snr_db, n_snap, theta_deg=0.0, *, positions, c):
    """
    Deterministic (conditional) CRB standard deviation on azimuth, in degrees, for one
    far-field source in spatially white noise:

        CRB(phi) = 1 / (2 K * SNR * Re{adot^H P_perp adot}),   P_perp = I - a a^H / N

    `snr_db` is per-element signal-to-noise ratio; `n_snap` is the number of snapshots.
    """
    a = steering_vector(theta_deg, phi_deg, freq, positions=positions, c=c)
    adot = steering_derivative_az(theta_deg, phi_deg, freq, positions=positions, c=c)
    n = len(a)
    P_perp = np.eye(n) - np.outer(a, a.conj()) / n
    fisher = 2.0 * n_snap * 10 ** (snr_db / 10) * np.real(np.conj(adot) @ P_perp @ adot)
    return np.degrees(np.sqrt(1.0 / fisher))


def music_spectrum(R, freq, n_sources, theta_deg=0.0, *, phi_grid, positions, c):
    """MUSIC pseudo-spectrum 1 / ||E_noise^H a||^2 from a sample covariance R."""
    evals, evecs = np.linalg.eigh(R)
    En = evecs[:, : R.shape[0] - n_sources]        # eigh returns ascending eigenvalues
    A = steering_vector(theta_deg, phi_grid, freq, positions=positions, c=c)
    proj = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
    return 1.0 / (proj + 1e-15)


def refine_peak_deg(phi_grid, P):
    """
    Peak azimuth with parabolic interpolation across the two neighbouring grid cells.
    Without this, bearing RMSE floors at the grid step / sqrt(12) and never reaches the CRB.
    Assumes a uniform grid wrapping over 360 degrees.
    """
    K = len(P)
    i = int(np.argmax(P))
    y0, y1, y2 = P[(i - 1) % K], P[i], P[(i + 1) % K]
    denom = y0 - 2 * y1 + y2
    delta = 0.0 if denom == 0 else np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)
    step = phi_grid[1] - phi_grid[0]
    return float((phi_grid[i] + delta * step) % 360.0)


def estimate_bearing_deg(method, R, freq, n_sources=1, theta_deg=0.0, loading=1e-2,
                          *, phi_grid, positions, c):
    """Single-target bearing estimate from a covariance, by 'das_td', 'mvdr' or 'music'."""
    if method == 'music':
        P = music_spectrum(R, freq, n_sources, theta_deg,
                           phi_grid=phi_grid, positions=positions, c=c)
    else:
        P = spatial_spectrum(method, R, freq, theta_deg, phi_grid, loading=loading,
                             positions=positions, c=c)
    return refine_peak_deg(phi_grid, P)


def rmse_vs_snr(snr_db_list, freq, n_snap, n_trials, phi_true, methods=('das_td', 'mvdr', 'music'),
                 coherence_fn=None, hpbw_deg=None, seed=0, *, phi_grid, positions, c):
    """
    Bearing RMSE vs per-element SNR by Monte Carlo.

    `coherence_fn(freq)` returns the noise spatial-coherence matrix (None = spatially white).
    Errors larger than hpbw_deg/2 are counted as outliers and reported separately, since
    they come from picking the wrong peak rather than from estimator variance.

    Returns {method: {'rmse': [...], 'rmse_inlier': [...], 'outlier_rate': [...]}}.
    """
    from .noise import simulate_snapshots_in_field
    from .beamform import sample_covariance

    Gamma = None if coherence_fn is None else coherence_fn(freq)
    out = {m: {'rmse': [], 'rmse_inlier': [], 'outlier_rate': []} for m in methods}
    gate = None if hpbw_deg is None else 0.5 * hpbw_deg

    for snr_db in snr_db_list:
        rng = np.random.default_rng(seed)       # same noise realisations for every method
        errs = {m: [] for m in methods}
        for _ in range(n_trials):
            x = simulate_snapshots_in_field([(0.0, phi_true, snr_db)], freq, n_snap, rng,
                                            coherence=Gamma, positions=positions, c=c)
            R = sample_covariance(x)
            for m in methods:
                phi_hat = estimate_bearing_deg(m, R, freq, phi_grid=phi_grid,
                                               positions=positions, c=c)
                errs[m].append(wrap180(phi_hat - phi_true))
        for m in methods:
            e = np.asarray(errs[m], dtype=float)
            out[m]['rmse'].append(float(np.sqrt(np.mean(e ** 2))))
            if gate is None:
                out[m]['rmse_inlier'].append(float(np.sqrt(np.mean(e ** 2))))
                out[m]['outlier_rate'].append(0.0)
            else:
                keep = np.abs(e) <= gate
                out[m]['outlier_rate'].append(float(1.0 - keep.mean()))
                out[m]['rmse_inlier'].append(
                    float(np.sqrt(np.mean(e[keep] ** 2))) if keep.any() else float('nan'))
    return out
