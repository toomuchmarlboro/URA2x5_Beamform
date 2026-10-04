"""Noise and signal models.

`ship_noise_model` is the Track A (ideal baseline) radiated-noise signature, moved here
unchanged from UCA9_Beamformed.ipynb. `noise_coherence` / `colored_noise_snapshots` /
`array_gain` are Track B (G1): spatially correlated ambient ocean noise, replacing the
spatially-white assumption used everywhere in Track A.
"""
import numpy as np
from scipy.special import j0

__all__ = [
    "ship_noise_model", "noise_coherence", "colored_noise_snapshots", "array_gain",
    "simulate_snapshots_in_field", "wenz_nl_db",
]


def ship_noise_model(shaft_rate_hz=2.0, n_blades=10, include_broadband=True, *, band):
    """List of (freq_hz, relative_amplitude) for a simple freight-ship signature."""
    comps = []
    bpf = shaft_rate_hz * n_blades
    for h in range(1, 8):
        comps.append((bpf * h, 1.0 / h))
    for h in range(1, 5):
        comps.append((50.0 * h, 0.4 / h))
    if include_broadband:
        for f in np.logspace(np.log10(band[0]), np.log10(band[1]), 25):
            comps.append((float(f), 0.08))
    return comps


def noise_coherence(model, f, d, c):
    """
    Spatial coherence Gamma(d) of an isotropic ambient-noise field at frequency f, for a
    pairwise-distance matrix d (e.g. from geometry.ring_derived). Unit diagonal by construction.

      'white'       : Gamma = I (spatially uncorrelated; the Track A assumption everywhere)
      'spherical'   : Gamma = sinc(k d) = sin(kd)/(kd)   (noise isotropic over the full sphere)
      'cylindrical' : Gamma = J0(k d)                     (noise isotropic in azimuth only,
                                                            near-horizontal arrivals -- the
                                                            usual shallow-water approximation)
      'mixed'       : equal blend of 'spherical' and 'cylindrical'
    """
    k = 2 * np.pi * f / c
    if model == 'white':
        return np.eye(d.shape[0])
    if model == 'spherical':
        return np.sinc(k * d / np.pi)          # np.sinc(x) = sin(pi x)/(pi x)
    if model == 'cylindrical':
        return j0(k * d)
    if model == 'mixed':
        return 0.5 * (noise_coherence('spherical', f, d, c) + noise_coherence('cylindrical', f, d, c))
    raise ValueError(model)


def colored_noise_snapshots(coherence, n_snap, rng):
    """
    (N, n_snap) complex snapshots of unit-power-per-element noise with the given spatial
    coherence matrix (N, N). coherence must be real, symmetric and positive semi-definite
    (true for all models in `noise_coherence`).
    """
    Nn = coherence.shape[0]
    L = np.linalg.cholesky(coherence)
    z = (rng.normal(size=(Nn, n_snap)) + 1j * rng.normal(size=(Nn, n_snap))) / np.sqrt(2)
    return L @ z


def simulate_snapshots_in_field(sources, freq, n_snap, rng, coherence=None, *, positions, c):
    """
    Narrowband snapshots for a list of (elevation_deg, azimuth_deg, element_SNR_dB) sources in
    a noise field of unit per-element power. `coherence=None` gives the Track A spatially
    white field; otherwise pass a coherence matrix from `noise_coherence` (G1).

    This generalises `beamform.simulate_snapshots`, which is white-noise only.
    """
    from .beamform import steering_vector

    Nn = positions.shape[0]
    if coherence is None:
        x = (rng.normal(size=(Nn, n_snap)) + 1j * rng.normal(size=(Nn, n_snap))) / np.sqrt(2)
    else:
        x = colored_noise_snapshots(coherence, n_snap, rng)
    for th, ph, snr in sources:
        s = (rng.normal(size=n_snap) + 1j * rng.normal(size=n_snap)) * np.sqrt(10 ** (snr / 10) / 2)
        x = x + np.outer(steering_vector(th, ph, freq, positions=positions, c=c), s)
    return x


def wenz_nl_db(f_hz, wind_speed_ms=0.0, shipping=0.5):
    """
    Wenz/Coates ambient-noise spectrum level, dB re 1 uPa^2/Hz, as the power sum of the
    turbulence, shipping, wind and thermal terms. `wind_speed_ms = 0` approximates sea state 0.

    Classic approximations, f in kHz:
      turbulence : 17  - 30*log10(f)
      shipping   : 40  + 20*(shipping - 0.5) + 26*log10(f) - 60*log10(f + 0.03)
      wind       : 50  + 7.5*sqrt(w)         + 20*log10(f) - 40*log10(f + 0.4)
      thermal    : -15 + 20*log10(f)
    """
    f = np.asarray(f_hz, dtype=float) / 1000.0
    turb = 17.0 - 30.0 * np.log10(f)
    ship = 40.0 + 20.0 * (shipping - 0.5) + 26.0 * np.log10(f) - 60.0 * np.log10(f + 0.03)
    wind = 50.0 + 7.5 * np.sqrt(wind_speed_ms) + 20.0 * np.log10(f) - 40.0 * np.log10(f + 0.4)
    therm = -15.0 + 20.0 * np.log10(f)
    return 10.0 * np.log10(sum(10 ** (t / 10) for t in (turb, ship, wind, therm)))


def array_gain(weights, coherence):
    """
    Array gain (linear) of beamformer `weights` against a noise field with the given spatial
    coherence, for unit per-element noise power and unit signal gain (w^H a = 1 at the look
    direction). AG = 1 / (w^H Gamma w).
    """
    denom = np.real(np.conj(weights) @ coherence @ weights)
    return 1.0 / denom
