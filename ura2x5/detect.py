"""Detection statistics and the sonar equation (Track B, G3, G12).

Track A has no detector at all: no threshold, no Pd/Pfa, no ROC, and no check that the
DT = -13 dB in the CONOPS budget is achievable. This module implements the broadband energy
detector, sets its threshold by Monte Carlo and in closed form, measures the ROC, and closes
the loop against the textbook DT = 5*log10(d/WT).
"""
import numpy as np
from scipy.stats import chi2, norm

__all__ = [
    "energy_statistic", "threshold_chi2", "threshold_mc", "pd_chi2", "pd_mc",
    "detection_index", "dt_formula_db", "required_snr_db", "thorp_absorption_db_per_km",
    "transmission_loss_db", "detection_range_m",
]


def energy_statistic(y):
    """Mean power of a complex beam output over its last axis (the energy detector output)."""
    return np.mean(np.abs(y) ** 2, axis=-1)


def threshold_chi2(n_samp, pfa):
    """
    Detection threshold on the *normalised* energy statistic (mean power, noise power = 1).
    Exact for complex Gaussian noise: 2*n_samp*T ~ chi^2 with 2*n_samp degrees of freedom.
    """
    return chi2.ppf(1.0 - pfa, 2 * n_samp) / (2 * n_samp)


def _h0_statistics(n_samp, n_trials, rng, chunk_bytes=64 << 20):
    """
    H0 energy statistics, generated in chunks. A naive (n_trials, n_samp) array is not an
    option at the operating point: W*T = 10000 samples x 50000 trials would be ~8 GB.
    """
    per_trial = max(1, int(chunk_bytes // (n_samp * 16)))
    out = np.empty(n_trials)
    done = 0
    while done < n_trials:
        m = min(per_trial, n_trials - done)
        z = (rng.normal(size=(m, n_samp)) + 1j * rng.normal(size=(m, n_samp))) / np.sqrt(2)
        out[done:done + m] = energy_statistic(z)
        done += m
    return out


def threshold_mc(n_samp, pfa, n_trials, rng):
    """
    Same threshold estimated by sample-level Monte Carlo, as an independent check that the
    statistic really is chi-square distributed.

    Note on choosing (n_samp, pfa, n_trials): estimating a 1e-4 quantile from n_trials draws
    leaves only n_trials*1e-4 samples in the tail, so a meaningful check needs either a huge
    trial count or a larger pfa. Validate the distribution at a moderate n_samp and pfa, then
    use `threshold_chi2` at the operating point.
    """
    return float(np.quantile(_h0_statistics(n_samp, n_trials, rng), 1.0 - pfa))


def pd_chi2(n_samp, thr, snr_lin):
    """Probability of detection for post-beamforming SNR `snr_lin` (linear, not dB)."""
    return chi2.sf(2 * n_samp * thr / (1.0 + snr_lin), 2 * n_samp)


def pd_mc(n_samp, thr, snr_lin, n_trials, rng, chunk_bytes=64 << 20):
    """Monte Carlo Pd, as an independent check on `pd_chi2`. Chunked, like `threshold_mc`."""
    scale = np.sqrt((1.0 + snr_lin) / 2.0)
    per_trial = max(1, int(chunk_bytes // (n_samp * 16)))
    hits, done = 0, 0
    while done < n_trials:
        m = min(per_trial, n_trials - done)
        z = (rng.normal(size=(m, n_samp)) + 1j * rng.normal(size=(m, n_samp))) * scale
        hits += int(np.count_nonzero(energy_statistic(z) > thr))
        done += m
    return hits / n_trials


def detection_index(pd, pfa):
    """Detection index d = (d')^2 from the Gaussian ROC, d' = Phi^-1(1-Pfa) - Phi^-1(1-Pd)."""
    return float((norm.isf(pfa) - norm.isf(pd)) ** 2)


def dt_formula_db(pd, pfa, bandwidth_hz, integration_s):
    """
    Textbook detection threshold for an energy detector,
        DT = 5*log10(d / (W*T)),
    i.e. the post-beamforming SNR in a 1-Hz band required to reach (Pd, Pfa) after
    integrating bandwidth W for time T.
    """
    d = detection_index(pd, pfa)
    return 5.0 * np.log10(d / (bandwidth_hz * integration_s))


def required_snr_db(n_samp, pfa, pd_target, lo_db=-40.0, hi_db=20.0, tol=1e-3):
    """
    SNR (dB) at which the exact chi-square detector reaches `pd_target`, by bisection.
    This is the simulated DT to compare against `dt_formula_db`.
    """
    thr = threshold_chi2(n_samp, pfa)
    f = lambda s_db: pd_chi2(n_samp, thr, 10 ** (s_db / 10)) - pd_target
    lo, hi = lo_db, hi_db
    if f(lo) > 0 or f(hi) < 0:
        return float('nan')
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def thorp_absorption_db_per_km(f_hz):
    """Thorp absorption coefficient, dB/km, for f in Hz."""
    f = np.asarray(f_hz, dtype=float) / 1000.0        # kHz
    return (0.11 * f ** 2 / (1 + f ** 2) + 44.0 * f ** 2 / (4100.0 + f ** 2)
            + 2.75e-4 * f ** 2 + 0.003)


def transmission_loss_db(range_m, f_hz, spreading='spherical', water_depth_m=None,
                          extra_loss_db_per_km=0.0):
    """
    Transmission loss: geometric spreading + Thorp absorption + an optional lumped
    shallow-water term.

    `spreading`:
      'spherical'   : 20 log r — free field, valid only out to roughly the water depth
      'cylindrical' : 10 log r — fully waveguide-trapped
      'transition'  : 20 log r out to `water_depth_m`, then 10 log r beyond. This is the
                      standard shallow-water approximation and needs `water_depth_m`.

    `extra_loss_db_per_km` lumps bottom-interaction and mode-stripping loss, which pure
    spreading laws ignore entirely. Leaving it at 0 in shallow water **grossly overestimates**
    range: spreading + Thorp alone put nothing between the source and the receiver except
    geometry. Real numbers need a Bellhop (or equivalent) run — pass one through
    `detection_range_m(tl_fn=...)`.
    """
    r = np.maximum(np.asarray(range_m, dtype=float), 1.0)
    if spreading == 'spherical':
        geo = 20.0 * np.log10(r)
    elif spreading == 'cylindrical':
        geo = 10.0 * np.log10(r)
    elif spreading == 'transition':
        if water_depth_m is None:
            raise ValueError("'transition' spreading needs water_depth_m")
        h = max(float(water_depth_m), 1.0)
        geo = np.where(r <= h, 20.0 * np.log10(r),
                       20.0 * np.log10(h) + 10.0 * np.log10(np.maximum(r / h, 1.0)))
    else:
        raise ValueError(spreading)
    return geo + (thorp_absorption_db_per_km(f_hz) + extra_loss_db_per_km) * r / 1000.0


def detection_range_m(sl_db, nl_db, ag_db, dt_db, f_hz, spreading='spherical',
                       water_depth_m=None, extra_loss_db_per_km=0.0,
                       tl_fn=None, r_max_m=200000.0, n_grid=20000):
    """
    Largest range at which the passive sonar equation still closes:

        SE(r) = SL - TL(r) - (NL - AG) - DT >= 0

    All four level terms must be in **consistent units**. The usual choice, and the one used
    by the notebook, is band-integrated: `sl_db` and `nl_db` are levels in the processing
    band, and `dt_db` is the required in-band signal-to-noise *power ratio* — which is exactly
    what `dt_formula_db` and `required_snr_db` return. Do not add 10*log10(W) to `dt_db`:
    it is already a ratio, not a spectral density.

    `tl_fn(range_m, f_hz)` overrides the built-in model. Returns NaN when the equation never
    closes, and `inf`-free clipping at `r_max_m` when it always does.
    """
    r = np.linspace(1.0, r_max_m, n_grid)
    tl = (transmission_loss_db(r, f_hz, spreading, water_depth_m, extra_loss_db_per_km)
          if tl_fn is None else np.asarray(tl_fn(r, f_hz)))
    se = sl_db - tl - (nl_db - ag_db) - dt_db
    ok = se >= 0
    return float(r[ok][-1]) if ok.any() else float('nan')
