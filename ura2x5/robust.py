"""Array imperfections and the robustness Monte Carlo (Track B, G5, G6, G7, G15).

Track A assumes perfect channels, exact element positions, a perfectly flat ring and a
processor whose assumed sound speed equals the truth. This module perturbs the *measured*
array while the processor keeps using the nominal model, which is what actually happens in
hardware, and reports the resulting pointing error, gain loss, peak sidelobe level and
bearing error. These are the MATLAB T10 cases, which were never run.
"""
import numpy as np

from .beamform import (
    steering_vector, das_weights, to_db_norm, lobe_metrics, beamformer_output,
)
from .geometry import wrap180

__all__ = ["rotation_matrix", "perturb_array", "measured_steering_vector",
           "robustness_trial", "robustness_mc", "tolerance_sweep",
           "heading_random_walk", "compass_reading"]


def heading_random_walk(n, dt, rate_std_dps, rng, initial_deg=0.0):
    """
    Array heading (deg) over `n` updates, as a random walk in *rotation rate*: a dipping array
    on a cable yaws slowly and smoothly rather than jumping. `rate_std_dps` is the std of the
    per-step rate change in deg/s.
    """
    rate = np.cumsum(rng.normal(scale=rate_std_dps, size=n)) * dt
    return initial_deg + np.cumsum(rate) * dt


def compass_reading(true_heading_deg, rng, noise_std_deg=0.0, bias_deg=0.0):
    """What the heading sensor reports: truth plus a fixed installation bias and random noise."""
    h = np.asarray(true_heading_deg, dtype=float)
    return h + bias_deg + rng.normal(scale=noise_std_deg, size=h.shape)


def rotation_matrix(tilt_deg, tilt_az_deg=0.0):
    """
    Rotate the array plane by `tilt_deg` about a horizontal axis whose azimuth is
    `tilt_az_deg` — i.e. the array tips over in that direction (G7 tilt, G5 case).
    """
    t = np.radians(tilt_deg)
    psi = np.radians(tilt_az_deg)
    axis = np.array([np.cos(psi), np.sin(psi), 0.0])          # horizontal rotation axis
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def perturb_array(positions, rng, sigma_gain_db=0.0, sigma_phase_deg=0.0, sigma_pos_m=0.0,
                   dead=(), tilt_deg=0.0, tilt_az_deg=0.0,
                   adc_skew_samples=0.0, adc_group_size=4, fs_hz=24000.0):
    """
    Return (perturbed_positions, complex_channel_gain) for one random realisation.

    * `sigma_gain_db`, `sigma_phase_deg` : per-channel amplitude / phase error (Gaussian)
    * `sigma_pos_m`                      : per-element 3-D position error (Gaussian)
    * `dead`                             : indices of failed channels (gain set to 0)
    * `tilt_deg`, `tilt_az_deg`          : rigid out-of-plane rotation of the whole array
    * `adc_skew_samples`                 : worst-case sample slip between ADC groups. Nine
      channels need three 4-channel ADAU1978 parts, so a slip applies to a whole group of
      `adc_group_size` channels at once. Stored as a delay in seconds (frequency-dependent
      phase), applied in `measured_steering_vector`.

    The ADC skew is returned as a third element only when non-zero is requested; callers use
    `measured_steering_vector`, which takes it explicitly.
    """
    n = positions.shape[0]
    pos_p = positions.astype(float).copy()
    if sigma_pos_m > 0:
        pos_p = pos_p + rng.normal(scale=sigma_pos_m, size=pos_p.shape)
    if tilt_deg != 0.0:
        pos_p = pos_p @ rotation_matrix(tilt_deg, tilt_az_deg).T

    gain_db = rng.normal(scale=sigma_gain_db, size=n) if sigma_gain_db > 0 else np.zeros(n)
    phase_deg = rng.normal(scale=sigma_phase_deg, size=n) if sigma_phase_deg > 0 else np.zeros(n)
    g = 10 ** (gain_db / 20) * np.exp(1j * np.radians(phase_deg))
    if len(dead):
        g[np.asarray(dead, dtype=int)] = 0.0

    if adc_skew_samples != 0.0:
        groups = np.arange(n) // adc_group_size
        # one group per ADC part; give each a uniformly random slip in +/- adc_skew_samples
        slips = rng.uniform(-adc_skew_samples, adc_skew_samples, size=groups.max() + 1)
        tau = slips[groups] / fs_hz
    else:
        tau = np.zeros(n)
    return pos_p, g, tau


def measured_steering_vector(theta_deg, phi_deg, freq, pos_p, g, tau, c):
    """What the hardware actually delivers: perturbed geometry, channel gain, and timing skew."""
    a = steering_vector(theta_deg, phi_deg, freq, positions=pos_p, c=c)
    skew = np.exp(-1j * 2 * np.pi * freq * tau)
    if a.ndim == 1:
        return g * skew * a
    return (g * skew)[:, None] * a


def robustness_trial(freq, steer_az, pos_nominal, pos_p, g, tau, c_assumed, c_true,
                      phi_grid, theta_deg=0.0):
    """
    One realisation: the processor builds weights from the nominal array and c_assumed; the
    signal arrives through the perturbed array in a medium with c_true.

    Returns pointing error (deg), gain loss (dB, positive = loss) and PSL (dB).
    """
    W = das_weights(theta_deg, phi_grid, freq, positions=pos_nominal, c=c_assumed)
    a_meas = measured_steering_vector(theta_deg, steer_az, freq, pos_p, g, tau, c_true)
    P = np.abs(np.conj(W).T @ a_meas) ** 2

    from .bearing import refine_peak_deg
    peak_az = refine_peak_deg(phi_grid, P)
    pdb = to_db_norm(P)
    m = lobe_metrics(phi_grid, pdb, steer_az)

    w_look = das_weights(theta_deg, steer_az, freq, positions=pos_nominal, c=c_assumed)
    gain = abs(beamformer_output(w_look, a_meas))
    return dict(
        pointing_err_deg=float(wrap180(peak_az - steer_az)),
        gain_loss_db=float(-20 * np.log10(gain + 1e-15)),
        psl_db=float(m['psl']),
    )


def robustness_mc(freq, steer_az, positions, c_assumed=1500.0, c_true=None, n_trials=200,
                   seed=0, phi_grid=None, **perturb_kw):
    """
    Monte Carlo over the perturbations in `perturb_array`. Returns per-trial arrays plus
    summary statistics (RMS pointing error, mean/worst gain loss, worst PSL).
    """
    if phi_grid is None:
        phi_grid = np.arange(0.0, 360.0, 0.2)
    if c_true is None:
        c_true = c_assumed
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_trials):
        pos_p, g, tau = perturb_array(positions, rng, **perturb_kw)
        rows.append(robustness_trial(freq, steer_az, positions, pos_p, g, tau,
                                     c_assumed, c_true, phi_grid))
    pe = np.array([r['pointing_err_deg'] for r in rows])
    gl = np.array([r['gain_loss_db'] for r in rows])
    psl = np.array([r['psl_db'] for r in rows])
    return dict(
        pointing_err_deg=pe, gain_loss_db=gl, psl_db=psl,
        pointing_rms_deg=float(np.sqrt(np.mean(pe ** 2))),
        pointing_p95_deg=float(np.percentile(np.abs(pe), 95)),
        gain_loss_mean_db=float(np.mean(gl)), gain_loss_p95_db=float(np.percentile(gl, 95)),
        psl_worst_db=float(np.nanmax(psl)), psl_mean_db=float(np.nanmean(psl)),
    )


def tolerance_sweep(param, values, freq, steer_az, positions, metric, limit,
                     n_trials=200, seed=0, c_assumed=1500.0, c_true=None, **fixed_kw):
    """
    Sweep one perturbation parameter and return the largest value whose `metric` still meets
    `limit`, for the FAT tolerance table. `metric` is a key of the `robustness_mc` summary;
    'psl_worst_db' is met when <= limit, the error metrics when <= limit as well.

    Returns (rows, largest_passing_value). `None` means even the smallest value tested fails.
    """
    rows, passing = [], None
    for v in values:
        kw = dict(fixed_kw)
        kw[param] = v
        s = robustness_mc(freq, steer_az, positions, c_assumed=c_assumed, c_true=c_true,
                          n_trials=n_trials, seed=seed, **kw)
        ok = s[metric] <= limit
        rows.append((v, s[metric], ok))
        if ok:
            passing = v
    return rows, passing
