"""DAS / MVDR beamforming core for the 2x5 URA (URA2x5_Beamformed.ipynb and its realistic
revision).

This module is array-agnostic: `positions` and `c` are required keyword-only arguments rather
than defaults tied to a particular geometry, so the same code drives the 2x5 panel here and
the 9-element ring in the sibling BEAMFORM_LEN project. Only `geometry.py` and the taper
helper below know what shape the array is.
"""
import numpy as np

from .geometry import unit_vector, wrap180

__all__ = [
    "steering_vector", "quantize_phase", "das_weights", "beamformer_output",
    "beam_pattern", "to_db_norm", "lobe_metrics", "theta_null_filled_deg",
    "theta_null_line_deg", "hpbw_line_deg", "two_source_response", "das_gain_vs_frequency",
    "sample_covariance", "load_diag", "mvdr_weights", "simulate_snapshots",
    "spatial_spectrum", "make_taper", "count_peaks_above",
]


def steering_vector(theta_deg, phi_deg, freq, *, positions, c):
    """(N,) or (N, M) complex steering vector(s) for far-field direction(s)."""
    proj = positions @ unit_vector(theta_deg, phi_deg)
    return np.exp(-1j * 2 * np.pi * freq * proj / c)


def quantize_phase(w, bits):
    """Round weight phases to 2**bits levels (bits <= 0 means no quantization)."""
    if bits is None or bits <= 0:
        return w
    step = 2 * np.pi / 2**bits
    return np.abs(w) * np.exp(1j * step * np.round(np.angle(w) / step))


def das_weights(theta_deg, phi_deg, freq, amp=None, bits=0, *, positions, c):
    """
    DAS weights w = g * a / sum(g), with optional amplitude taper g (None = uniform)
    and optional phase quantization. Returns (N,) or (N, M).
    """
    a = quantize_phase(steering_vector(theta_deg, phi_deg, freq, positions=positions, c=c), bits)
    g = np.ones(positions.shape[0]) if amp is None else np.asarray(amp, dtype=float)
    g = g / g.sum()
    return a * (g if a.ndim == 1 else g[:, None])


def beamformer_output(weights, x):
    """Standard beamformer output y = w^H x (shared by DAS and MVDR)."""
    return np.conj(weights) @ x


def beam_pattern(steer_theta, steer_phi, freq, amp=None, bits=0, cut_theta=None,
                  *, phi_grid, positions, c):
    """|w^H a(phi)|^2 along an azimuth cut at elevation cut_theta (default = steer elevation)."""
    w = das_weights(steer_theta, steer_phi, freq, amp, bits, positions=positions, c=c)
    ct = steer_theta if cut_theta is None else cut_theta
    A = steering_vector(ct, phi_grid, freq, positions=positions, c=c)
    return np.abs(np.conj(w) @ A) ** 2


def to_db_norm(p):
    pdb = 10 * np.log10(np.asarray(p) + 1e-15)
    return pdb - pdb.max()


def lobe_metrics(ang, pdb, steer, circular=True, search=60.0):
    """
    Main-lobe peak, -3 dB width (HPBW), first-null half-width and peak sidelobe level (PSL)
    of a normalized dB cut. The main lobe is found by walking down from the peak to the
    first local minimum on each side, so no fixed exclusion zone is needed.
    """
    K, step = len(pdb), ang[1] - ang[0]
    dist = np.abs(wrap180(ang - steer)) if circular else np.abs(ang - steer)
    near = np.flatnonzero(dist <= search)
    ipk = near[np.argmax(pdb[near])]
    nxt = (lambda i, d: (i + d) % K) if circular else (lambda i, d: min(max(i + d, 0), K - 1))

    n3 = 0
    for d in (-1, 1):
        i, n = ipk, 0
        while n < K and nxt(i, d) != i and pdb[nxt(i, d)] >= pdb[ipk] - 3:
            i, n = nxt(i, d), n + 1
        n3 += n
    hpbw = np.nan if n3 >= K - 1 else (n3 + 1) * step

    lim = []
    for d in (-1, 1):
        i, n = ipk, 0
        while n < K and nxt(i, d) != i and pdb[nxt(i, d)] <= pdb[i]:
            i, n = nxt(i, d), n + 1
        lim.append(n)
    mask = np.zeros(K, bool)
    for d in range(-lim[0], lim[1] + 1):
        mask[nxt(ipk, d)] = True
    if mask.all() or sum(lim) >= K - 1:
        psl, null_half = np.nan, np.nan
    else:
        psl = pdb[~mask].max() - pdb[ipk]
        null_half = 0.5 * sum(lim) * step
    return dict(peak=ang[ipk], hpbw=hpbw, null_half=null_half, psl=psl)


def theta_null_filled_deg(freq, *, D, c):
    """Filled-aperture rule of thumb: theta_null ~= lambda/D (degrees)."""
    return np.degrees((c / np.asarray(freq, float)) / D)


def theta_null_line_deg(freq, *, n, d, c, steer_deg=0.0, order=1):
    """
    First-null half-width of a uniform line array of `n` elements spaced `d`, measured from
    the steer direction.

    The array factor is sin(N psi/2) / (N sin(psi/2)) with psi = kd(sin th - sin th_s), so
    nulls fall at sin(th) - sin(th_s) = m*lambda/(N d). Note the **N d**, not (N-1) d.

    Returns NaN where the null falls outside visible space — which is exactly what happens
    when the aperture is too short at low frequency, or when steering too close to endfire.
    """
    lam = c / np.asarray(freq, dtype=float)
    s = np.sin(np.radians(steer_deg)) + order * lam / (n * d)
    return np.where(np.abs(s) <= 1.0,
                    np.degrees(np.arcsin(np.clip(s, -1, 1))) - steer_deg,
                    np.nan)


def hpbw_line_deg(freq, *, n, d, c, steer_deg=0.0):
    """
    Half-power beamwidth of a uniform line array, including the 1/cos(steer) broadening that
    a planar array suffers and a ring does not: HPBW ~= 0.886 * lambda / (N d cos(theta_s)).

    NaN past the point where the approximation breaks down near endfire.
    """
    lam = c / np.asarray(freq, dtype=float)
    cs = np.cos(np.radians(steer_deg))
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.degrees(0.886 * lam / (n * d * cs))
    return np.where((cs > 0.05) & (w < 180.0), w, np.nan)


def two_source_response(theta_el, phi1, phi2, freq, snr_db=None, seed=0,
                         *, phi_grid, positions, c):
    """DAS azimuth scan of two equal, in-phase narrowband sources (optional white noise)."""
    rng = np.random.default_rng(seed)
    x = (steering_vector(theta_el, phi1, freq, positions=positions, c=c)
         + steering_vector(theta_el, phi2, freq, positions=positions, c=c))
    if snr_db is not None:
        npow = 10 ** (-snr_db / 10)
        x = x + (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * np.sqrt(npow / 2)
    W = das_weights(theta_el, phi_grid, freq, positions=positions, c=c)
    return phi_grid, np.abs(np.conj(W).T @ x) ** 2


def das_gain_vs_frequency(theta_el, phi, f_design, freq_range, bits=0, *, positions, c):
    """|w^H a(f)| with weights computed ONCE at f_design (phase-shift beamformer)."""
    w = das_weights(theta_el, phi, f_design, bits=bits, positions=positions, c=c)
    return np.array([
        abs(beamformer_output(w, steering_vector(theta_el, phi, f, positions=positions, c=c)))
        for f in freq_range
    ])


def sample_covariance(x):
    """x: (N, K) snapshots -> (N, N) sample covariance."""
    return x @ x.conj().T / x.shape[1]


def load_diag(R, loading=1e-2):
    Nn = R.shape[0]
    return R + loading * np.trace(R).real / Nn * np.eye(Nn)


def mvdr_weights(theta_el, phi, freq, R, loading=1e-2, *, positions, c):
    """MVDR weights steered to (theta_el, phi); returns (N,) or (N, M) for an array of phi."""
    a = steering_vector(theta_el, phi, freq, positions=positions, c=c)
    Ria = np.linalg.solve(load_diag(R, loading), a)
    return Ria / np.sum(np.conj(a) * Ria, axis=0)


def simulate_snapshots(sources, freq, n_snap, rng, *, positions, c):
    """
    sources: list of (theta_el, phi, element_SNR_dB). Unit-power white noise per element.
    Returns x (N, K) of narrowband complex snapshots.
    """
    Nn = positions.shape[0]
    x = (rng.normal(size=(Nn, n_snap)) + 1j * rng.normal(size=(Nn, n_snap))) / np.sqrt(2)
    for th, ph, snr in sources:
        s = (rng.normal(size=n_snap) + 1j * rng.normal(size=n_snap)) * np.sqrt(10 ** (snr / 10) / 2)
        x += np.outer(steering_vector(th, ph, freq, positions=positions, c=c), s)
    return x


def spatial_spectrum(method, R, freq, theta_el=0.0, phi_grid=None, f_fixed=None, bits=0,
                      loading=1e-2, *, positions, c):
    """
    Scanned power vs azimuth from a covariance R.
      'das_td' : DAS re-steered at this frequency (true delay / per-bin)
      'das_ps' : DAS with phase-shift weights fixed at f_fixed (optionally quantized)
      'mvdr'   : Capon spectrum 1 / (a^H R^-1 a)
    """
    Nn = positions.shape[0]
    if method == 'das_ps':
        W = das_weights(theta_el, phi_grid, f_fixed, bits=bits, positions=positions, c=c)
        return np.real(np.sum(np.conj(W) * (R @ W), axis=0))
    A = steering_vector(theta_el, phi_grid, freq, positions=positions, c=c)
    if method == 'das_td':
        return np.real(np.sum(np.conj(A) * (R @ A), axis=0)) / Nn**2
    if method == 'mvdr':
        return 1.0 / np.real(np.sum(np.conj(A) * np.linalg.solve(load_diag(R, loading), A), axis=0))
    raise ValueError(method)


def make_taper(taper_type, *, positions, axis=1, cheb_sidelobe_db=25.0):
    """
    Per-element amplitude weights for a **separable** taper on a rectangular array.

    Unlike a ring — where the aperture edge depends on look direction and conventional tapers
    actively hurt — a URA has fixed physical edges, so a window applied along a coordinate
    axis does what the textbook says: lower sidelobes, wider main lobe, some gain traded away.

    `axis` selects the coordinate the window runs along (1 = Y, the 5-element long axis;
    2 = Z, the 2-element stack). Elements sharing a coordinate get the same weight, so the
    result is a true separable taper rather than a projection.
    """
    from scipy.signal.windows import chebwin

    coord = np.asarray(positions)[:, axis]
    uniq = np.unique(np.round(coord, 9))
    n = len(uniq)

    if taper_type == 'uniform':
        w_axis = np.ones(n)
    elif taper_type == 'hamming':
        w_axis = np.hamming(n)
    elif taper_type == 'hann':
        # np.hanning zeroes its endpoints, which would switch off the outer elements and
        # throw away aperture; use the symmetric interior form instead.
        w_axis = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 1) / (n + 1))
    elif taper_type == 'chebyshev':
        w_axis = chebwin(n, at=cheb_sidelobe_db)
    else:
        raise ValueError(taper_type)

    lookup = {u: w for u, w in zip(uniq, w_axis)}
    return np.array([lookup[round(v, 9)] for v in coord])


def count_peaks_above(pdb, thr=-6.0):
    """Number of circular local maxima above thr dB."""
    pk = (pdb > np.roll(pdb, 1)) & (pdb >= np.roll(pdb, -1)) & (pdb > thr)
    return int(pk.sum())
