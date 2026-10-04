"""Time-domain signal chain: 96 kHz -> decimate -> STFT -> per-bin beamform (Track B, G8).

Track A works directly on synthetic per-bin narrowband snapshots: there is no sample rate,
no anti-alias filter, no decimation and no FFT. The hardware chain is
96 kHz -> filter -> decimate to 24 kHz -> STFT -> beamform, and SYS-014 (alias rejection)
can only be tested once that chain exists.

The STFT here is written out explicitly (Hann window + rfft) rather than taken from
scipy.signal, so the frame/bin convention is fixed and does not shift under SciPy API
changes.
"""
import numpy as np
from scipy.signal import decimate

from .geometry import unit_vector

__all__ = ["bandlimited_noise", "fractional_delay", "simulate_timeseries",
           "decimate_chain", "stft", "stft_covariances"]


def bandlimited_noise(n, fs, band, rng):
    """Real Gaussian noise with unit total power, flat inside `band` and zero outside."""
    X = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    X[(f < band[0]) | (f > band[1])] = 0.0
    x = np.fft.irfft(X, n=n)
    rms = np.sqrt(np.mean(x ** 2))
    return x / rms if rms > 0 else x


def fractional_delay(x, delay_samples):
    """
    Delay a real signal by an arbitrary (possibly negative, possibly fractional) number of
    samples, via an exact phase ramp in the rfft domain. Circular, which is harmless for the
    stationary noise signals used here and exact for the per-bin phases the beamformer sees.
    """
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_samples), n=n, axis=-1)


def simulate_timeseries(sources, fs, duration_s, positions, c, rng, band=(100.0, 1500.0),
                         noise_power=1.0, extra_signals=None):
    """
    Broadband multichannel time series with *true propagation delays* per element.

    `sources`: list of (elevation_deg, azimuth_deg, snr_db) — snr_db is per-element signal
    power relative to `noise_power`. Each source gets an independent band-limited waveform,
    delayed per element by +(k_hat . r_n)/c. That sign is what makes the resulting per-bin
    phases equal `exp(-j 2 pi f (k.r)/c)`, i.e. the steering-vector convention used
    throughout `ura2x5.beamform`; flipping it reflects every bearing through the array centre
    (an apparent 180 degree error on a ring).

    `extra_signals`: optional list of (elevation_deg, azimuth_deg, waveform) to inject a
    specific waveform (e.g. the amplitude-modulated ship signature used by DEMON).

    Returns x of shape (n_elements, n_samples).
    """
    n = int(round(fs * duration_s))
    nel = positions.shape[0]
    x = rng.normal(scale=np.sqrt(noise_power), size=(nel, n))

    def inject(el, az, wave):
        tau = (positions @ unit_vector(el, az)) / c           # seconds; see docstring on sign
        for i in range(nel):
            x[i] += fractional_delay(wave, tau[i] * fs)

    for el, az, snr_db in sources:
        wave = bandlimited_noise(n, fs, band, rng) * np.sqrt(noise_power * 10 ** (snr_db / 10))
        inject(el, az, wave)
    for el, az, wave in (extra_signals or []):
        inject(el, az, np.asarray(wave, dtype=float))
    return x


def decimate_chain(x, q, fs):
    """Anti-alias filter + decimate by `q` (the UATR_TDM 96 kHz -> 24 kHz step for q = 4)."""
    y = decimate(x, q, ftype='fir', zero_phase=True, axis=-1)
    return y, fs / q


def stft(x, nperseg, overlap=0.5, fs=1.0, window='hann'):
    """
    Short-time Fourier transform of (n_channels, n_samples).
    Returns (X, freqs) with X of shape (n_channels, n_bins, n_frames).
    """
    step = int(nperseg * (1.0 - overlap))
    w = np.hanning(nperseg) if window == 'hann' else np.ones(nperseg)
    n = x.shape[-1]
    starts = range(0, n - nperseg + 1, step)
    frames = np.stack([x[..., s:s + nperseg] * w for s in starts], axis=-1)   # (ch, nperseg, frames)
    X = np.fft.rfft(frames, axis=-2)
    return X, np.fft.rfftfreq(nperseg, d=1.0 / fs)


def stft_covariances(X, freqs, band):
    """
    Per-bin sample covariances from an STFT, averaging over frames (the snapshots).
    Returns (bin_freqs, R) with R of shape (n_bins_in_band, n_channels, n_channels).
    """
    keep = np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))
    Xs = X[:, keep, :]                                     # (ch, bins, frames)
    n_frames = Xs.shape[-1]
    R = np.einsum('cbf,dbf->bcd', Xs, Xs.conj()) / n_frames
    return freqs[keep], R
