"""LOFAR and DEMON classification displays (Track B, G10).

Both are named in the CONOPS and the MATLAB TESTBENCH config but never implemented in
Track A. LOFAR resolves narrowband tonals (machinery, blade lines) on a beam output; DEMON
recovers the *modulation* rate of broadband cavitation noise, which gives shaft rate and
blade count even when the tonals themselves are buried.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

__all__ = ["lofargram", "split_window_normalise", "demon_spectrum", "modulated_cavitation"]


def lofargram(x, fs, nperseg=4096, overlap=0.75, band=None):
    """
    Narrowband spectrogram of a single beam output (power, dB), for tonal analysis.
    Returns (freqs, times, S_db).
    """
    step = int(nperseg * (1.0 - overlap))
    w = np.hanning(nperseg)
    starts = np.arange(0, len(x) - nperseg + 1, step)
    frames = np.stack([x[s:s + nperseg] * w for s in starts], axis=0)
    S = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    if band is not None:
        keep = (freqs >= band[0]) & (freqs <= band[1])
        freqs, S = freqs[keep], S[:, keep]
    times = starts / fs
    return freqs, times, 10 * np.log10(S.T + 1e-20)


def split_window_normalise(S_db, n_train=24, n_guard=4):
    """
    Split-window (two-pass mean) background normaliser along the frequency axis of a
    LOFARgram, the standard display normaliser: it flattens the broadband shape so tonals
    stand out as excess over the local background rather than as absolute level.

    `S_db` is (n_freq, n_time). Returns the same shape, in dB above local background.
    """
    S_lin = 10 ** (np.asarray(S_db) / 10)
    nf = S_lin.shape[0]
    half = n_train // 2
    bg = np.empty_like(S_lin)
    for i in range(nf):
        lo_a, lo_b = max(0, i - n_guard - half), max(0, i - n_guard)
        hi_a, hi_b = min(nf, i + n_guard + 1), min(nf, i + n_guard + 1 + half)
        train = np.concatenate([S_lin[lo_a:lo_b], S_lin[hi_a:hi_b]], axis=0)
        bg[i] = train.mean(axis=0) if train.size else S_lin[i]
    return 10 * np.log10(S_lin / (bg + 1e-20) + 1e-20)


def demon_spectrum(x, fs, band, demon_max_hz=50.0, nfft=None, detrend=True):
    """
    DEMON: band-pass the cavitation band, take the square-law envelope, then a low-rate
    spectrum of that envelope. Peaks appear at the shaft rate and the blade rate.

    Returns (mod_freqs, spectrum_db).
    """
    nyq = fs / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.999)
    sos = butter(4, [lo, hi], btype='band', output='sos')
    xb = sosfiltfilt(sos, x)

    env = np.abs(hilbert(xb)) ** 2                    # square-law envelope
    if detrend:
        env = env - env.mean()

    n = nfft or len(env)
    w = np.hanning(len(env))
    S = np.abs(np.fft.rfft(env * w, n=n)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    keep = f <= demon_max_hz
    S = S[keep]
    return f[keep], 10 * np.log10(S / (S.max() + 1e-20) + 1e-20)


def modulated_cavitation(n, fs, band, shaft_rate_hz, n_blades, mod_depth, rng,
                          blade_depth=None):
    """
    Broadband cavitation noise amplitude-modulated at the shaft rate and the blade rate —
    the signal DEMON is designed to recover. Returns a unit-RMS real waveform.
    """
    from .timeseries import bandlimited_noise

    carrier = bandlimited_noise(n, fs, band, rng)
    t = np.arange(n) / fs
    bpf = shaft_rate_hz * n_blades
    bd = mod_depth if blade_depth is None else blade_depth
    m = (1.0
         + mod_depth * np.sin(2 * np.pi * shaft_rate_hz * t)
         + bd * np.sin(2 * np.pi * bpf * t))
    y = carrier * m
    return y / np.sqrt(np.mean(y ** 2))
