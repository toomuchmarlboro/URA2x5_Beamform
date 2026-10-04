"""Config and recording I/O shared by both notebooks (Track B, G9).

`load_config` is used by both notebooks. The recording path below is the G9 scaffolding:
a loader, a channel map (element -> ADC channel, polarity) and a per-channel complex
calibration.

NOTE: the UATR_TDM output format is still TBD, so `load_recording` supports WAV and raw
interleaved PCM and is validated here only by a synthetic round-trip. It has NOT been run
against a real HACAR recording — that validation is still outstanding.
"""
import numpy as np
import yaml

__all__ = ["load_config", "load_recording", "load_channel_map", "apply_channel_map",
           "save_calibration", "load_calibration", "apply_calibration"]


def load_config(path="config.yaml"):
    """Load the shared parameter file (config.yaml) as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_recording(path, n_channels=None, dtype=None, fs=None):
    """
    Load a multichannel recording as (x, fs, meta) with x of shape (n_channels, n_samples),
    float64, scaled to +/-1 full scale.

    * `.wav` : channel count, rate and sample format come from the file.
    * raw    : `n_channels`, `dtype` and `fs` must all be given (interleaved frames).
    """
    path = str(path)
    if path.lower().endswith(".wav"):
        from scipy.io import wavfile
        fs_file, data = wavfile.read(path)
        if data.ndim == 1:
            data = data[:, None]
        x = data.T.astype(np.float64)
        if np.issubdtype(data.dtype, np.integer):
            x = x / float(np.iinfo(data.dtype).max)
        return x, float(fs_file), {"source": path, "format": "wav", "dtype": str(data.dtype)}

    if n_channels is None or dtype is None or fs is None:
        raise ValueError("raw recordings need n_channels, dtype and fs")
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    n_frames = raw.size // n_channels
    x = raw[: n_frames * n_channels].reshape(n_frames, n_channels).T.astype(np.float64)
    if np.issubdtype(np.dtype(dtype), np.integer):
        x = x / float(np.iinfo(np.dtype(dtype)).max)
    return x, float(fs), {"source": path, "format": "raw", "dtype": str(dtype)}


def load_channel_map(path):
    """
    Read channel_map.yaml:

        elements:
          - {element: 1, channel: 0, polarity: +1}
          ...

    Returns (channel_index[n_elements], polarity[n_elements]) ordered by element number.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = sorted(cfg["elements"], key=lambda r: r["element"])
    ch = np.array([r["channel"] for r in rows], dtype=int)
    pol = np.array([r.get("polarity", 1) for r in rows], dtype=float)
    return ch, pol


def apply_channel_map(x, channel_index, polarity):
    """Reorder raw ADC channels into element order and correct wiring polarity."""
    if np.max(channel_index) >= x.shape[0]:
        raise ValueError(f"channel map wants channel {np.max(channel_index)} "
                         f"but the recording has {x.shape[0]}")
    return x[channel_index, :] * polarity[:, None]


def save_calibration(path, freqs_hz, sensitivity):
    """Store complex per-channel sensitivity vs frequency; `sensitivity` is (n_ch, n_freq)."""
    np.savez(path, freqs_hz=np.asarray(freqs_hz, dtype=float),
             sensitivity=np.asarray(sensitivity, dtype=complex))


def load_calibration(path):
    d = np.load(path)
    return d["freqs_hz"], d["sensitivity"]


def apply_calibration(X, bin_freqs, cal_freqs, cal_sensitivity):
    """
    Equalise an STFT `X` of shape (n_ch, n_bins, n_frames) by the inverse of each channel's
    complex sensitivity, interpolated onto the FFT bin frequencies. Magnitude and phase are
    interpolated separately so that unwrapped phase does not get mangled by interpolating
    across the complex plane.
    """
    mag = np.stack([np.interp(bin_freqs, cal_freqs, np.abs(c)) for c in cal_sensitivity])
    pha = np.stack([np.interp(bin_freqs, cal_freqs, np.unwrap(np.angle(c)))
                    for c in cal_sensitivity])
    H = mag * np.exp(1j * pha)
    return X / H[:, :, None]
