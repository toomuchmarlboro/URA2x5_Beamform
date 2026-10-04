"""Element response and receive-chain self-noise (Track B, G13).

Track A treats every element as isotropic, flat in frequency and noiseless. SYS-013 asks for
array self-noise below the sea-state-0 ambient level, which needs the hydrophone sensitivity,
the preamp noise and the ADC noise referred back to acoustic pressure.

All constants come from `config.yaml` -> `track_b.hardware`, where they are marked as
placeholders until the real component datasheet values are entered.
"""
import numpy as np

__all__ = ["element_sensitivity_db", "adc_noise_v_per_rthz", "self_noise_db_re_upa",
           "self_noise_margin_db"]


def element_sensitivity_db(f_hz, m0_db, f_ref_hz=1000.0, rolloff_db_per_decade=0.0,
                            hp_corner_hz=None):
    """
    Receive sensitivity, dB re 1 V/uPa, versus frequency: a flat value `m0_db` at `f_ref_hz`,
    an optional roll-off, and an optional single-pole high-pass corner (most piezo
    hydrophones lose sensitivity below resonance-set corners).
    """
    f = np.asarray(f_hz, dtype=float)
    m = m0_db + rolloff_db_per_decade * np.log10(f / f_ref_hz)
    if hp_corner_hz:
        m = m + 10 * np.log10(f ** 2 / (f ** 2 + hp_corner_hz ** 2))
    return m


def adc_noise_v_per_rthz(fullscale_vrms, dynamic_range_db, bandwidth_hz):
    """
    ADC noise voltage density from its datasheet dynamic range, assumed spread evenly over
    the converter's output bandwidth.
    """
    v_noise_rms = fullscale_vrms * 10 ** (-dynamic_range_db / 20)
    return v_noise_rms / np.sqrt(bandwidth_hz)


def self_noise_db_re_upa(f_hz, sensitivity_db, preamp_noise_v_per_rthz, preamp_gain_db,
                          adc_noise_v_per_rthz_val):
    """
    Equivalent input self-noise spectrum level, dB re 1 uPa^2/Hz.

    Preamp noise appears at the input directly; ADC noise is referred back through the preamp
    gain, so more gain suppresses the ADC's contribution. The two add in power, then divide by
    the hydrophone sensitivity to convert volts to pressure.
    """
    g = 10 ** (preamp_gain_db / 20)
    v_in = np.sqrt(preamp_noise_v_per_rthz ** 2 + (adc_noise_v_per_rthz_val / g) ** 2)
    m_v_per_upa = 10 ** (np.asarray(sensitivity_db, dtype=float) / 20)
    p_upa_per_rthz = v_in / m_v_per_upa
    return 20 * np.log10(p_upa_per_rthz)


def self_noise_margin_db(self_noise_db, ambient_db):
    """Margin of ambient over self-noise. Positive means ambient-limited (the design goal)."""
    return np.asarray(ambient_db) - np.asarray(self_noise_db)
