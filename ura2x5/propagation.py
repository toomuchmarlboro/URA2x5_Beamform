"""Shallow-water multipath and target kinematics (Track B, G14, G12).

Track A puts every arrival at elevation 0 with a single path. In shallow water each target
also arrives via the surface and the bottom at non-zero elevation, and a horizontal ring
responds identically to +el and -el, so those copies are indistinguishable from the direct
path's elevation response.
"""
import numpy as np

__all__ = ["image_source_arrivals", "bearing_rate_deg_per_s", "max_integration_time_s"]


def image_source_arrivals(range_m, src_depth_m, array_depth_m, water_depth_m,
                           surface_coeff=-1.0, bottom_coeff=0.7, n_bottom=1):
    """
    Direct, surface-reflected and bottom-reflected arrivals for one target, via image sources.

    Depths are positive downward. Elevation is positive upward as seen from the array, so an
    arrival from a source shallower than the array has positive elevation.

    Returns a list of dicts with:
      'elevation_deg' : arrival elevation at the array
      'amplitude'     : relative amplitude (1/path-length spreading, times reflection coeffs)
      'path_m'        : slant path length
      'label'         : 'direct' / 'surface' / 'bottom'
    """
    out = []

    def add(z_img, coeff, label):
        dz = array_depth_m - z_img          # >0 when the image is above the array
        path = float(np.hypot(range_m, dz))
        out.append(dict(
            elevation_deg=float(np.degrees(np.arctan2(dz, range_m))),
            amplitude=float(coeff / max(path, 1e-9)),
            path_m=path,
            label=label,
        ))

    add(src_depth_m, 1.0, 'direct')
    add(-src_depth_m, surface_coeff, 'surface')
    for m in range(1, n_bottom + 1):
        add(2 * m * water_depth_m - src_depth_m, bottom_coeff ** m, f'bottom x{m}')
    return out


def bearing_rate_deg_per_s(speed_ms, range_m):
    """
    Bearing rate of a target on a straight track, at its closest point of approach, where the
    rate is highest: d(phi)/dt = v / r.
    """
    return float(np.degrees(speed_ms / max(range_m, 1e-9)))


def max_integration_time_s(speed_ms, cpa_range_m, hpbw_deg, fraction=0.25):
    """
    Longest coherent integration time before the target's bearing drifts more than
    `fraction * hpbw_deg` — the cap that target motion puts on the T in DT = 5*log10(d/WT).
    """
    rate = bearing_rate_deg_per_s(speed_ms, cpa_range_m)
    return float(fraction * hpbw_deg / rate) if rate > 0 else float('inf')
