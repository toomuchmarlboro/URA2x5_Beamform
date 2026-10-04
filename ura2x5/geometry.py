"""Array geometry for the 2x5 uniform rectangular array (URA).

Coordinate convention, chosen so that "the panel faces azimuth 0":

    * The panel lies in the **Y-Z plane**; its normal (broadside) is +X = azimuth 0 deg.
    * The **long axis is Y** — 5 elements, 4d = 3.32 m aperture. This is the azimuth aperture.
    * The **stack axis is Z** (vertical) — 2 elements, 1d = 0.83 m aperture. This gives only
      coarse elevation discrimination; two elements cannot form a real vertical beam.
    * Azimuth is measured in the X-Y plane from +X toward +Y; elevation is positive upward.

Consequences that fall straight out of this geometry, and which the notebook quantifies:

    * **Front/back ambiguity.** Reflecting a source through the panel plane (x -> -x) maps
      azimuth phi to 180 - phi and leaves every path length unchanged, so the array cannot
      tell a bearing ahead of the panel from its mirror behind. A baffle or a second panel
      is the only fix; no amount of processing resolves it.
    * **Steering is NOT invariant.** Unlike a ring, a planar array has a broadside: its beam
      broadens as 1/cos(angle from broadside) and degenerates entirely at endfire.
    * **Spatial aliasing at c/2d.** With d = 0.83 m that is 904 Hz — far lower than a ring of
      similar element count would give, and the dominant constraint on this design.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

__all__ = ["wrap180", "unit_vector", "build_ura", "ura_derived", "mirror_azimuth_deg",
           "baffle_response", "required_fbr_db"]


def wrap180(a):
    """Wrap angle(s) in degrees to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def unit_vector(theta_deg, phi_deg):
    """
    Unit vector(s) pointing FROM the array TOWARD a source at elevation theta_deg
    (0 = horizon, +90 = straight up) and azimuth phi_deg (0 = +X = broadside, toward +Y).
    Returns (3,) for scalars or (3, M) when an angle is an array.
    """
    th, ph = np.broadcast_arrays(np.radians(theta_deg), np.radians(phi_deg))
    return np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])


def build_ura(n_long=5, n_stack=2, d=0.83):
    """
    Planar 2-D array in the Y-Z plane: `n_long` elements along Y spaced `d`, `n_stack` rows
    along Z spaced `d`, both centred on the origin.

    Returns (positions (N,3), labels (N,) of "(row, col)", idx (N,)) with elements numbered
    row-major starting at 1 — row 1 is the LOWER row.
    """
    y = (np.arange(n_long) - (n_long - 1) / 2.0) * d
    z = (np.arange(n_stack) - (n_stack - 1) / 2.0) * d
    yy, zz = np.meshgrid(y, z)                       # (n_stack, n_long)
    pos = np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()])
    labels = np.array([f"r{r+1}c{c+1}" for r in range(n_stack) for c in range(n_long)])
    return pos, labels, np.arange(1, pos.shape[0] + 1)


def ura_derived(pos, d, c_water, n_long=5, n_stack=2):
    """Apertures, spacing limits and the beamwidth scales implied by the geometry."""
    dmat = squareform(pdist(pos))
    ap_long = (n_long - 1) * d                       # azimuth aperture (Y)
    ap_stack = (n_stack - 1) * d                     # elevation aperture (Z)
    return dict(
        dmat=dmat,
        d=d,
        n_long=n_long,
        n_stack=n_stack,
        aperture_long=ap_long,
        aperture_stack=ap_stack,
        D_max=dmat.max(),
        # d = lambda/2 limit. Above this a planar array grows true grating lobes.
        f_alias=c_water / (2 * d),
        # Frequency at which the long aperture spans one wavelength (lambda = D).
        f_lambda_eq_D=c_water / ap_long if ap_long > 0 else np.inf,
    )


def mirror_azimuth_deg(phi_deg):
    """
    The front/back ambiguous partner of an azimuth, for a panel in the Y-Z plane:
    reflecting through the panel (x -> -x) maps phi to 180 - phi.
    """
    return (180.0 - np.asarray(phi_deg)) % 360.0


def baffle_response(phi_deg, fbr_db, theta_deg=0.0, transition_deg=12.0):
    """
    Amplitude shading applied by a **back baffle** to an arrival from (theta, phi).

    The baffle does not change the array manifold — the front and back steering vectors stay
    identical (`mirror_azimuth_deg`), which is why no processor can separate them. What the
    baffle does is attenuate the back arrival *before* it reaches the elements, so the
    ambiguity is resolved at the input instead.

    Model: unity in the forward half-space (|phi| < 90 relative to the panel normal +X),
    10**(-fbr_db/20) behind, with a raised-cosine transition of width `transition_deg`
    straddling the rim. The transition matters: a hard step would put an artificial
    discontinuity at endfire, exactly where the beam is already degenerate.

    `fbr_db` is the front-to-back rejection in dB (amplitude ratio expressed in dB, so a
    source behind is attenuated by `fbr_db` in level). Returns a real array broadcastable
    against the azimuth grid.

    CAVEAT the notebook states explicitly: real baffles lose rejection at low frequency,
    where the wavelength is large compared with the baffle thickness. Treating `fbr_db` as
    frequency-flat is optimistic at the bottom of the band.
    """
    back = 10.0 ** (-abs(fbr_db) / 20.0)
    # Angle off the panel normal (+X), including elevation: cos of that angle is the
    # x-component of the arrival direction.
    u = unit_vector(theta_deg, phi_deg)
    ux = np.atleast_1d(u[0])
    # Map ux (+1 dead ahead, -1 dead astern) through a raised cosine centred on the rim.
    half = np.sin(np.radians(transition_deg / 2.0)) + 1e-12
    t = np.clip(ux / half, -1.0, 1.0)
    blend = 0.5 * (1.0 + np.sin(np.pi * t / 2.0))          # 0 fully behind, 1 fully ahead
    g = back + (1.0 - back) * blend
    return g if np.ndim(phi_deg) else float(g[0])


def required_fbr_db(back_source_rel_db, margin_db):
    """
    Front-to-back rejection needed so that a source BEHIND the panel, `back_source_rel_db`
    louder than the wanted target ahead, leaves a false contact at least `margin_db` below
    the true one.

        FBR >= back_source_rel_db + margin_db
    """
    return np.asarray(back_source_rel_db, dtype=float) + margin_db
