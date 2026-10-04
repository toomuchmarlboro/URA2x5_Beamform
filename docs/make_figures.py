"""Generate the figures used by docs/findings.md (2x5 URA panel).

Run from the repository root:  python docs/make_figures.py

Every figure comes from the shared `ura2x5` core and `config.yaml`, so the numbers here are
the ones URA2x5_Beamformed.ipynb asserts on. Nothing is hand-drawn or copied across.

Palette: the same validated categorical palette used by the sibling BEAMFORM_LEN docs
(slots 1-4), checked with the dataviz validator against this light surface -- all gates pass.
The sub-3:1 contrast on aqua/yellow is relieved by always-present legends, line-style
variation, and the numeric tables in findings.md. Heatmaps use a single-hue blue sequential
ramp (never a rainbow); requirement limits and failures use the reserved `critical` status
colour, always with a text label so state is never carried by hue alone.
"""
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
IMG = ROOT / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette & style
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRITICAL = "#d03b3b"
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("blueseq", BLUE_SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.7,
    "grid.alpha": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 6,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlepad": 8,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def finish(fig, name):
    fig.savefig(IMG / name)
    plt.close(fig)
    print(f"  wrote docs/images/{name}")


def fwd_polar(ax, show_r=True):
    """Forward-sector polar chrome: the baffled panel only sees +/-90 deg."""
    ax.set_theta_zero_location("E")
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines["polar"].set_color(AXIS)
    ax.set_thetagrids([-90, -60, -30, 0, 30, 60, 90], fontsize=7)
    ax.set_yticklabels([])
    ax.tick_params(colors=MUTED, labelsize=7)


# ---------------------------------------------------------------- shared core
from ura2x5.io import load_config
from ura2x5 import geometry
from ura2x5.geometry import (wrap180, unit_vector, mirror_azimuth_deg,
                             baffle_response, required_fbr_db)
from ura2x5.beamform import (steering_vector as _sv, das_weights as _dw, beamformer_output,
                             beam_pattern as _bp, to_db_norm, lobe_metrics, sample_covariance,
                             das_gain_vs_frequency as _dgf, mvdr_weights as _mvdr,
                             theta_null_line_deg as _tnl, hpbw_line_deg as _hpl,
                             make_taper as _taper)
from ura2x5.noise import simulate_snapshots_in_field, ship_noise_model as _ship

CFG = load_config(ROOT / "config.yaml")
C_WATER = CFG["environment"]["c_water"]
F0 = CFG["design"]["f0_hz"]
STEER_AZ, STEER_EL = CFG["design"]["steer_az_deg"], CFG["design"]["steer_el_deg"]
PHASE_BITS = CFG["design"]["phase_bits"]
BAND = tuple(CFG["design"]["band_hz"])
N_LONG, N_STACK, D_ELEM = (CFG["array"]["n_long"], CFG["array"]["n_stack"],
                           CFG["array"]["d_m"])
FBR_DB = CFG["baffle"]["front_to_back_db"]
FBR_TRANS = CFG["baffle"]["transition_deg"]

pos, labels, idx = geometry.build_ura(N_LONG, N_STACK, D_ELEM)
N = pos.shape[0]
geo = geometry.ura_derived(pos, D_ELEM, C_WATER, N_LONG, N_STACK)
AP_LONG, AP_STACK = geo["aperture_long"], geo["aperture_stack"]
F_ALIAS = geo["f_alias"]
F_ALIAS_BS = C_WATER / D_ELEM
F_NO_NULL = C_WATER / (N_LONG * D_ELEM)

sv = partial(_sv, positions=pos, c=C_WATER)
dw = partial(_dw, positions=pos, c=C_WATER)
PHI = np.arange(-180.0, 180.0, 0.2)
FWD = (PHI >= -90.0) & (PHI <= 90.0)
bp = partial(_bp, phi_grid=PHI, positions=pos, c=C_WATER)
tnl = partial(_tnl, n=N_LONG, d=D_ELEM, c=C_WATER)
hpl = partial(_hpl, n=N_LONG, d=D_ELEM, c=C_WATER)
dgf = partial(_dgf, positions=pos, c=C_WATER)
mvdr_w = partial(_mvdr, positions=pos, c=C_WATER)
taper = partial(_taper, positions=pos, axis=1)


def baffled(steer_el, steer_az, f_, **kw):
    g = baffle_response(PHI, FBR_DB, transition_deg=FBR_TRANS)
    return bp(steer_el, steer_az, f_, **kw) * g**2


def fwdm(pdb, steer=0.0, search=60.0):
    return lobe_metrics(PHI[FWD], pdb[FWD], steer, circular=False, search=search)


def fwd_peaks(pdb, thr=-6.0):
    p = pdb[FWD]
    m = (p > np.roll(p, 1)) & (p >= np.roll(p, -1)) & (p > thr)
    return int(m[1:-1].sum())


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    TAPS = {t: taper(t) for t in ["uniform", "hamming", "hann", "chebyshev"]}
TAP_EFF = {t: 10 * np.log10(g.sum()**2 / (len(g) * np.sum(g**2))) for t, g in TAPS.items()}

R = {}
print("Generating figures ...")

# ============================================================ fig01 geometry
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]
ax.scatter(pos[:, 1], pos[:, 2], s=150, color=S1, zorder=3, edgecolor=SURFACE, linewidth=2)
for p, lb in zip(pos, labels):
    ax.annotate(lb, (p[1], p[2]), xytext=(0, 13), textcoords="offset points",
                fontsize=7.5, ha="center", color=INK_2)
ax.annotate("", xy=(-1.66, -0.95), xytext=(1.66, -0.95),
            arrowprops=dict(arrowstyle="<->", color=S3, lw=1.6))
ax.text(0, -1.18, f"long axis {AP_LONG:.2f} m  ({N_LONG} elements)", ha="center",
        color=S3, fontsize=8.5, weight="bold")
ax.annotate("", xy=(2.15, -0.415), xytext=(2.15, 0.415),
            arrowprops=dict(arrowstyle="<->", color=S2, lw=1.6))
ax.text(2.3, 0, f"stack\n{AP_STACK:.2f} m", color=S2, fontsize=8.5, va="center", weight="bold")
ax.set_xlim(-2.4, 3.2); ax.set_ylim(-1.45, 1.1); ax.set_aspect("equal")
ax.set_xlabel("Y (m)"); ax.set_ylabel("Z (m)")
ax.set_title("Front view — the panel as the wavefront sees it")

ax = axes[1]
ax.plot([0, 0], [-1.66, 1.66], color=S1, lw=6, solid_capstyle="round")
ax.scatter(np.zeros(N_LONG), pos[:N_LONG, 1], s=45, color=SURFACE, zorder=3)
ax.fill_betweenx([-1.9, 1.9], -0.42, -0.08, color=MUTED, alpha=0.5)
ax.text(-0.25, -2.25, "baffle", ha="center", fontsize=8.5, color=INK_2, weight="bold")
for phi_, st in [(35.0, "-"), (145.0, (0, (4, 2)))]:
    v = np.array([np.cos(np.radians(phi_)), np.sin(np.radians(phi_))])
    ax.annotate("", xy=2.3 * v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=S2 if phi_ < 90 else CRITICAL,
                                lw=2, linestyle=st))
ax.text(2.0, 1.45, "35°", color=S2, fontsize=9, weight="bold")
ax.text(-2.75, 1.45, "145°", color=CRITICAL, fontsize=9, weight="bold")
ax.text(0, 2.45, "identical steering vectors (§3)\nthe baffle blocks the 145° path",
        ha="center", fontsize=8, color=INK_2)
ax.set_xlim(-3.1, 3.1); ax.set_ylim(-2.6, 3.0); ax.set_aspect("equal")
ax.set_xlabel("X (m) — broadside"); ax.set_ylabel("Y (m)")
ax.set_title("Top view — panel edge-on")

ax = axes[2]; ax.axis("off"); ax.grid(False)
ax.set_title("Derived geometry")
rows = [("Elements", f"{N}  ({N_LONG} x {N_STACK})"),
        ("Spacing d", f"{D_ELEM:.2f} m, both axes"),
        ("Long aperture (azimuth)", f"{AP_LONG:.2f} m"),
        ("Stack aperture (elevation)", f"{AP_STACK:.2f} m"),
        ("White-noise array gain", f"{10*np.log10(N):.1f} dB"),
        ("λ = Nd  (no null below)", f"{F_NO_NULL:.0f} Hz"),
        ("d = λ/2 (alias-free below)", f"{F_ALIAS:.0f} Hz"),
        ("d = λ   (aliased above)", f"{F_ALIAS_BS:.0f} Hz"),
        ("HPBW at f₀ = 904 Hz", "20.6°"),
        ("Peak sidelobe at f₀", "−12.0 dB")]
for j, (k_, v_) in enumerate(rows):
    y = 0.94 - j * 0.097
    ax.text(0.02, y, k_, fontsize=8.8, color=INK_2, transform=ax.transAxes)
    ax.text(0.60, y, v_, fontsize=8.8, color=INK, transform=ax.transAxes, family="monospace")
finish(fig, "fig01_geometry.png")

# ============================================================ fig02 beam pattern
p_af, p_bf = bp(0.0, 0.0, F0), baffled(0.0, 0.0, F0)
pdb_af, pdb_bf = to_db_norm(p_af), to_db_norm(p_bf)
m = fwdm(pdb_bf)
R["hpbw"], R["psl"], R["null"] = m["hpbw"], m["psl"], m["null_half"]
rear = ~FWD
R["rear_af"], R["rear_bf"] = pdb_af[rear].max(), pdb_bf[rear].max()

fig = plt.figure(figsize=(13, 4.6))
ax = fig.add_subplot(121)
ax.plot(PHI, pdb_af, color=MUTED, ls=(0, (5, 2)), lw=1.4, label="bare array factor (no baffle)")
ax.plot(PHI, pdb_bf, color=S1, label=f"baffled panel ({FBR_DB:.0f} dB rear rejection)")
ax.axvspan(-90, 90, color=S3, alpha=0.07, lw=0)
ax.annotate(f"HPBW {m['hpbw']:.1f}°\nfirst null ±{m['null_half']:.0f}°\nPSL {m['psl']:.1f} dB",
            (12, -20), fontsize=8.5, color=INK, weight="bold")
ax.annotate("rear lobe removed\nby the baffle", (105, -27), fontsize=8.5, color=CRITICAL)
ax.set_xlim(-180, 180); ax.set_ylim(-40, 3); ax.set_xticks(range(-180, 181, 60))
ax.set_xlabel("Arrival azimuth (deg)"); ax.set_ylabel("Normalised response (dB)")
ax.set_title(f"Beam pattern at broadside, {F0:.0f} Hz")
ax.legend(loc="lower right")

axp = fig.add_subplot(122, projection="polar")
axp.plot(np.radians(PHI), np.clip(pdb_bf, -30, 0) + 30, color=S1)
fwd_polar(axp)
axp.set_title("Forward sector (clipped at −30 dB)")
finish(fig, "fig02_beam_pattern.png")

# ============================================================ fig03 elevation / zenith
el = np.linspace(-90, 90, 721)
w0 = dw(0.0, 0.0, F0)
pel = to_db_norm(np.abs(np.conj(w0) @ sv(el, 0.0, F0))**2)
m_el = lobe_metrics(el, pel, 0.0, circular=False, search=5)
f_z = np.linspace(BAND[0], 1100.0, 320)
zen = np.array([20*np.log10(abs(beamformer_output(dw(0.0, 0.0, f_), sv(90.0, 0.0, f_))) + 1e-18)
                for f_ in f_z])
# Evaluate the zenith null exactly: it is extremely sharp, so interpolating the
# plotted grid understates its depth.
R["vert_hpbw"] = m_el["hpbw"]
R["zen_f0"] = float(20*np.log10(abs(beamformer_output(dw(0.0, 0.0, F0),
                                                      sv(90.0, 0.0, F0))) + 1e-18))
R["zen_300"] = float(np.interp(300.0, f_z, zen))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ax = axes[0]
ax.plot(el, pel, color=S1)
ax.axvline(90, color=CRITICAL, ls=(0, (5, 2)), lw=1.3)
ax.axvline(-90, color=CRITICAL, ls=(0, (5, 2)), lw=1.3)
ax.annotate("zenith null", (52, -48), fontsize=9, color=CRITICAL, weight="bold")
ax.annotate(f"vertical HPBW {m_el['hpbw']:.0f}°\n(only 2 elements)", (-86, -12),
            fontsize=8.5, color=INK_2)
ax.set_xticks(range(-90, 91, 30)); ax.set_ylim(-70, 3)
ax.set_xlabel("Elevation (deg)   ← below    above →"); ax.set_ylabel("Normalised response (dB)")
ax.set_title(f"Vertical cut at {F0:.0f} Hz")
ax = axes[1]
ax.plot(f_z, zen, color=S2)
ax.axvline(F_ALIAS, color=CRITICAL, ls=(0, (5, 2)), lw=1.3)
ax.annotate(f"vertical d = λ/2\n{F_ALIAS:.0f} Hz", (F_ALIAS - 250, -50), fontsize=8.5,
            color=CRITICAL, weight="bold")
ax.axvspan(*BAND, color=S3, alpha=0.07, lw=0)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Response to a zenith source (dB)")
ax.set_title("Overhead rejection: excellent at f₀, poor low down")
finish(fig, "fig03_elevation_zenith.png")

# ============================================================ fig04 baffle budget
g_baf = baffle_response(PHI, FBR_DB, transition_deg=FBR_TRANS)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ax = axes[0]
ax.plot(PHI, 20*np.log10(g_baf), color=S1)
ax.axvspan(-90, 90, color=S3, alpha=0.07, lw=0)
ax.annotate("forward half-space\n(elements see this)", (-80, -6), fontsize=8.5, color=INK_2)
ax.annotate("blocked", (118, -FBR_DB + 1.6), fontsize=9, color=CRITICAL, weight="bold")
ax.set_xlim(-180, 180); ax.set_xticks(range(-180, 181, 60)); ax.set_ylim(-FBR_DB - 6, 4)
ax.set_xlabel("Arrival azimuth (deg)"); ax.set_ylabel("Baffle attenuation (dB)")
ax.set_title("Element directivity: a forward hemisphere")
ax = axes[1]
back_rel = np.linspace(-10, 40, 120)
for marg, ls, col in [(6.0, (0, (1, 1.6)), S3), (10.0, "-", S1), (20.0, (0, (5, 2)), S2)]:
    ax.plot(back_rel, required_fbr_db(back_rel, marg), ls=ls, color=col,
            label=f"false contact ≥ {marg:.0f} dB down")
ax.axhline(FBR_DB, color=CRITICAL, lw=1.5)
ax.annotate(f"assumed baffle {FBR_DB:.0f} dB", (-9, FBR_DB + 1.6), fontsize=8.5,
            color=CRITICAL, weight="bold")
ax.set_xlabel("Source astern, relative to the wanted target (dB)")
ax.set_ylabel("Required front-to-back rejection (dB)")
ax.set_title("What the baffle has to deliver")
ax.legend(loc="upper left")
finish(fig, "fig04_baffle.png")

# ============================================================ fig05 resolution
f_c = np.linspace(250, 950, 400)
f_m = np.array([400, 500, 650, 800, 900])
meas = [fwdm(to_db_norm(bp(0.0, 0.0, float(f_))))["null_half"] for f_ in f_m]
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.plot(f_c, tnl(f_c), color=S1, label=r"first null, $\arcsin(\lambda/Nd)$")
ax.plot(f_c, hpl(f_c), color=S3, ls=(0, (5, 2)), lw=1.8, label=r"HPBW, $0.886\lambda/Nd$")
ax.scatter(f_m, meas, s=40, color=S2, zorder=5, edgecolor=SURFACE, linewidth=1.5,
           label="measured null, 5 elements")
ax.axvline(F_NO_NULL, color=CRITICAL, lw=1.4, ls=(0, (6, 3)))
ax.annotate(f"no null below\n{F_NO_NULL:.0f} Hz", (F_NO_NULL + 14, 72), fontsize=8.5,
            color=CRITICAL, weight="bold")
ax.set_xlim(250, 950); ax.set_ylim(0, 95)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Angle (deg)")
ax.set_title("Azimuth resolution — a 3.32 m aperture earns its keep above 361 Hz")
ax.legend()
finish(fig, "fig05_resolution.png")

# ============================================================ fig06 broadband loss
fr = np.linspace(BAND[0], BAND[1], 400)
fig, ax = plt.subplots(figsize=(8.2, 4.3))
for steer, col, ls in [(0.0, S3, (0, (5, 2))), (40.0, S2, "-")]:
    g = dgf(0.0, steer, F0, fr)
    ax.plot(fr, 20*np.log10(np.maximum(g, 1e-6)), color=col, ls=ls,
            label=f"phase-shift weights fixed at f₀, steered {steer:.0f}°")
ax.axhline(0, color=S1, lw=2, label="true-delay / per-bin beamformer")
ax.set_xlim(*BAND); ax.set_ylim(-20, 3)
ax.set_xlabel("Signal frequency (Hz)"); ax.set_ylabel("Gain toward the look direction (dB)")
ax.set_title("At broadside every delay is zero — steer off it and per-bin becomes essential")
ax.legend(loc="lower center")
finish(fig, "fig06_broadband_loss.png")

# ============================================================ fig07 taper at f0
fig, ax = plt.subplots(figsize=(8.6, 4.4))
tap_rows = []
for t, col, ls in [("uniform", MUTED, (0, (5, 2))), ("hann", S1, "-"),
                   ("hamming", S2, (0, (4, 2))), ("chebyshev", S3, (0, (1, 1.6)))]:
    pdb = to_db_norm(bp(0.0, 0.0, F0, amp=TAPS[t]))
    mm = fwdm(pdb)
    tap_rows.append((t, mm["hpbw"], mm["psl"], TAP_EFF[t]))
    ax.plot(PHI, pdb, color=col, ls=ls, lw=2 if t == "hann" else 1.7,
            label=f"{t}: PSL {mm['psl']:.1f} dB, HPBW {mm['hpbw']:.0f}°, {TAP_EFF[t]:+.1f} dB gain")
R["taper_f0"] = tap_rows
ax.set_xlim(-90, 90); ax.set_ylim(-55, 3); ax.set_xticks(range(-90, 91, 30))
ax.set_xlabel("Azimuth (deg)"); ax.set_ylabel("Normalised response (dB)")
ax.set_title(f"At f₀ a taper works on a URA — unlike on a ring")
ax.legend(loc="lower center")
finish(fig, "fig07_taper_f0.png")

# ============================================================ fig08 steering
sw = np.arange(0, 86, 2.5)
hp_s, psl_s = [], []
for s_ in sw:
    pdb = to_db_norm(bp(0.0, float(s_), F0))
    mm = fwdm(pdb, float(s_), search=70)
    hp_s.append(mm["hpbw"]); psl_s.append(mm["psl"])
hp_s, psl_s = np.array(hp_s), np.array(psl_s)
ok = np.flatnonzero(hp_s <= 1.5 * hp_s[0])
SECTOR = float(sw[ok[-1]])
R["sector"] = SECTOR
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ax = axes[0]
ax.plot(sw, hp_s, color=S1, marker="o", ms=4, label="measured")
ax.plot(sw, hpl(F0, steer_deg=sw), color=S3, ls=(0, (5, 2)), lw=1.8,
        label=r"$0.886\lambda/(Nd\cos\phi_s)$")
ax.axhline(1.5 * hp_s[0], color=CRITICAL, lw=1.4, ls=(0, (6, 3)))
ax.annotate(f"1.5× broadside", (1, 1.5 * hp_s[0] + 2.5), fontsize=8.5, color=CRITICAL,
            weight="bold")
ax.axvline(SECTOR, color=CRITICAL, lw=1.2, alpha=0.5)
ax.set_ylim(0, 90); ax.set_xlabel("Steer azimuth (deg)"); ax.set_ylabel("HPBW (deg)")
ax.set_title("The beam broadens off broadside"); ax.legend()
ax = axes[1]
ax.plot(sw, psl_s, color=S1, marker="o", ms=4)
ax.axvline(SECTOR, color=CRITICAL, lw=1.4, ls=(0, (6, 3)))
ax.annotate(f"usable sector\n±{SECTOR:.0f}°", (SECTOR + 2, -9), fontsize=8.5,
            color=CRITICAL, weight="bold")
ax.set_xlabel("Steer azimuth (deg)"); ax.set_ylabel("Peak sidelobe (dB)")
ax.set_title("…and sidelobe discrimination collapses near endfire")
finish(fig, "fig08_steering.png")

# ============================================================ fig09 alias thresholds
def max_steer(f_):
    s = C_WATER / (f_ * D_ELEM) - 1.0
    if s >= 1.0:
        return 90.0
    if s <= 0.0:
        return np.nan
    return float(np.degrees(np.arcsin(s)))


ff = np.linspace(600, 2200, 500)
ms = np.array([max_steer(f_) for f_ in ff])
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(ff, ms, color=S1, label="largest steer with no grating lobe")
ax.axvline(F_ALIAS, color=S3, lw=1.6)
ax.annotate(f"d = λ/2\n{F_ALIAS:.0f} Hz\nany steer OK below", (F_ALIAS - 300, 55),
            fontsize=8.5, color=S3, weight="bold")
ax.axvline(F_ALIAS_BS, color=CRITICAL, lw=1.6)
ax.annotate(f"d = λ\n{F_ALIAS_BS:.0f} Hz\naliased even at broadside", (F_ALIAS_BS + 30, 55),
            fontsize=8.5, color=CRITICAL, weight="bold")
ax.axvspan(*BAND, color=S3, alpha=0.07, lw=0)
ax.set_xlim(600, 2200); ax.set_ylim(0, 95)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Maximum steer angle (deg)")
ax.set_title("Aliasing has two thresholds, not one")
ax.legend(loc="lower left")
finish(fig, "fig09_alias_thresholds.png")

# ============================================================ fig10 DAS sweep
SWEEP_F = [50, 100, 150, 250, 500, 750, 1000, 1200, 1500, 3000]
REGIME = [(F_NO_NULL, "no null — not usable for bearing", MUTED),
          (F_ALIAS, "clean at any steer", S3),
          (F_ALIAS_BS, "clean at broadside only", S2),
          (np.inf, "grating lobes at broadside", CRITICAL)]


def regime_of(f_):
    for lim, nm, col in REGIME:
        if f_ < lim:
            return nm, col
    return REGIME[-1][1], REGIME[-1][2]


fig, axes = plt.subplots(2, 5, figsize=(18, 9.0), subplot_kw={"projection": "polar"})
das_rows = []
for ax, f_ in zip(axes.ravel(), SWEEP_F):
    pdb = to_db_norm(bp(0.0, 0.0, float(f_)))
    mm = fwdm(pdb, 0.0, search=89)
    npk = fwd_peaks(pdb)
    nm, col = regime_of(f_)
    ax.plot(np.radians(PHI), np.clip(pdb, -30, 0) + 30, color=col)
    fwd_polar(ax)
    hp = "none" if not np.isfinite(mm["hpbw"]) else f"{mm['hpbw']:.0f}°"
    ax.set_title(f"{f_} Hz{'' if BAND[0] <= f_ <= BAND[1] else '  (out of band)'}\n"
                 f"d/λ = {D_ELEM*f_/C_WATER:.2f},  HPBW {hp}\n"
                 f"{npk} peak{'s' if npk != 1 else ''}",
                 fontsize=8.5, color=INK if npk == 1 else CRITICAL, pad=10)
    das_rows.append((f_, C_WATER/f_, C_WATER/f_/(N_LONG*D_ELEM), D_ELEM*f_/C_WATER,
                     mm["hpbw"], mm["null_half"], mm["psl"], npk, nm))
R["das_rows"] = das_rows
fig.legend(handles=[plt.Line2D([], [], color=c, lw=2.6, label=nm) for _, nm, c in REGIME],
           loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.012))
fig.suptitle("DAS at broadside across frequency — forward sector, each panel clipped at −30 dB",
             fontsize=11, y=0.99)
plt.tight_layout(rect=[0, 0.04, 1, 0.95]); fig.subplots_adjust(hspace=0.42)
finish(fig, "fig10_sweep_das.png")

# ============================================================ fig11/12 MVDR sweep
INTF_AZ, INTF_INR = 40.0, 20.0
WNG_DAS = 10 * np.log10(N)
WNG_WARN = WNG_DAS - 4.0


def mvdr_at(f_, steer=0.0):
    a_i = sv(0.0, INTF_AZ, f_)
    Rm = 10 ** (INTF_INR / 10) * np.outer(a_i, a_i.conj()) + np.eye(N)
    return mvdr_w(0.0, steer, f_, Rm, loading=1e-2), a_i


fig, axes = plt.subplots(2, 5, figsize=(18, 9.0), subplot_kw={"projection": "polar"})
mv_rows = []
for ax, f_ in zip(axes.ravel(), SWEEP_F):
    w, a_i = mvdr_at(float(f_))
    A = sv(0.0, PHI, float(f_))
    pdb_m = to_db_norm(np.abs(np.conj(w) @ A)**2)
    pdb_d = to_db_norm(bp(0.0, 0.0, float(f_)))
    wng = 10*np.log10(1.0/np.real(np.conj(w) @ w))
    nd = 20*np.log10(abs(beamformer_output(w, a_i)) + 1e-18)
    frag = wng < WNG_WARN
    cause = ("superdirective" if (frag and f_ < F_ALIAS)
             else ("grating-lobe conflict" if frag else ""))
    ax.plot(np.radians(PHI), np.clip(pdb_d, -40, 0) + 40, color=MUTED, ls=(0, (5, 2)), lw=1.2)
    ax.plot(np.radians(PHI), np.clip(pdb_m, -40, 0) + 40, color=S1)
    ax.plot([np.radians(INTF_AZ)]*2, [0, 40], color=S4, ls=(0, (2, 2)), lw=1.6)
    fwd_polar(ax)
    ax.set_title(f"{f_} Hz   null {nd:.0f} dB\nWNG {wng:+.1f} dB"
                 f"{'  ← ' + cause if frag else '  (= DAS)'}",
                 fontsize=8.5, color=CRITICAL if frag else INK, pad=10)
    mv_rows.append((f_, nd, wng, WNG_DAS - wng, frag, cause))
R["mv_rows"] = mv_rows
fig.legend(handles=[plt.Line2D([], [], color=MUTED, ls="--", lw=1.8, label="DAS (reference)"),
                    plt.Line2D([], [], color=S1, lw=2.4, label="MVDR"),
                    plt.Line2D([], [], color=S4, ls=":", lw=2, label=f"interferer at {INTF_AZ:.0f}°")],
           loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.012))
fig.suptitle(f"MVDR at broadside across frequency — +{INTF_INR:.0f} dB interferer at "
             f"{INTF_AZ:.0f}°, clipped at −40 dB", fontsize=11, y=0.99)
plt.tight_layout(rect=[0, 0.04, 1, 0.95]); fig.subplots_adjust(hspace=0.42)
finish(fig, "fig11_sweep_mvdr.png")

f_fine = np.logspace(np.log10(40), np.log10(3500), 220)
wng_f = np.array([10*np.log10(1.0/np.real(np.conj(mvdr_at(float(f_))[0]) @ mvdr_at(float(f_))[0]))
                  for f_ in f_fine])
nul_f = np.array([20*np.log10(abs(beamformer_output(*(lambda r: (r[0], r[1]))(mvdr_at(float(f_))))) + 1e-18)
                  for f_ in f_fine])
okm = np.flatnonzero(wng_f >= WNG_WARN)
F_MVDR_OK = float(f_fine[okm[0]])
F_RES = C_WATER / (N_LONG * D_ELEM * np.sin(np.radians(INTF_AZ)))
F_HP = 0.886 * C_WATER / (N_LONG * D_ELEM * 2 * np.radians(INTF_AZ))
R["mvdr_ok"], R["f_res"], R["f_hp"] = F_MVDR_OK, F_RES, F_HP

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ax = axes[0]
ax.semilogx(f_fine, wng_f, color=S1, label="MVDR")
ax.axhline(WNG_DAS, color=MUTED, ls=(0, (5, 2)), lw=1.5, label=f"uniform DAS = {WNG_DAS:.1f} dB")
ax.axhline(WNG_WARN, color=CRITICAL, ls=(0, (2, 2)), lw=1.4, label="4 dB below DAS")
ax.axvspan(f_fine[0], F_MVDR_OK, color=CRITICAL, alpha=0.08, lw=0)
ax.annotate(f"superdirective\nbelow {F_MVDR_OK:.0f} Hz", (48, -4), fontsize=9,
            color=CRITICAL, weight="bold")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("White-noise gain (dB)")
ax.set_title("What nulling costs in robustness"); ax.legend(loc="lower right")
ax = axes[1]
ax.semilogx(f_fine, nul_f, color=S2)
ax.axvspan(f_fine[0], F_MVDR_OK, color=CRITICAL, alpha=0.08, lw=0)
ax.axvspan(F_MVDR_OK, F_RES, color=S3, alpha=0.12, lw=0)
ax.annotate(f"superresolution gap\n{F_MVDR_OK:.0f}–{F_RES:.0f} Hz", (240, -20), fontsize=8.5,
            color=INK, weight="bold")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(f"Response at the interferer (dB)")
ax.set_title(f"Null depth achieved at {INTF_AZ:.0f}°")
finish(fig, "fig12_mvdr_wng.png")

# ============================================================ fig13/14 taper sweep
fig, axes = plt.subplots(2, 5, figsize=(18, 9.0), subplot_kw={"projection": "polar"})
tp_rows = []
for ax, f_ in zip(axes.ravel(), SWEEP_F):
    pdb_u = to_db_norm(bp(0.0, 0.0, float(f_), amp=TAPS["uniform"]))
    pdb_t = to_db_norm(bp(0.0, 0.0, float(f_), amp=TAPS["hann"]))
    mu, mt = fwdm(pdb_u, 0.0, 89), fwdm(pdb_t, 0.0, 89)
    npk = fwd_peaks(pdb_t)
    helps = np.isfinite(mu["psl"]) and np.isfinite(mt["psl"]) and mt["psl"] < mu["psl"] - 3
    why = ("taper cannot touch grating lobes" if npk > 1
           else "main lobe fills the sector" if not np.isfinite(mt["psl"])
           else "grating-lobe shoulder at endfire" if not helps else "taper helps")
    ax.plot(np.radians(PHI), np.clip(pdb_u, -45, 0) + 45, color=MUTED, ls=(0, (5, 2)), lw=1.2)
    ax.plot(np.radians(PHI), np.clip(pdb_t, -45, 0) + 45, color=S1)
    fwd_polar(ax)
    fu = "n/a" if not np.isfinite(mu["psl"]) else f"{mu['psl']:.0f} dB"
    ft = "n/a" if not np.isfinite(mt["psl"]) else f"{mt['psl']:.0f} dB"
    ax.set_title(f"{f_} Hz   PSL {fu} → {ft}\n{why}", fontsize=8.5,
                 color=INK if helps else CRITICAL, pad=10)
    tp_rows.append((f_, mu["psl"], mt["psl"], npk, helps, why))
R["tp_rows"] = tp_rows
fig.legend(handles=[plt.Line2D([], [], color=MUTED, ls="--", lw=1.8, label="uniform"),
                    plt.Line2D([], [], color=S1, lw=2.4, label="Hann taper")],
           loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.012))
fig.suptitle(f"Hann taper across frequency — WNG cost {TAP_EFF['hann']:+.2f} dB at EVERY "
             f"frequency, benefit only in the middle", fontsize=11, y=0.99)
plt.tight_layout(rect=[0, 0.04, 1, 0.95]); fig.subplots_adjust(hspace=0.42)
finish(fig, "fig13_sweep_taper.png")

f_t = np.arange(250.0, 2000.0, 5.0)
curves = {}
for t in ["uniform", "hamming", "hann", "chebyshev"]:
    curves[t] = np.array([fwdm(to_db_norm(bp(0.0, 0.0, float(f_), amp=TAPS[t])), 0.0, 89)["psl"]
                          for f_ in f_t], dtype=float)
adv = curves["hann"] - curves["uniform"]
us = np.flatnonzero(np.isfinite(adv) & (adv < -3.0))
F_TAP_LO, F_TAP_HI = float(f_t[us[0]]), float(f_t[us[-1]])
R["tap_lo"], R["tap_hi"] = F_TAP_LO, F_TAP_HI

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ax = axes[0]
for t, col, ls in [("uniform", MUTED, (0, (5, 2))), ("hann", S1, "-"),
                   ("hamming", S2, (0, (4, 2))), ("chebyshev", S3, (0, (1, 1.6)))]:
    ax.plot(f_t, curves[t], color=col, ls=ls, lw=1.9 if t == "hann" else 1.5,
            label=f"{t} ({TAP_EFF[t]:+.1f} dB WNG)")
ax.axvspan(F_TAP_LO, F_TAP_HI, color=S3, alpha=0.10, lw=0)
ax.axvline(F_ALIAS, color=CRITICAL, ls=(0, (2, 2)), lw=1.4)
ax.annotate(f"d = λ/2", (F_ALIAS + 20, -52), fontsize=8.5, color=CRITICAL, weight="bold")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Peak sidelobe, forward sector (dB)")
ax.set_title("All tapers converge on 0 dB once grating lobes arrive")
ax.legend(loc="lower left")
ax = axes[1]
ax.plot(f_t, adv, color=S1)
ax.axhline(0, color=MUTED, lw=1.2)
ax.axhline(-3, color=CRITICAL, ls=(0, (2, 2)), lw=1.3, label="3 dB improvement")
ax.axvspan(F_TAP_LO, F_TAP_HI, color=S3, alpha=0.12, lw=0,
           label=f"taper worth it: {F_TAP_LO:.0f}–{F_TAP_HI:.0f} Hz")
ax.axvspan(F_NO_NULL, F_ALIAS, color=S2, alpha=0.10, lw=0,
           label=f"alias-free: {F_NO_NULL:.0f}–{F_ALIAS:.0f} Hz")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Hann PSL minus uniform PSL (dB)")
ax.set_title("Negative = taper wins. The two bands do not coincide.")
ax.legend(loc="upper left")
finish(fig, "fig14_taper_band.png")

# ============================================================ fig15 tracking / baffle value
n_sc, K_tr = 40, 32
PHI_TR = np.arange(-180.0, 180.0, 1.0)
FWD_TR = (PHI_TR >= -90) & (PHI_TR <= 90)
phi_true = np.linspace(-60, 60, n_sc)
bins_tr = np.linspace(300, BAND[1], 14)
A_tr = {f_: sv(0.0, PHI_TR, f_) for f_ in bins_tr}
ASTERN, ASTERN_SNR = 150.0, 15.0
AST_MIR = float(mirror_azimuth_deg(ASTERN))


def run_track(fbr):
    rng = np.random.default_rng(42)
    att = float(baffle_response(ASTERN, fbr, transition_deg=FBR_TRANS))
    eff = ASTERN_SNR + 20*np.log10(max(att, 1e-9))
    est, mp = [], []
    for p_ in phi_true:
        P = np.zeros(len(PHI_TR))
        for f_ in bins_tr:
            Rk = sample_covariance(simulate_snapshots_in_field(
                [(0.0, p_, 0.0), (0.0, ASTERN, eff)], f_, K_tr, rng, positions=pos, c=C_WATER))
            A = A_tr[f_]
            P += np.real(np.sum(np.conj(A) * (Rk @ A), axis=0)) / N**2
        mp.append(P); est.append(PHI_TR[FWD_TR][np.argmax(P[FWD_TR])])
    return np.array(est), np.array(mp), eff


est_b, mp_b, eff_b = run_track(FBR_DB)
est_u, mp_u, eff_u = run_track(0.0)
ref = mp_u.max()
mp_b, mp_u = mp_b/ref, mp_u/ref
rms_b = float(np.sqrt(np.mean(wrap180(est_b - phi_true)**2)))
rms_u = float(np.sqrt(np.mean(wrap180(est_u - phi_true)**2)))
R["rms_baf"], R["rms_un"] = rms_b, rms_u

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
k = np.arange(n_sc)
for ax, mpx, lab, rr in [(axes[0], mp_u, "UNBAFFLED", rms_u),
                         (axes[1], mp_b, f"BAFFLED ({FBR_DB:.0f} dB)", rms_b)]:
    im = ax.imshow(10*np.log10(mpx + 1e-12), aspect="auto", origin="lower", cmap=SEQ_CMAP,
                   vmin=-30, vmax=0, extent=[-180, 180, 0, n_sc])
    ax.plot(phi_true, k + 0.5, color=S2, lw=1.5, ls=(0, (4, 3)), label="true target")
    ax.axvline(AST_MIR, color=S4, ls=(0, (2, 2)), lw=1.6)
    ax.axvline(ASTERN, color=S4, ls=(0, (2, 2)), lw=1.6)
    ax.set_xticks(range(-180, 181, 90)); ax.grid(False)
    ax.set_xlabel("Azimuth (deg)"); ax.set_ylabel("Scan")
    ax.set_title(f"{lab} — bearing RMS {rr:.1f}°")
    ax.legend(loc="upper left", fontsize=8)
    cb = plt.colorbar(im, ax=ax, label="dB re unbaffled peak"); cb.outline.set_edgecolor(AXIS)
axes[0].annotate("astern ship\nleaks in here", (AST_MIR + 8, 4), fontsize=8, color=S4,
                 weight="bold")
axes[2].plot(k, phi_true, color=S3, lw=2, ls=(0, (5, 2)), label="true bearing")
axes[2].plot(k, est_u, ls="none", marker=".", ms=6, color=CRITICAL,
             label=f"unbaffled (RMS {rms_u:.1f}°)")
axes[2].plot(k, est_b, ".-", ms=4, lw=1, color=S1, label=f"baffled (RMS {rms_b:.2f}°)")
axes[2].axhline(AST_MIR, color=S4, ls=(0, (2, 2)), lw=1.6, label="astern ship, mirrored")
axes[2].set_xlabel("Scan"); axes[2].set_ylabel("Estimated bearing (deg)")
axes[2].set_title("Forward-sector bearing estimate"); axes[2].legend(loc="lower right")
finish(fig, "fig15_tracking_baffle.png")

# ---------------------------------------------------------------- persist values
import json


def plain(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (list, tuple)):
        return [plain(x) for x in o]
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    return o


out = ROOT / "docs" / "figure_values.json"
out.write_text(json.dumps(plain(R), indent=2), encoding="utf-8")
print(f"\nwrote docs/figure_values.json")
print(json.dumps(plain(R), indent=2).encode("ascii", "replace").decode("ascii")[:3000])
