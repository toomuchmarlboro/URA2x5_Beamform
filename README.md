# URA2x5_Beamform

Beamforming analysis of a **2 × 5 uniform rectangular hydrophone array** — 10 elements on a
0.83 m grid, back-baffled, standing upright and facing broadside.

**→ [Read the findings](docs/findings.md)** — 15 figures and the full set of results.

## The array

| | |
|---|---|
| Elements | 10 (5 along the horizontal axis × 2 stacked vertically) |
| Spacing | 0.83 m, both axes |
| Apertures | 3.32 m azimuth × 0.83 m elevation |
| Design point | f₀ = 904 Hz (where d = λ/2), c = 1500 m/s |
| Processing band | 100–900 Hz |
| Mounting | back-baffled, broadside at azimuth 0° |

## Headline results

| | |
|---|---|
| Beamwidth / peak sidelobe at f₀ | **20.6°** / **−12.0 dB** |
| With a Hann taper | **−33.6 dB** sidelobes for 0.97 dB of array gain |
| Usable steering sector | **±45°** of one face — ~4 panels for 360° |
| Alias-free | to **904 Hz** at any steer; **1807 Hz** at broadside only |
| Overhead (zenith) rejection | **−63.5 dB** at f₀, but only −1.2 dB at 300 Hz |
| Front/back ambiguity | exact in the array manifold — the baffle is what resolves it |

**The band where everything works at once is 605–904 Hz** — narrower than the configured
processing band. Alias-free at any steer, tapering usefully, and MVDR nulling for free.

## Layout

```
Testbenches/URA2x5_Beamformed.ipynb   ideal-condition baseline, 26 asserted results
ura2x5/                               shared core (geometry, beamforming, noise, detection,
                                      tracking, robustness, I/O) — array-agnostic
config.yaml                           every parameter, in one place
docs/findings.md                      the write-up
docs/make_figures.py                  regenerates every figure from the core
```

## Reproducing

```bash
pip install numpy scipy matplotlib pyyaml nbformat nbclient ipywidgets

python docs/make_figures.py                                   # 15 figures -> docs/images/
jupyter nbconvert --execute --inplace Testbenches/URA2x5_Beamformed.ipynb
```

The notebook executes end to end with every closed-form result asserted (|y| = 1 at the look
direction, HPBW 20.6°, first null 23.6°, the exact front/back degeneracy, the two aliasing
thresholds), so the core cannot drift without failing loudly.

## Scope

This covers **ideal conditions only**: perfect elements, spatially white noise, constant sound
speed, far-field sources, no multipath, no host platform. It is an upper bound and a
regression suite, not a performance prediction. A realistic-conditions revision — correlated
ocean noise, channel and position errors, ADC timing skew, heading, multipath, a detector and
a tracker — is the next step; the modules in `ura2x5/` are already in place for it.

The baffle's front-to-back rejection is a **placeholder (20 dB)** pending measurement of the
installed figure. `docs/findings.md` §3 gives the requirement as a formula.
