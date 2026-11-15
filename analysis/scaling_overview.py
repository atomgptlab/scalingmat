"""Time vs system size for ALIGNN-FF (energy only).

Run from this directory: `python scaling_overview.py`. Writes `scaling_overview.png`
from ../b200/alignn_ff/scaling_alignn_v6.npz
and ../{gb10,h200}/alignn_ff_b200settings/*.npz. Every ALIGNN-FF curve here is
energy only with the B200 run's settings (Cu, 12 neighbours,
v12.2.2024_dft_3d_307k); the GB10 and H200 curves are reruns of those settings on
the pure-PyTorch model. The energy+forces+stress sweeps are in scaling_forces.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root (analysis/ sits at the top level)
B200 = ROOT / "b200"

# ALIGNN-FF: total time per energy evaluation (times_graph already includes nl)
d = np.load(B200 / "alignn_ff" / "scaling_alignn_v6.npz")
an = d["natoms"][1:].astype(float)  # drop N=4 (warm-up)
at = (d["times_graph"] + d["times_inference"])[1:]

# The energy+forces+stress sweeps (GB10, H200) are plotted separately by
# scaling_forces.py.


# ALIGNN-FF with the B200 run's own settings (energy only, Cu, 12 neighbours,
# same checkpoint), rerun on the pure-PyTorch model; first row = warm-up, dropped
# as for the B200 series. The H200 file is added once its run has finished.
def load_b200set(gpu_dir):
    fs = sorted(
        (ROOT / gpu_dir / "alignn_ff_b200settings").glob(
            "scaling_alignn_pure_*.npz"
        )
    )
    if not fs:
        return None
    z = np.load(fs[-1])
    return (
        z["natoms"][1:].astype(float),
        (z["times_graph"] + z["times_inference"])[1:],
    )


eg = load_b200set("gb10")
eh = load_b200set("h200")

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200)
ax.loglog(
    an,
    at,
    "-o",
    ms=3.5,
    lw=1.8,
    color="#2a78d6",
    label="ALIGNN-FF, 1 B200, energy only",
)
if eg is not None:
    ax.loglog(
        *eg,
        "--D",
        ms=3.5,
        lw=1.5,
        color="#2ca02c",
        label="ALIGNN-FF, 1 GB10, energy only"
    )
if eh is not None:
    ax.loglog(
        *eh,
        "--P",
        ms=4,
        lw=1.5,
        color="#7b3294",
        label="ALIGNN-FF, 1 H200, energy only"
    )
ax.set_xlabel("Number of atoms")
ax.set_ylabel("Wall time (s)")
ax.grid(True, which="major", color="#e4e3df")
order = [
    "ALIGNN-FF, 1 B200, energy only",
    "ALIGNN-FF, 1 GB10, energy only",
    "ALIGNN-FF, 1 H200, energy only",
]
handles = dict(zip(*ax.get_legend_handles_labels()[::-1]))
order = [k for k in order if k in handles]
ax.legend(
    [handles[k] for k in order],
    order,
    loc="upper left",
    bbox_to_anchor=(0.0, -0.14),
    frameon=False,
    fontsize=8.5,
    ncol=1,
)
fig.tight_layout()
fig.savefig(
    HERE / "scaling_overview.png", facecolor="white", bbox_inches="tight"
)
