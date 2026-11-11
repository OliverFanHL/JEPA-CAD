"""Vector comparison of integrated-gradient attributions for three STEP parts."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / ".plot_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"],
    "font.size": 8.2, "axes.labelsize": 8.4, "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.4, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
})

data = json.loads((HERE / "gradient_attribution.json").read_text(encoding="utf-8"))
assert data["groups"] == ["Typed entities", "B-rep instances", "Connectivity", "Global STEP",
                          "Alternate view", "Geometry", "Surface profile"]
assets = ("coupling_337_hires.png", "pulley_530_hires.png",
          "coupling_257_hires.png")
abbreviations = ("Typed", "B-rep", "Connect.", "Global", "Alt.", "Geom.", "Surface")
colors = LinearSegmentedColormap.from_list(
    "signed_contribution", ["#687f8b", "#f6f7f6", "#688677"])
norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=6.5)

fig = plt.figure(figsize=(7.15, 4.0))
grid = fig.add_gridspec(2, 3, height_ratios=[1.48, .90], left=.105, right=.99,
                       top=.98, bottom=.17, hspace=.12, wspace=.12)
for i, (case, asset) in enumerate(zip(data["cases"], assets)):
    cell = grid[0, i].get_position(fig)
    image_axis = fig.add_axes([cell.x0 + .1 * cell.width,
                               cell.y0 + .1 * cell.height,
                               .8 * cell.width, .8 * cell.height])
    with Image.open(HERE / "qualitative_assets" / asset) as opened:
        image_axis.imshow(opened.convert("RGB"), interpolation="none")
    image_axis.axis("off")
    fig.text(cell.x0 + .006, cell.y1 - .012, "abc"[i],
             fontsize=10, fontweight="bold", ha="left", va="top")

    ax = fig.add_subplot(grid[1, i])
    values = np.asarray([case["control_group_attribution"],
                         case["residual_group_attribution"]], dtype=float)
    ax.imshow(values, cmap=colors, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(7), abbreviations, rotation=50, ha="right")
    if i == 0:
        ax.set_yticks([0, 1], ["Control", "Residual"])
    else:
        ax.set_yticks([0, 1], [])
    ax.tick_params(length=0, pad=2)
    for row in range(2):
        for column in range(7):
            value = values[row, column]
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center",
                    fontsize=6.7, color="white" if value < -1.35 or value > 3.1 else "#263b42")
    ax.set_xlim(-.5, 6.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.savefig(HERE / "tmcad_gradient_attribution.pdf", dpi=450,
            bbox_inches="tight", pad_inches=.035)
preview = HERE / "qa_previews"
preview.mkdir(exist_ok=True)
fig.savefig(preview / "tmcad_gradient_attribution.png", dpi=230,
            bbox_inches="tight", pad_inches=.035)
plt.close(fig)
