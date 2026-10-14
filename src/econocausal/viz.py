"""Shared plot style and helpers for notebooks and reports."""

import matplotlib.pyplot as plt

from econocausal.paths import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8984"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"
BACKGROUND = "#fcfcfb"


def set_style() -> None:
    """Light, minimal matplotlib style: thin marks, quiet grid and axes."""
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG in reports/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
