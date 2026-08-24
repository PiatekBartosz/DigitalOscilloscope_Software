"""Shared visual style for figures prepared for the thesis."""

from __future__ import annotations


def format_thesis_axis(axis, x_label: str, y_label: str, *, legend: bool = True) -> None:
    """Apply the project-wide thesis style to a Matplotlib axis.

    Figures have labelled axes, a legend when labelled traces are present, and
    no title because the thesis caption supplies the figure description.
    """
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title("")
    axis.grid(True, which="major", color="#B0B0B0", alpha=0.55, linewidth=0.7)
    axis.minorticks_on()
    axis.grid(True, which="minor", color="#D8D8D8", alpha=0.45, linewidth=0.4)

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", frameon=True)
