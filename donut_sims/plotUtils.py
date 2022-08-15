"""Plotting functions."""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec


def plotCwfs(
    images: Dict[str, npt.NDArray[np.float64]], vmax: int = None
) -> plt.Figure:
    """Plot the CWFS images.

    Parameters
    ----------
    images: dict
        The dictionary of CWFS images. Format is
        {`chip_name`: np.ndarray}.
    vmax: int, optional
        Max range of the colormap. See plt.imshow.

    Returns
    -------
    plt.Figure
        Figure of the images.
    """

    # create the figure
    fig = plt.figure(figsize=(6, 6), dpi=120, constrained_layout=True)

    # create the grid of detectors
    gs = GridSpec(4, 4, figure=fig)
    axes = {
        "R40_SW1": fig.add_subplot(gs[:2, 0]),
        "R40_SW0": fig.add_subplot(gs[:2, 1]),
        "R44_SW1": fig.add_subplot(gs[0, -2:]),
        "R44_SW0": fig.add_subplot(gs[1, -2:]),
        "R00_SW0": fig.add_subplot(gs[2, :2]),
        "R00_SW1": fig.add_subplot(gs[3, :2]),
        "R04_SW0": fig.add_subplot(gs[-2:, 2]),
        "R04_SW1": fig.add_subplot(gs[-2:, 3]),
    }

    # plot the available detectors
    for chip, img in images.items():
        axes[chip].imshow(img, origin="lower", vmax=vmax)

    # remove all the tickmarks
    for ax in axes.values():
        ax.set(xticks=[], yticks=[])

    return fig


def plotStamps(stamps: Dict[str, npt.NDArray[np.float64]]) -> plt.Figure:
    """Plot the donut postage stamps.

    Parameters
    ----------
    stamps: dict
        The dictionary of donut postage stamps.

    Returns
    -------
    plt.Figure
        Figure of the images.
    """
    # create the figure with the correct number of rows
    nrows = np.ceil(len(stamps.values()) / 10).astype(int)
    fig, axes = plt.subplots(
        nrows, 10, figsize=(16, 1.6 * nrows), constrained_layout=True
    )

    # plot all the stamps
    for i, img in enumerate(stamps.values()):
        ax = axes.flatten()[i]
        ax.imshow(img, origin="lower")
        ax.set(xticks=[], yticks=[], title=i)

    # remove empty frames
    for ax in axes.flatten()[i + 1 :]:
        ax.remove()

    return fig
