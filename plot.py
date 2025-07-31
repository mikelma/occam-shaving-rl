# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def load_csv(file: str) -> pd.DataFrame:
    return pd.read_csv(file)


def parse_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    steps = data["Step"].to_numpy()
    coldata = []
    for colname in reversed(data.columns.to_list()):
        if colname == "Step" or colname.endswith("__MIN") or colname.endswith("__MAX"):
            continue
        coldata.append(data[colname].to_numpy())

    return steps, np.asarray(coldata)


def get_mean_stat(rundata: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, Any, str]:
    steps, runs = rundata

    stattype = "minmax"
    means = runs.mean(axis=0)
    mins = runs.min(axis=0)
    maxs = runs.max(axis=0)

    return steps, means, (mins, maxs), stattype


def makefig() -> tuple[Figure, Axes]:
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "Latin Modern Math"

    fig, ax = plt.subplots(1, 1)

    ax.set_box_aspect(1)
    ax.grid(alpha=0.15)
    ax.set_xlabel("t")
    ax.set_ylabel("R")

    fig.set_dpi(150)
    fig.tight_layout()

    return fig, ax


def plot_mean_stat(fig: Figure, ax: Axes, steps: np.ndarray, mean: np.ndarray, stat: Any, stattype: str, color: str | None = None, label: str | None = None) -> None:
    plotted = ax.plot(steps, mean, color=color, label=label)
    if stattype == "minmax":
        ax.fill_between(steps, stat[0], stat[1], color=plotted[0].get_color(), alpha=0.1)


def save_plot(fig: Figure, file: str, show_legend: bool):
    if show_legend:
        legend = plt.gca().legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fancybox=False, edgecolor="black")
        lines = legend.get_lines()
        for line in lines:
            line.set_solid_capstyle("butt")
            decoration = Line2D(
                line.get_xdata(), line.get_ydata(), linewidth=15, color=line.get_color(), alpha=0.5, solid_capstyle="butt", zorder=line.get_zorder()  # type: ignore
            )

            assert line.axes != None
            line.axes.add_artist(decoration)
            decoration.set_transform(line.get_transform())
            decoration.set_clip_path(line.get_clip_path())
            decoration.set_clip_box(line.get_clip_box())

    savename = file.replace(".csv", ".svg")
    fig.savefig(savename)
    print(f"Saved to '{savename}'.")


def main(legend: bool, pairs: list[tuple[str, str]]):
    fig, ax = makefig()

    for file, label in pairs:
        data = load_csv(file)
        runs = parse_data(data)
        steps, mean, stat, stattype = get_mean_stat(runs)
        plot_mean_stat(fig, ax, steps, mean, stat, stattype, label=label)

    save_plot(fig, file, legend)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("pairs", nargs="+")
    args = vars(parser.parse_args())

    if len(args["pairs"]) % 2 != 0:
        parser.error(f"Expected even number of pair arguments, got {len(args["pairs"])}")
    pairs = [(args["pairs"][i], args["pairs"][i + 1]) for i in range(0, len(args["pairs"]), 2)]
    args["pairs"] = pairs

    main(**args)
