# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///

import xml.etree.ElementTree as ET
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def load_csv(file: str) -> pd.DataFrame:
    return pd.read_csv(file)


def parse_data(data: pd.DataFrame, useglobalstep: bool) -> tuple[np.ndarray, np.ndarray]:
    steps = data["Step"].to_numpy() if not useglobalstep else data["global_step"].to_numpy()
    coldata = []
    for colname in reversed(data.columns.to_list()):
        if colname in ["Step", "global_step"] or colname.endswith("__MIN") or colname.endswith("__MAX") or colname.endswith("_step"):
            continue
        coldata.append(data[colname].to_numpy())

    if useglobalstep:
        for col in coldata:
            nonzerofirst = lambda z: z.nonzero()[0]
            colnan = np.isnan(col)
            col[colnan] = np.interp(nonzerofirst(colnan), nonzerofirst(~colnan), col[~colnan])

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

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.set_box_aspect(1)
    ax.grid(alpha=0.15)
    ax.set_xlabel("t")
    ax.set_ylabel("R")
    ax.ticklabel_format(useOffset=False, style="plain")

    fig.set_dpi(150)
    fig.tight_layout()

    return fig, ax


def plot_mean_stat(fig: Figure, ax: Axes, steps: np.ndarray, mean: np.ndarray, stat: Any, stattype: str, color: str | None = None, label: str | None = None) -> None:
    plotted = ax.plot(steps, mean, color=color, label=label)
    if stattype == "minmax":
        ax.fill_between(steps, stat[0], stat[1], color=plotted[0].get_color(), alpha=0.1)


def save_plot(fig: Figure, file: str, show_legend: bool) -> str:
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

    fig.savefig(file)
    print(f"Saved to '{file}'.")

    return file


def remove_svg_element_and_resize(file: str, element_id: str):
    tree = ET.parse(file)
    root = tree.getroot()

    element_to_remove = root.find(f".//*[@id='{element_id}']")
    if element_to_remove is None:
        print(f"Element with id '{element_id}' not found")
        return
    parent = root.find(f".//*[{element_id}]/..")
    if parent is None:
        for elem in root.iter():
            if element_to_remove in elem:
                elem.remove(element_to_remove)
                break
    else:
        parent.remove(element_to_remove)

    del root.attrib["viewBox"]
    # root.set("viewBox", "")

    tree.write(file, encoding="utf-8", xml_declaration=True)


def main(legend: bool, pairs: list[tuple[str, str]], output: str, useglobalstep: bool):
    fig, ax = makefig()

    for file, label in pairs:
        data = load_csv(file)
        runs = parse_data(data, useglobalstep)
        steps, mean, stat, stattype = get_mean_stat(runs)
        plot_mean_stat(fig, ax, steps, mean, stat, stattype, label=label)

    filename = save_plot(fig, output, legend)
    # remove_svg_element_and_resize(filename, "patch_1")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--useglobalstep", action="store_true", default=False)
    parser.add_argument("pairs", nargs="+")
    args = vars(parser.parse_args())

    if len(args["pairs"]) % 2 != 0:
        parser.error(f"Expected even number of pair arguments, got {len(args["pairs"])}")
    pairs = [(args["pairs"][i], args["pairs"][i + 1]) for i in range(0, len(args["pairs"]), 2)]
    args["pairs"] = pairs

    main(**args)
