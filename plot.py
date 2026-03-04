# /// script
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "tyro",
#     "msgpack",
#     "polars",
#     "tqdm",
#     "scikit-learn",
#     "PyArrow>=14.0.0"
# ]
# ///

import glob
import pprint

import matplotlib.pyplot as plt
import msgpack
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
from tyro.extras import SubcommandApp

import config

app = SubcommandApp()


def load_config(cfg_path):
    f = open(cfg_path, "rb")
    bin = f.read()
    f.close()
    return msgpack.unpackb(bin, raw=False)


def config_to_vec(cfg, subkey):
    v = []
    for key, value in cfg.items():
        options = config.META_CONFIG[subkey][key]
        if isinstance(options, list):
            idx = options.index(value)
            one_hot = [0]*len(options)
            one_hot[idx] = 1.
            v += one_hot
    return np.array(v)


@app.command
def hyper_plot(cfg: str, dir: str):
    configs = load_config(cfg)

    if "minatar" in dir:
        subkey = "minatar_small"
        default_config = config.META_CONFIG["minatar_baseline"]
    else:
        raise NotImplementedError()

    dfs = []
    cfg_vecs = []
    i = 0
    config_id = None
    for i, path in tqdm(enumerate(glob.glob(dir + "/*/*"))):
        run_id = int(path.split("/")[-1].split("_")[-1].split(".")[0])

        cfg = configs[run_id]
        equal = True
        for key, value in default_config.items():
            if cfg[key] != value:
                equal = False
                break

        if equal:
            print("Found default config!")
            config_id =i

        cfg_vecs.append(config_to_vec(cfg, subkey))

        df = pl.read_csv(path)

        df = df.group_by("seed").agg(pl.col("episodic_return").sum().alias("auc"))
        df = df.with_columns(pl.lit(run_id).alias("id"))
        dfs.append(df)

    assert config_id is not None, "Cannot find default config"
    print("Default config id:", config_id)

    cfg_vecs = np.vstack(cfg_vecs)
    print(cfg_vecs)
    print("Shape of config vectors:", cfg_vecs.shape)

    cfg_low_dim = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
    ).fit_transform(cfg_vecs)

    # cols: id, seed, auc
    df = pl.concat(dfs)

    # compute normalized auc
    q_5 = df.select(pl.col("auc").quantile(0.05)).item()
    q_95 = df.select(pl.col("auc").quantile(0.95)).item()
    df_norm = (
        df.group_by("id")
        .agg(pl.col("auc").mean())
        .with_columns(((pl.col("auc") - q_5) / (q_95 - q_5)).alias("auc_norm"))
    )

    # print(cfg_low_dim.shape)
    print(df_norm)

    plt.scatter(x=cfg_low_dim[:, 0], y=cfg_low_dim[:, 1], c=df_norm["auc_norm"])
    cbar = plt.colorbar()
    cbar.set_label("Normalized performance", rotation=270)
    plt.xlabel("X1")
    plt.ylabel("X2")

    # defaults_perf = df_norm.filter(pl.col("id") == config_id)["auc_norm"].item()
    # sns.kdeplot(data=df_norm, x="auc_norm")
    # plt.axvline(x=defaults_perf, color="tab:grey", label="baseline")
    # plt.legend()
    # plt.xlabel("Normalized performance")
    plt.show()


@app.command
def perf_distrib(cfg: str, dir: str):
    configs = load_config(cfg)

    if "minatar" in dir:
        default_config = config.META_CONFIG["minatar_baseline"]
    else:
        raise NotImplementedError()

    dfs = []
    i = 0
    config_id = None
    for i, path in tqdm(enumerate(glob.glob(dir + "/*/*"))):
        run_id = int(path.split("/")[-1].split("_")[-1].split(".")[0])

        cfg = configs[run_id]
        equal = True
        for key, value in default_config.items():
            if cfg[key] != value:
                equal = False
                break

        if equal:
            print("Found default config!")
            config_id =i

        df = pl.read_csv(path)

        df = df.group_by("seed").agg(pl.col("episodic_return").sum().alias("auc"))
        df = df.with_columns(pl.lit(run_id).alias("id"))
        dfs.append(df)

    assert config_id is not None, "Cannot find default config"
    print("Default config id:", config_id)

    # cols: id, seed, auc
    df = pl.concat(dfs)

    # compute normalized auc
    q_5 = df.select(pl.col("auc").quantile(0.05)).item()
    q_95 = df.select(pl.col("auc").quantile(0.95)).item()
    df_norm = (
        df.group_by("id")
        .agg(pl.col("auc").mean())
        .with_columns(((pl.col("auc") - q_5) / (q_95 - q_5)).alias("auc_norm"))
    )

    defaults_perf = df_norm.filter(pl.col("id") == config_id)["auc_norm"].item()
    sns.kdeplot(data=df_norm, x="auc_norm")
    plt.axvline(x=defaults_perf, color="tab:grey", label="baseline")
    plt.legend()
    plt.xlabel("Normalized performance")
    plt.show()


@app.command
def single_config(file: str, cfg: str):
    """Plot the results of a single configuration of PPO (all seeds)"""
    configs = load_config(cfg)

    run_id = int(file.split("/")[-1].split("_")[-1].split(".")[0])
    config = configs[run_id]

    pprint.pp(config)

    df = pd.read_csv(file)

    sns.lineplot(data=df, x="step", y="episodic_return", estimator=None, units="seed")
    plt.title(config["ENV_NAME"])
    plt.show()


if __name__ == "__main__":
    app.cli()
