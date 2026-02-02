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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import msgpack
import pprint
import glob
import polars as pl
from tqdm import tqdm
import numpy as np
from tyro.extras import SubcommandApp
from sklearn.manifold import TSNE

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
            v.append(idx)
    # v = []
    # for key, value in cfg.items():
    #     if isinstance(value, bool):
    #         v.append(int(value))

    #     elif isinstance(value, int) or isinstance(value, float):
    #         v.append(value)

    #     elif key in ["ACTIVATION", "INITIALIZERS", "INITIALIZERS", "ENV_NAME"]:
    #         idx = config.META_CONFIG[subkey][key].index(value)
    #         v.append(idx)

    #     elif key == "LOG_DIR":
    #         pass

    #     else:
    #         print(key, value)
    #         raise NotImplementedError()
    return np.array(v)


@app.command
def perf_distrib(cfg: str, dir: str):
    configs = load_config(cfg)

    if "minatar" in dir:
        subkey = "minatar_small"
    else:
        raise NotImplementedError()

    dfs = []
    cfg_vecs = []
    i = 0
    for path in tqdm(glob.glob(dir + "/*/*")):
        run_id = int(path.split("/")[-1].split("_")[-1].split(".")[0])

        cfg = configs[run_id]
        cfg_vecs.append(config_to_vec(cfg, subkey))

        df = pl.read_csv(path)

        df = df.group_by("seed").agg(pl.col("episodic_return").sum().alias("auc"))
        df = df.with_columns(pl.lit(run_id).alias("id"))
        dfs.append(df)

        i += 1
        # if i == 100:
        #     break

    cfg_vecs = np.vstack(cfg_vecs)
    print("Shape of config vectors:", cfg_vecs.shape)

    # vecs_min = cfg_vecs.min(axis=0)
    # vecs_max = cfg_vecs.max(axis=0)
    # cfg_vecs = (cfg_vecs - vecs_min) / (vecs_max - vecs_min)

    # FIX as we're using the index of hyperparameters: option 2 is closer to option 3 than to option 5 (we have to fix this)
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

    print(cfg_low_dim.shape)
    print(df_norm["auc_norm"])

    plt.scatter(x=cfg_low_dim[:, 0], y=cfg_low_dim[:, 1], c=df_norm["auc_norm"])
    cbar = plt.colorbar()
    cbar.set_label("Normalized performance", rotation=270)
    plt.xlabel("X1")
    plt.ylabel("X2")
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
