import pandas as pd
from tqdm import tqdm
import msgpack
import glob
import config
import tyro
from tyro.extras import SubcommandApp

app = SubcommandApp()


def load_config(cfg_path):
    f = open(cfg_path, "rb")
    bin = f.read()
    f.close()
    return msgpack.unpackb(bin, raw=False)


def run_id_from_path(path):
    return int(path.split("/")[-1].split("_")[-1].split(".")[0])


def csv_dir_to_df(csv_dir, bin_path, configs, default_config, progess=False):
    dfs = []
    config_ids = []
    # print("Num. of configs:", len(configs))
    iter_prog = tqdm if progess else lambda x: x
    for i, path in iter_prog(enumerate(glob.glob(csv_dir + "/*/*"))):
        run_id = run_id_from_path(path)
        # print("run_id:", run_id)
        cfg = configs[run_id]
        equal = True
        for key, value in default_config.items():
            if cfg[key] != value:
                equal = False
                break
        if equal:
            print(f"Found default config at id={i}!")
            config_ids.append(i)

        df = pd.read_csv(path)
        df = df.groupby(by="seed").sum()
        df = df.rename(columns={"episodic_return": "auc"})
        df["env"] = cfg["ENV_NAME"]
        df["id"] = run_id
        df = df.drop(["step", "episode_len"], axis="columns")
        dfs.append(df)

    return pd.concat(dfs), config_ids


@app.command
def cache_results(
    out_csv: str = "./minatar_results.csv", out_config: str = "./minatar_configs.bin"
):
    breakout_dir = "results/minatar_breakout"
    breakout_bin_path = "results/minatar_breakout.bin"

    # get the configuration dictionaries
    breakout_configs = load_config(breakout_bin_path)
    breakout_default = config.META_CONFIG["minatar_baseline"]

    df_breakout, default_id_breakout = csv_dir_to_df(
        csv_dir=breakout_dir,
        bin_path=breakout_bin_path,
        configs=breakout_configs,
        default_config=breakout_default,
        progess=True,
    )

    ast_inv_dir = "results/minatar_asterix_invaders"
    ast_inv_bin_path = "results/minatar_small_asterix_invaders.bin"

    # get the configuration dictionaries
    ast_inv_configs = load_config(ast_inv_bin_path)
    ast_inv_default = config.META_CONFIG["minatar_baseline"]

    df_ast_inv, default_id_ast_inv = csv_dir_to_df(
        csv_dir=ast_inv_dir,
        bin_path=ast_inv_bin_path,
        configs=ast_inv_configs,
        default_config=ast_inv_default,
        progess=True,
    )

    # Join both dataframes into a single one
    minatar_configs = breakout_configs + ast_inv_configs

    df_ast_inv["id"] += len(breakout_configs)
    df_minatar = pd.concat([df_breakout, df_ast_inv])

    df_minatar.to_csv(out_csv)

    bin_data = msgpack.packb(minatar_configs, use_bin_type=True)
    with open(out_config, "wb") as binary_file:
        binary_file.write(bin_data)


@app.command
def parallel_plot(
    csv_path="./minatar_results.csv", config_path="./minatar_configs.bin"
):
    df = pd.read_csv(csv_path)
    lst_conf = load_config(config_path)
    print(df)
    print(len(lst_conf), len(df["id"].unique()))


if __name__ == "__main__":
    app.cli()
