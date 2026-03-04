import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import glob
    import pprint

    import numpy as np
    import matplotlib.pyplot as plt
    import msgpack
    import numpy as np
    import polars as pl
    import seaborn as sns

    import config

    return config, glob, mo, msgpack, np, pl, pprint


@app.cell
def _(msgpack):
    def load_config(cfg_path):
        f = open(cfg_path, "rb")
        bin = f.read()
        f.close()
        return msgpack.unpackb(bin, raw=False)

    return (load_config,)


@app.function
def run_id_from_path(path):
    return int(path.split("/")[-1].split("_")[-1].split(".")[0])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load MinAtar data
    """)
    return


@app.cell
def _(glob, pl):
    def csv_dir_to_df(csv_dir, bin_path, configs, default_config):
        dfs = []
        config_ids = [] 
        # print("Num. of configs:", len(configs))
        for i, path in enumerate(glob.glob(csv_dir + "/*/*")):
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

            df = pl.read_csv(path)
            df = df.group_by("seed").agg(pl.col("episodic_return").sum().alias("auc"))
            df = df.with_columns(pl.lit(run_id).alias("id"))
            df = df.with_columns(pl.lit(cfg["ENV_NAME"]).alias("env"))
            dfs.append(df)

        # cols: id, seed, auc, env
        return pl.concat(dfs), config_ids

    return (csv_dir_to_df,)


@app.cell
def _(config, csv_dir_to_df, load_config):
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
    )
    return breakout_configs, df_breakout


@app.cell
def _(config, csv_dir_to_df, load_config):
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
    )
    return ast_inv_configs, df_ast_inv


@app.cell
def _(ast_inv_configs, breakout_configs, df_ast_inv, df_breakout, pl):
    minatar_configs = breakout_configs + ast_inv_configs

    df_ast_inv_new_ids = df_ast_inv.with_columns(pl.col("id") + len(breakout_configs))
    df_minatar = pl.concat([df_breakout, df_ast_inv_new_ids]) 

    print(df_minatar)
    return df_minatar, minatar_configs


@app.cell
def _(df_minatar, pl):
    # compute per-environment 5% and 95% quantiles (global across all runs in that env)
    q_env = (
        df_minatar.group_by("env")
          .agg(
              pl.col("auc").quantile(0.05).alias("q_5"),
              pl.col("auc").quantile(0.95).alias("q_95"),
          )
    )

    print(q_env)

    # aggregate per (env, id) then normalize using that env's quantiles
    df_norm = (
        df_minatar.group_by(["env", "id"])
          .agg(pl.col("auc").mean().alias("auc"))
          .join(q_env, on="env", how="left")
          .with_columns(
              ((pl.col("auc") - pl.col("q_5")) / (pl.col("q_95") - pl.col("q_5"))).alias("auc_norm")
          )
          .drop(["q_5", "q_95"])
    )

    df_norm
    return (df_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting
    """)
    return


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _(df_norm, pl):
    df_plot = (     
        df_norm.group_by("id")
            .agg(pl.col("auc_norm")
                .mean()
                .alias("auc_norm_avg"),
                pl.col("env").first()
                )
        )
    df_plot
    return (df_plot,)


@app.cell
def _(alt, df_plot, mo):
    chart = alt.Chart(df_plot).mark_point().encode(x="id", y="auc_norm_avg", tooltip=["id"], color="env")
    mo_chart = mo.ui.altair_chart(chart)
    mo_chart
    return (mo_chart,)


@app.cell
def _():
    from collections import defaultdict

    def values_per_key(dicts):
        out = defaultdict(set)          
        for d in dicts:
            for k, v in d.items():
                out[k].add(v)
        return {k: list(vs) for k, vs in out.items()}  

    def dict_diff(dicts):
        sames = {}
        diffs = {} 
        for key in dicts[0].keys():
            value = dicts[0][key]
            # search for same key and value pairs
            shared = True
            different = True
            diff_vals = []
            for d in dicts:
                if shared and d[key] != value:
                    shared = False

                if len(diff_vals) == 0:
                    diff_vals.append(d[key])
                else:
                    for v in diff_vals:
                        if d[key] == d:
                            different = False
                            break
                    if different:
                        diff_vals.append(d[key])
                    
                if not shared and not different:
                        break
            
            if shared:
                sames[key] = value    
            elif different:
                diffs[key] = value
        return sames, diffs    

    return (dict_diff,)


@app.cell
def _(dict_diff, minatar_configs, mo_chart, pprint):
    sel_ids = mo_chart.value["id"]

    for id in sel_ids:
        print(f"\n*** ID: {id}")
        pprint.pprint(minatar_configs[id])
    print()

    if len(sel_ids) > 1:
        same_params, different_params = dict_diff([minatar_configs[id] for id in sel_ids])

        print("--- Shared hyperparameters ---")
        pprint.pprint(same_params)
        print()
        print("--- Different hyperparameters ---")
        pprint.pprint(different_params)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sensitivity plots
    """)
    return


@app.function
def filter_hyperparams(hypers, configs):
    filtered = []
    ids = []
    for i, conf in enumerate(configs):
        use = True
        for key, value in hypers.items():
            if conf[key] != value:
                use = False
                break
        if use:
            filtered.append(conf)
            ids.append(i)
    
    return filtered, ids


@app.cell
def _(df_norm, minatar_configs, np, pl):
    algorithms = [
        {"LAYER_NORM": True},
        {"LAYER_NORM": False},
        {"GAE_NORMALIZATION": True},
        {"GAE_NORMALIZATION": False},
        {"ANNEAL_LR": False},
        {"ANNEAL_LR": True},
    ]

    df_sens = (     
        df_norm.group_by("id")
            .agg(
                pl.col("auc_norm")
                .mean()
                .alias("perf"),  # average auc_norm over seeds
                pl.col("env").first())
        )


    algo_ids = []
    sens_list = []
    per_env_list = []
    algo_txt = []
    for i, algo in enumerate(algorithms):
        confs, ids = filter_hyperparams(algo, minatar_configs)

        ids = np.array(ids)
        df_sel = df_sens.filter(pl.col("id").is_in(ids))
    
        per_env = df_sel.group_by("env").agg(pl.col("perf").max())
        per_env = per_env["perf"].mean()

        across_env = df_sel.group_by("id").agg(pl.col("perf").mean())
        across_env = across_env["perf"].max()

        sensitivity = per_env - across_env
    
        sens_list.append(sensitivity)
        per_env_list.append(per_env)
        algo_ids.append(i)
        algo_txt.append(", ".join([f"{k}={v}" for k, v in algo.items()]))
    return algo_ids, algo_txt, algorithms, per_env_list, sens_list


@app.cell(hide_code=True)
def _(algo_ids, algo_txt, alt, mo, per_env_list, pl, sens_list):
    df_sens_plot = pl.DataFrame({"per env perf": per_env_list, "sensitivity": sens_list, "algo id": algo_ids, "label": algo_txt}) 

    chart2 = alt.Chart(df_sens_plot).mark_point().encode(x="sensitivity", y="per env perf")
    text = chart2.mark_text(
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        text='label'
    )
    mo_chart2 = mo.ui.altair_chart(chart2 + text)
    mo_chart2
    return (mo_chart2,)


@app.cell
def _(algorithms, mo_chart2):
    sel_algos = mo_chart2.value["algo id"]
    for algo_id in sel_algos:
        print("Selected algorithm:", algorithms[algo_id])
    return


if __name__ == "__main__":
    app.run()
