# /// script
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "tyro",
#     "msgpack",
# ]
# ///

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tyro
import msgpack
import pprint


def main(file: str, cfg: str):
    f = open(cfg, "rb")
    bin = f.read()
    f.close()
    configs = msgpack.unpackb(bin, raw=False)

    run_id = int(file.split("/")[-1].split("_")[-1].split(".")[0])
    config = configs[run_id]

    pprint.pp(config)

    df = pd.read_csv(file)

    sns.lineplot(data=df, x="step", y="episodic_return", estimator=None, units="seed")
    plt.title(config["ENV_NAME"])
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
