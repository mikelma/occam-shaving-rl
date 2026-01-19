# /// script
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "tyro",
# ]
# ///

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tyro


def main(file: str):
    df = pd.read_csv(file)

    sns.lineplot(data=df, x="step", y="episodic_return", estimator=None, units="seed")
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
