import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def eda_report(df: pd.DataFrame, target: str = None, plots: bool = True):
    """
    Print dataset overview and (optional) plots.
    """
    print("=== Dataset Overview ===")
    print(df.info())
    print(df.describe(include="all").T)

    if plots:
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

        if target and target in df.columns:
            sns.histplot(df[target], kde=True)
            plt.title(f"Distribution of {target}")
            plt.show()
