import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heatmap(df: pd.DataFrame, x_col, y_col):
    plt.figure(figsize=(30, 6))
    df_new = df.groupby([x_col, y_col])["txid"].size().reset_index(name="count")
    sns.heatmap(df_new.pivot(x_col, y_col, "count"), cmap='viridis')


def bar(df: pd.DataFrame, col):
    plt.figure(figsize=(30, 6))
    counts = (df.groupby(['simple_journal'])[col]
              .value_counts(normalize=True)
              .rename('percentage')
              .mul(100)
              .reset_index()
              # .sort_values('occupation')
              )
    sns.barplot(x=col, y="percentage", hue="simple_journal", data=counts)
