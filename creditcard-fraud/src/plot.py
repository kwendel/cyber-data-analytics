import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heatmap(df: pd.DataFrame, x_col, y_col):
    plt.figure(figsize=(30, 6))
    df_new = df.groupby([x_col, y_col])["txid"].size().reset_index(name="count")
    sns.heatmap(df_new.pivot(x_col, y_col, "count"), cmap='viridis')


def bar(df: pd.DataFrame, col):
    plt.figure(figsize=(8, 8))
    counts = (df.groupby(['simple_journal'])[col]
              .value_counts(normalize=True)
              .rename('percentage')
              .mul(100)
              .reset_index()
              # .sort_values('occupation')
              )

    # Combine values than are below 1 percent
    others = counts[counts['percentage'] <= 1].groupby(['simple_journal']).sum().reset_index()
    others[col] = 'Other'
    counts = counts[counts['percentage'] > 1]

    counts = counts.append(others, ignore_index=True)

    sns.set(style="whitegrid", font_scale=1.5)
    barplot = sns.barplot(x=col, y="percentage", hue="simple_journal", data=counts)
    plt.legend(loc='upper right')

    if col == 'txvariantcode':
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=30)
    barplot.get_figure().savefig('../plots/bar_' + col + ".png", dpi=300, format="png")
