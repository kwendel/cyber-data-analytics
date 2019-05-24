import matplotlib.pyplot as plt
import seaborn as sns

from .data import parse_to_df


def plot_correlation(df):
    # Select data in the first week
    df = df.loc[0:7 * 24, :]

    # Plot the water level in tank 3 and the force and on/off signal of pump 4 as they are heavenly correlated
    plt.figure()

    # Fill the on/off sections with green and red
    plt.fill_between(df.index, 0, 40, where=df['s_pu4'] == 1, color='g', alpha=0.3)
    plt.fill_between(df.index, 0, 40, where=df['s_pu4'] == 0, color='r', alpha=0.3)

    # Plot sensor data
    sns.lineplot(df.index, df['l_t3'], label='Level T3')
    sns.lineplot(df.index, df['f_pu4'], label='Flow Pump 4')

    plt.show()


def plot_negative_corr(df):
    # Select data in the first week
    df = df.loc[0:7 * 24, :]

    # Plot T1 level and the corresponding junctions that are close for negative correlation
    plt.figure()

    sns.lineplot(df.index, df['f_pu1'], label='Flow P1')
    # sns.lineplot(df.index, df['p_j280'], label='Suction P1')
    sns.lineplot(df.index, df['p_j269'], label='Discharge P1')

    plt.show()


def plot_nocorrelation(df):
    df = df.loc[0:7 * 24, :]

    # Plot multiple sensor that are far away from each other, as they are likely not correlated
    plt.figure()

    sns.lineplot(df.index, df['l_t1'], label='Level T1')
    sns.lineplot(df.index, df['l_t4'], label='Level T4')
    sns.lineplot(df.index, df['f_pu8'], label='Flow P8')

    plt.show()


if __name__ == '__main__':
    path = '../data/BATADAL_training1.csv'
    sns.set()
    df = parse_to_df(path)
    print(df.columns)

    # Correlation plots
    plot_correlation(df)
    plot_negative_corr(df)
    plot_nocorrelation(df)

    # TODOs:
    # - outliers
    # - more time intervals
    # - maybe show something strange in the data
    # - analyse training 2 and testing data
