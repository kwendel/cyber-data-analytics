import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data import parse_to_df, select_between_datetime


def plot_correlation(df):
    # Select data in the first week
    # df = df.loc[start_day * 24:end_day * 24, :]

    # Plot the water level in tank 3 and the force and on/off signal of pump 4 as they are heavenly correlated
    plt.figure()

    # Fill the on/off sections with green and red
    plt.fill_between(df.index, 0, 40, where=df['s_pu4'] == 1, color='g', alpha=0.3)
    plt.fill_between(df.index, 0, 40, where=df['s_pu4'] == 0, color='r', alpha=0.3)

    # Plot sensor data
    sns.lineplot(df.index, df['l_t3'], label='Level T3')
    # sns.lineplot(df.index, df['f_pu4'], label='Flow Pump 4')

    plt.show()


def plot_negative_corr(df):
    # Select data in the first week
    # df = df.loc[0:7 * 24, :]

    # Plot T1 level and the corresponding junctions that are close for negative correlation
    plt.figure()

    sns.lineplot(df.index, df['f_pu1'], label='Flow P1')
    # sns.lineplot(df.index, df['p_j280'], label='Suction P1')
    sns.lineplot(df.index, df['p_j269'], label='Discharge P1')

    plt.show()


def plot_nocorrelation(df):
    # df = df.loc[0:7 * 24, :]

    # Plot multiple sensor that are far away from each other, as they are likely not correlated
    plt.figure()

    sns.lineplot(df.index, df['l_t1'], label='Level T1')
    sns.lineplot(df.index, df['l_t4'], label='Level T4')
    sns.lineplot(df.index, df['f_pu8'], label='Flow P8')

    plt.show()


def plot_attack5(df_normal: pd.DataFrame, df_attack: pd.DataFrame):
    # Plot attack 5 of the training2 dataset
    # Attack 5 & 6 both reduced the working speed of PU7 which led to a lower water level in T4
    # Attack 5 - 26/11/2016 17 - 29/11/2016 04 - 60 hours - 6 labeled
    # No SCADA concealment

    # Select the same timeperiod for both dataset
    # Note that the datasets are from different years, but they are compared to see if there are no seasonal patterns

    # df_normal = select_between_datetime(df_normal, '2014-11-26 17:00:00', '2014-11-29 04:00:00')
    df_attack = select_between_datetime(df_attack, '2016-11-15 17:00:00', '2016-12-15 04:00:00')

    start = df_attack[df_attack['datetime'] == '2016-11-26 17:00:00'].index[0]
    print(start)
    end = df_attack[df_attack['datetime'] == '2016-11-29 04:00:00'].index[0]
    print(end)

    # plt.figure()
    # sns.lineplot(df_normal.index, df_normal['f_pu7'], label='Flow P7')
    # sns.lineplot(df_normal.index, df_normal['l_t4'], label='Level T4')
    # plt.show()

    def __fill_between(start, end):
        s = df_attack[df_attack['datetime'] == start].index[0]
        e = df_attack[df_attack['datetime'] == end].index[0]

        plt.fill_between(df_attack.index, 0, 60, color='r', alpha=0.3,
                         where=(df_attack.index > s) & (df_attack.index <= e))

    plt.figure()
    sns.lineplot(df_attack.index, df_attack['f_pu7'], label='Flow P7')
    sns.lineplot(df_attack.index, df_attack['l_t4'], label='Level T4')
    __fill_between('2016-11-26 17:00:00', '2016-11-29 04:00:00')
    plt.show()


def plot_attack6(df_normal, df_attack):
    # Plot attack 5 of the training2 dataset
    # Attack 5 & 6 both reduced the working speed of PU7 which led to a lower water level in T4
    # Attack 6 - 06/12/2016 07 - 10/12/2016 04 - 94 hours
    # SCADA concealment - L_T4 drop concealed with replay attack

    df_normal = select_between_datetime(df_normal, '2014-12-06 07:00:00', '2014-12-10 04:00:00')
    df_attack = select_between_datetime(df_attack, '2016-12-06 07:00:00', '2016-12-10 04:00:00')

    plt.figure()
    sns.lineplot(df_normal.index, df_normal['f_pu7'], label='Flow P7')
    sns.lineplot(df_normal.index, df_normal['l_t4'], label='Level T4')
    plt.show()

    plt.figure()
    sns.lineplot(df_attack.index, df_attack['f_pu7'], label='Flow P7')
    sns.lineplot(df_attack.index, df_attack['l_t4'], label='Level T4')
    plt.show()


if __name__ == '__main__':
    sns.set()

    df_n = parse_to_df('../data/BATADAL_training1.csv')
    df_a = parse_to_df('../data/BATADAL_training2.csv')

    # Correlation plots of first week
    # plot_correlation(df_n[0:7 * 24, :])
    # plot_negative_corr(df_n[0:7 * 24, :])
    # plot_nocorrelation(df_n[0:7 * 24, :])

    # Plots of attacks
    plot_attack5(df_n, df_a)
    plot_attack6(df_n, df_a)

    # TODOs:
    # - outliers
    # - more time intervals
    # - maybe show something strange in the data
    # - analyse training 2 and testing data
