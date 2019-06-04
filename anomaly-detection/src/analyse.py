import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import parse_to_df, select_between_datetime

path_training_1 = '../data/BATADAL_training1.csv'
path_training_2 = '../data/BATADAL_training2.csv'
path_testing = '../data/BATADAL_test.csv'

sns.set()


def plot_correlation(df):
    # Plot the water level in tank 3 and the force and on/off signal of pump 4 as they are heavenly correlated
    plt.figure()

    # Fill the on/off sections with green and red
    plt.fill_between(df.index, 0, 40, where=df['s_pu4'] == 1, color='g', alpha=0.3)
    plt.fill_between(df.index, 0, 40, where=df['s_pu4'] == 0, color='r', alpha=0.3)

    # Plot sensor data
    sns.lineplot(df.index, df['l_t3'], label='Level T3')
    sns.lineplot(df.index, df['f_pu4'], label='Flow Pump 4')

    plt.title("Correlation: Tank 3 and Pump 4", fontweight='bold')
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")

    print(np.corrcoef([df['l_t3'], df['f_pu4'], df['s_pu4']]))

    plt.savefig("../plots/correlation.png")
    plt.show()


def plot_negative_corr(df):
    # Plot T1 level and the corresponding junctions that are close for negative correlation
    plt.figure()

    sns.lineplot(df.index, df['f_pu1'], label='Flow P1')
    # sns.lineplot(df.index, df['p_j280'], label='Suction P1')
    sns.lineplot(df.index, df['p_j269'], label='Discharge J269')

    plt.title("Negative correlation: Pump 1 and Junction 269", fontweight='bold')
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")

    print(np.corrcoef([df['f_pu1'], df['p_j269']]))

    plt.savefig("../plots/neg_correlation.png")
    plt.show()


def plot_nocorrelation(df):
    # Plot multiple sensor that are far away from each other, as they are likely not correlated
    plt.figure()

    sns.lineplot(df.index, df['l_t1'], label='Level T1')
    sns.lineplot(df.index, df['l_t4'], label='Level T4')
    sns.lineplot(df.index, df['f_pu8'], label='Flow P8')

    plt.title("No correlation: Tank 1, Tank 4 and Pump 8", fontweight='bold')
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")

    print(np.corrcoef([df['l_t1'], df['l_t4'], df['f_pu8']]))

    plt.savefig("../plots/no_correlation.png")
    plt.show()


def __fill_between(df, start, end, top_val):
    s = df[df['datetime'] == start].index[0]
    e = df[df['datetime'] == end].index[0]

    plt.fill_between(df.index, 0, top_val, color='r', alpha=0.3,
                     where=(df.index > s) & (df.index <= e))


def plot_attack_5_6(df_attack: pd.DataFrame):
    # Plot attack 5 & 6 of the training2 dataset
    # Attack 5 & 6 both reduced the working speed of PU7 which led to a lower water level in T4

    # Attack 5 - 26/11/2016 17 - 29/11/2016 04 - 60 hours - 6 labeled
    # No SCADA concealment

    # Attack 6 - 06/12/2016 07 - 10/12/2016 04 - 94 hours
    # SCADA concealment - L_T4 drop concealed with replay attack

    df_attack = select_between_datetime(df_attack, '2016-11-20 17:00:00', '2016-12-15 04:00:00')

    plt.figure()
    sns.lineplot(df_attack.index, df_attack['f_pu7'], label='Flow P7')
    sns.lineplot(df_attack.index, df_attack['l_t4'], label='Level T4')
    __fill_between(df_attack, '2016-11-26 17:00:00', '2016-11-29 04:00:00', 55)
    __fill_between(df_attack, '2016-12-06 07:00:00', '2016-12-10 04:00:00', 55)

    plt.title("Attack 5 and 6: reduce working speed Pump 7", fontweight='bold')
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")

    plt.savefig("../plots/attack_5_6.png")
    plt.show()


def plot_attack_3_4(df_attack):
    # Plot attack 3 & 4 of the training2 dataset
    # Attack 3 & 4 both alter the readings that L_T1 sents to PLC1 to PLC2, which keeps PU1 and PU2 on causing an overflow
    # Attack 3 - 09/10/2016 09 - 11/10/2016 20 - 60 hours
    # Attack 4 - 29/10/2016 09 - 02/11/2016 16 - 94 hours
    # SCADA concealment 3 - Polyline to offset increase L_T1
    # SCADA concealment 4 - Replay attack on L_T1, PU1 and PU2

    df_attack = select_between_datetime(df_attack, '2016-10-01 04:00:00', '2016-11-09 04:00:00')

    plt.figure()
    sns.lineplot(df_attack.index, df_attack['f_pu2'], label='Flow P2')
    sns.lineplot(df_attack.index, df_attack['l_t1'], label='Level T1')
    __fill_between(df_attack, '2016-10-09 09:00:00', '2016-10-11 20:00:00', 120)
    __fill_between(df_attack, '2016-10-29 09:00:00', '2016-11-02 16:00:00', 120)

    plt.title("Attack 3 and 4: alter readings of Tank 1", fontweight='bold')
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")

    # plt.savefig("../plots/attack_3_4.png")
    plt.show()


def plot_attack_1(df_attack):
    # Plot attack 1
    # Attacker changes water L_T7 thresholds which controls PU10 and PU11
    df_attack = select_between_datetime(df_attack, '2016-09-12 00:00:00', '2016-09-17 00:00:00')

    plt.figure()
    ax = sns.lineplot(df_attack.index, df_attack['f_pu10'], label='Flow P10')
    ax = sns.lineplot(df_attack.index, df_attack['f_pu11'], label='Flow P11')
    ax.lines[1].set_linestyle(":")
    sns.lineplot(df_attack.index, df_attack['l_t7'], label='Level T7')
    __fill_between(df_attack, '2016-09-13 23:00:00', '2016-09-16 00:00:00', 35)
    plt.title("Attack 1: alter water level thresholds of Tank 7", fontweight='bold')
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")

    # plt.savefig("../plots/attack_1.png")
    plt.show()


def data_analysis():
    # sns.set()
    df_n = parse_to_df(path_training_1)
    df_a = parse_to_df(path_training_2)

    # Correlation plots of first week
    plot_correlation(df_n.loc[0:7 * 24, :])
    plot_negative_corr(df_n.loc[0:7 * 24, :])
    plot_nocorrelation(df_n.loc[0:7 * 24, :])

    # Plots of attacks
    plot_attack_5_6(df_a)
    plot_attack_3_4(df_a)
    plot_attack_1(df_a)


if __name__ == '__main__':
    sns.set(font_scale=1.5)
    data_analysis()
