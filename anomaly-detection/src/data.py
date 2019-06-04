import numpy as np
import pandas as pd


def select_between_datetime(df, start, end):
    return df[
        (df['datetime'] > start) &
        (df['datetime'] <= end)
        ]


def parse_date(date):
    return pd.datetime.strptime(date, '%d/%m/%y %H')


attack_dates_training2 = [
    ['2016-09-13 23:00:00', '2016-09-16 00:00:00'],
    ['2016-09-26 11:00:00', '2016-09-27 10:00:00'],
    ['2016-10-09 09:00:00', '2016-10-11 20:00:00'],
    ['2016-10-29 19:00:00', '2016-11-02 16:00:00'],
    ['2016-11-26 17:00:00', '2016-11-29 04:00:00'],
    ['2016-12-06 07:00:00', '2016-12-10 04:00:00'],
    ['2016-12-14 15:00:00', '2016-12-19 04:00:00'],
]
attack_dates_test = [
    ['2017-01-16 09:00:00', '2017-01-19 06:00:00'],
    ['2017-01-30 08:00:00', '2017-02-02 00:00:00'],
    ['2017-02-09 03:00:00', '2017-02-10 09:00:00'],
    ['2017-02-12 01:00:00', '2017-02-13 07:00:00'],
    ['2017-02-24 05:00:00', '2017-02-28 08:00:00'],
    ['2017-03-10 14:00:00', '2017-03-13 21:00:00'],
    ['2017-03-20 20:00:00', '2017-03-27 01:00:00'],
]


def get_attack_dates(df: pd.DataFrame):
    if df['datetime'].iloc[0].year == 2016:
        dates = attack_dates_training2
    elif df['datetime'].iloc[0].year == 2017:
        dates = attack_dates_test
    else:
        print("NO ATTACK LABELS FOR THIS DATASET")
        return
    return dates


def label_data(df: pd.DataFrame):
    dates = get_attack_dates(df)

    df_datetime = df.set_index('datetime')
    flag = pd.Series(False, index=df_datetime.index)
    for d in dates:
        flag[d[0]:d[1]] = True
    df['flag'] = flag.values
    return df


def parse_to_df(path: str) -> pd.DataFrame:
    # Read and parse the date as datetime objects
    dateparser = lambda dates: [parse_date(d) for d in dates]
    df = pd.read_csv(path, parse_dates=['DATETIME'], date_parser=dateparser)

    # Convert all column names to lowercase and strip the whitespace
    df.columns = [c.lower().strip() for c in df.columns]

    # Compute timedelta (in hours) from start and add as a new column
    start_t = df.iloc[0]['datetime']
    df['h'] = ((df['datetime'] - start_t) / np.timedelta64(1, 'h')).astype(int)

    # Index the data on this new column
    df = df.set_index(['h'])

    return df
