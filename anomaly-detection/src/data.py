import numpy as np
import pandas as pd


def select_between_datetime(df, start, end):
    return df[
        (df['datetime'] > start) &
        (df['datetime'] <= end)
    ]


def parse_date(date):
    return pd.datetime.strptime(date, '%d/%m/%y %H')


def parse_to_df(path: str) -> pd.DataFrame:
    # Read and parse the date as datetime objects
    dateparser = lambda dates: [parse_date(d) for d in dates]
    df = pd.read_csv(path, parse_dates=['DATETIME'], date_parser=dateparser)

    # Convert all column names to lowercase and strip the whitespace
    df.columns = [c.lower().strip(" ") for c in df.columns]

    # Compute timedelta (in hours) from start and add as a new column
    start_t = df.iloc[0]['datetime']
    df['h'] = (df['datetime'] - start_t) / np.timedelta64(1, 'h')

    # Index the data on this new column
    df = df.set_index(['h'])

    return df
