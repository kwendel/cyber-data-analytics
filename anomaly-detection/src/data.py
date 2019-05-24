import numpy as np
import pandas as pd


def parse_to_df(path: str) -> pd.DataFrame:
    # Read and parse the date as datetime objects
    dateparser = lambda dates: [pd.datetime.strptime(d, '%d/%m/%y %H') for d in dates]
    df = pd.read_csv(path, parse_dates=['DATETIME'], date_parser=dateparser)

    # Convert all column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Compute timedelta (in hours) from start and add as a new column
    start_t = df.iloc[0]['datetime']
    df['h'] = (df['datetime'] - start_t) / np.timedelta64(1, 'h')

    # Index the data on this new column
    df = df.set_index(['h'])

    return df
