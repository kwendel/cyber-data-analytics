from sklearn.preprocessing import LabelEncoder
import pandas as pd

def aggregator_per_card(df):
    # Each df is a dataframe with all transactions for that card
    # Sort on creation date
    df = df.sort_values(by='creationdate', axis=0)

    # Aggregrate transaction amount per day
    df['day_sum'] = df.rolling("1d", on="creationdate")['amount'].sum()

    return df

def parse_data(input_path):
    # Read csv with pandas
    df = pd.read_csv(input_path, parse_dates=['bookingdate', 'creationdate'])

    # Drop rows with column bin=NaN, email=NaN, currencycode=Nan
    df = df[df['bin'].notna()]
    df = df[df['mail_id'].notna()]
    df = df[df['currencycode'].notna()]

    # Drop rows with refused label as fraud is unknown
    df = df[df['simple_journal'] != 'Refused']

    # Make 3-6 cvc reponses all 3
    df['cvcresponsecode'] = df['cvcresponsecode'].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3})

    # Convert NaN in categorical string column to "NaN"
    df['issuercountrycode'] = df['issuercountrycode'].map("{}".format)
    df['txvariantcode'] = df['txvariantcode'].map("{}".format)
    df['currencycode'] = df['currencycode'].map("{}".format)
    df['shoppercountrycode'] = df['shoppercountrycode'].map("{}".format)
    df['shopperinteraction'] = df['shopperinteraction'].map("{}".format)
    df['cardverificationcodesupplied'] = df['cardverificationcodesupplied'].map("{}".format)
    df['accountcode'] = df['accountcode'].map("{}".format)

    # # Map timestamps to date objects
    # df['creationdate'] = df['creationdate'].map(pd.Timestamp)

    return df


def category_to_number(df):
    enc = LabelEncoder()
    df['issuercountrycode'] = enc.fit_transform(df['issuercountrycode'])
    df['txvariantcode'] = enc.fit_transform(df['txvariantcode'])
    df['currencycode'] = enc.fit_transform(df['currencycode'])
    df['shoppercountrycode'] = enc.fit_transform(df['shoppercountrycode'])
    df['shopperinteraction'] = enc.fit_transform(df['shopperinteraction'])
    df['cardverificationcodesupplied'] = enc.fit_transform(df['cardverificationcodesupplied'])
    df['accountcode'] = enc.fit_transform(df['accountcode'])

    return df

