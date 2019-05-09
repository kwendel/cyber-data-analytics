import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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

    # Chargeback is the positive class
    df['simple_journal'] = df['simple_journal'].map({'Chargeback': 1, 'Settled': 0})

    return df


def convert_currency(df):
    # Transform all amounts to euros -- conversion rates may be slightly off
    conversion_dict = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
    df['amount_convert'] = df.apply(lambda row: row['amount'] * conversion_dict[row['currencycode']], axis=1)

    return df


def split_data_label(df: pd.DataFrame):
    # Copy such that the original df will not be changed
    data = df.copy()

    # Save labels
    y = np.array(data['simple_journal'])

    # Drop the labels from the dataframe and take the rest as features
    data = data.drop(columns=['simple_journal'])
    X = np.array(data)

    return X, y


def smote_df(df: pd.DataFrame, test_size=0.3):
    # del df['bookingdate']
    # del df['creationdate']
    # del df['mail_id']
    # del df['ip_id']
    # del df['card_id']

    X = np.array(df.ix[:, df.columns != 'simple_journal'])
    y = np.array(df.ix[:, df.columns == 'simple_journal'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    X_train_res, y_train_res = smote(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test


def smote(X, y):
    sm = SMOTE(sampling_strategy='minority', random_state=42, n_jobs=-2)
    return sm.fit_sample(X, y.ravel())
