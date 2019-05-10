from pandas import read_pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .classify import tenfold_cv, combine_preds
from .preprocess import parse_data, convert_currency, onehot, delete_features

def get_df_aggregated():
    return read_pickle('../data/df_aggregated.pkl')


def read_csv():
    input_path = '../data/data.csv'

    ## PREPROCESSING STEPS
    # Load the data
    df = parse_data(input_path)

    return df


def prep():
    df = read_csv()
    # Convert all the amounts to euro
    df = convert_currency(df)
    # Map the categorial features to numbers
    # df = category_to_number(df)

    df = delete_features(df)

    df = onehot(df)

    return df


def run_white_box(X, y, df=None, threshold=0.5):
    # Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', max_features='sqrt', random_state=42, max_depth=50)
    tenfold_cv(dt, X, y, smote_data=True, threshold=threshold, df=df)


def run_black_box(X, y):
    jobs = -2

    # Make predictions with three XGB classifiers
    gradient = XGBClassifier(n_estimators=100, random_state=42, n_jobs=jobs, max_depth=5, learning_rate=0.1,
                             objective='binary:logistic')
    preds1 = tenfold_cv(gradient, X, y, smote_data=True, predict=False)

    gradient = XGBClassifier(n_estimators=100, random_state=42, n_jobs=jobs, max_depth=5, learning_rate=0.05,
                             objective='binary:logistic')
    preds2 = tenfold_cv(gradient, X, y, smote_data=True, predict=False)

    gradient = XGBClassifier(n_estimators=100, random_state=42, n_jobs=jobs, max_depth=5, learning_rate=0.15,
                             objective='binary:logistic')
    preds3 = tenfold_cv(gradient, X, y, smote_data=True, predict=False)

    # Combine predictions -- thresholds are hard code for the black-box algorithm scenario
    combine_preds(preds1, preds2, preds3)
