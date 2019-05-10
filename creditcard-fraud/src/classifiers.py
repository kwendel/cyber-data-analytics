from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .classify import tenfold_cv
from .preprocess import parse_data, convert_currency, category_to_number, split_data_label, onehot


def prep():
    input_path = '../data/data.csv'

    ## PREPROCESSING STEPS
    # Load the data
    df = parse_data(input_path)
    # Convert all the amounts to euro
    df = convert_currency(df)
    # Map the categorial features to numbers
    # df = category_to_number(df)

    # Delete features that are not useful
    # no temporal information is used
    del df['creationdate']
    del df['bookingdate']
    # no aggregation will be done so ids can be removed
    del df['mail_id']
    del df['ip_id']
    del df['card_id']
    # id must be removed as it contains hidden information:
    # the original data was first sorted based on chargeback and settled and then ids were assigned
    del df['txid']
    # Remove the amount in the original currency
    del df['amount']

    # Remove features that are not usefull
    # del df['bin']
    del df['cardverificationcodesupplied']
    del df['issuercountrycode']
    del df['shoppercountrycode']
    del df['accountcode']

    df = onehot(df)

    return df


def run_white_box(X, y, threshold=0.5):
    # Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', max_features='sqrt', random_state=42, max_depth=50)
    tenfold_cv(dt, X, y, smote_data=True, threshold=threshold)


def run_black_box():
    X, y = prep()
    jobs = -2

    # Adaboost
    base = DecisionTreeClassifier(criterion='gini', max_features='sqrt', random_state=42)
    ada = AdaBoostClassifier(base_estimator=base)
    tenfold_cv(ada, X, y, smote_data=True)

    # Random Forest
    random = RandomForestClassifier(n_estimators=50, max_features='sqrt', n_jobs=jobs, random_state=42)
    tenfold_cv(random, X, y, smote_data=True)

    # eXtreme Gradient Boosting with Trees
    gradient = XGBClassifier(n_estimators=50, random_state=42, n_jobs=jobs)
    tenfold_cv(gradient, X, y, smote_data=True)

    # Neural Network
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
    tenfold_cv(nn, X, y, smote_data=True)
