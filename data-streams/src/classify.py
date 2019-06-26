import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from data import process_file, get_infected, get_normal

data_path = '../data/capture20110818.pcap.netflow.labeled'


# %%
def label_encode(df, column):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[column])
    df[f'{column}_encoded'] = encoder.transform(df[column])


# %%
def aggregate_column(df, column, window, func):
    values = df.groupby('src')[column].rolling(window).apply(func, raw=False)
    values.index = [x[1] for x in values.index.to_flat_index()]
    return values.sort_index()


def calculate_aggregated_features(df):
    # df['unique_src_ports_60s'] = aggregate_column(df, 'src_port', '60s', lambda x: len(x.unique()))
    df['unique_dst_60s'] = aggregate_column(df, 'dst_encoded', '60s', lambda x: len(x.unique()))
    # df['unique_dst_ports_60s'] = aggregate_column(df, 'dst_port', '60s', lambda x: len(x.unique()))
    df['amount_flows_60s'] = aggregate_column(df, 'src_encoded', '60s', lambda x: len(x))
    df['amount_bytes_60s'] = aggregate_column(df, 'bytes', '60s', lambda x: x.sum())
    df['amount_packets_60s'] = aggregate_column(df, 'packets', '60s', lambda x: x.sum())


# %%
def split_data_label(df: pd.DataFrame):
    # Copy such that the original df will not be changed
    data = df.copy()

    # Save labels
    y = np.array(data['label_encoded'])

    # Drop the labels from the dataframe and take the rest as features
    data = data.drop(columns=['label_encoded'])
    X = np.array(data)

    return X, y


def tenfold_cv(clf, X, y):
    tn, fp, fn, tp = 0, 0, 0, 0
    # preds = []
    predictions = np.zeros(len(X))

    print('Running 10-fold cross-validation')
    folds = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    i = 0

    for trn_idx, tst_idx in folds.split(X, y):
        i += 1
        print(f'Fold {i}')
        # Get the data from the split
        X_train, X_test = X[trn_idx], X[tst_idx]
        y_train, y_test = y[trn_idx].ravel(), y[tst_idx].ravel()

        print('\tTrain')
        # Train and test
        clf.fit(X_train, y_train)
        print('\tPredict')
        y_pred = clf.predict(X_test)

        # preds.append((y_test, y_pred))
        predictions[tst_idx] = y_pred

        # Show the results of this fold
        conf = confusion_matrix(y_test, y_pred)

        # Sum for the total confusion matrix
        _tn, _fp, _fn, _tp = conf.ravel()
        tn += _tn
        fp += _fp
        fn += _fn
        tp += _tp

    print('TP: ' + str(tp))
    print('FP: ' + str(fp))
    print('FN: ' + str(fn))
    print('TN: ' + str(tn))
    print("\n")
    return predictions


def get_data():
    pickle = '../data/data.pickle'
    if os.path.isfile(pickle):
        print('Read data from pickle')
        df = pd.read_pickle(pickle)
    else:
        print('Filter out background noise')
        generator = process_file(data_path, lambda line: 'Background' not in line)
        df = pd.DataFrame([flow.__dict__ for flow in generator]).set_index('start').sort_index()

        print("Label encode ['src', 'dst', 'label']")
        label_encode(df, 'src')
        label_encode(df, 'dst')
        label_encode(df, 'label')

        print('Calculate aggregated features')
        calculate_aggregated_features(df)
        df.to_pickle(pickle)
    return df


# %%
def prepare_df(df: pd.DataFrame):
    print("One-hot encode ['protocol', 'flags']")
    one_hot_columns = pd.get_dummies(df[['protocol', 'flags']])
    df[one_hot_columns.columns.tolist()] = one_hot_columns
    df = df.drop(columns=['dst', 'src', 'label', 'protocol', 'flags', 'tos', 'flows'])
    # Make 'Botnet' labeled as 1 and 'LEGITIMATE as 0, instead of the other way around.
    df['label_encoded'] = df['label_encoded'].apply(lambda x: (x - 1) * -1)
    # index_icmp_none_ports = (df['protocol_ICMP'] == 1) & (df['src_port'].isin([None])) & (df['dst_port'].isin([None]))
    # df.loc[index_icmp_none_ports, 'src_port'] = 0
    # df.loc[index_icmp_none_ports, 'dst_port'] = 0
    return df


def output_csv():
    #  create csv for the 'botnet detectors comparer' from the paper
    output = pd.DataFrame(index=df_orig.index)
    output['Date flow start'] = pd.Series(df_orig.index.to_list(), index=df_orig.index)
    output['Durat'] = df_orig['duration']
    output['Prot'] = df_orig['protocol']
    #  concat src ip and port if there is a port
    output['Src IP Addr:Port'] = df_orig['src']
    not_na_idx = df_orig['src_port'].isna() == False
    output.loc[not_na_idx, 'Src IP Addr:Port'] = df_orig[not_na_idx].apply(lambda x: f"{x['src']}:{x['src_port']}",
                                                                           axis=1)
    #  arrow separator
    output[''] = np.array(['->'] * len(output))
    #  concat dst ip and port if there is a port
    output['Dst IP Addr:Port'] = df_orig['dst']
    not_na_idx = df_orig['dst_port'].isna() == False
    output.loc[not_na_idx, 'Dst IP Addr:Port'] = df_orig[not_na_idx].apply(lambda x: f"{x['dst']}:{x['dst_port']}",
                                                                           axis=1)
    output['Flags'] = df_orig['flags']
    output['Tos'] = df_orig['tos']
    output['Packets'] = df_orig['packets']
    output['Bytes'] = df_orig['bytes']
    output['Flows'] = df_orig['flows']
    output['Label'] = df_orig['label']
    output['Pred'] = preds_timed.map({0: 'LEGITIMATE', 1: 'Botnet'})
    output.to_csv('../data/output', sep='\t', index=False)


# %%
if __name__ == '__main__':
    # %%
    df_orig = get_data()
    # %%
    df = prepare_df(df_orig)
    # %%
    df = df.drop(columns=['dst_port', 'src_port', 'dst_encoded', 'src_encoded'])
    # %%
    # df = df.drop(columns=['unique_src_ports_60s', 'unique_dst_ports_60s'])
    X, y = split_data_label(df)

    # %%
    rf = RandomForestClassifier(random_state=42, max_features='sqrt', max_depth=50, n_estimators=100, n_jobs=-1)
    preds = tenfold_cv(rf, X, y)

    # %%
    preds_timed = pd.Series(preds, index=df.index)
    df['predicted'] = preds_timed

    # %%
    importances = list(rf.feature_importances_)
    feature_list = df.drop(columns=['label_encoded']).columns.to_list()
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    for pair in feature_importances:
        print('Variable: {:20} Importance: {}'.format(*pair))

    # %%
    output_csv()

    # %%
    # print confusion matrices for each IP
    df['src'] = df_orig['src']
    df['dst'] = df_orig['dst']
    infected = df[df['src'].isin(get_infected(data_path)) | df['dst'].isin(get_infected(data_path))]
    normal = df[df['src'].isin(get_normal(data_path)) | df['dst'].isin(get_normal(data_path))]
    # %%
    infected.groupby('src').apply(lambda group: confusion_matrix(group['label_encoded'], group['predicted']))
