import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import parse_to_df

path_training_1 = '../data/BATADAL_training1.csv'
path_training_2 = '../data/BATADAL_training2.csv'
path_testing = '../data/BATADAL_test.csv'


def replace_pump_outliers(df, pumpname):
    # Define pump keys
    flow = 'f_' + pumpname
    switch = 's_' + pumpname

    # Find outliers that differ a lot from the mean (five times the standard deviation)
    mean = df[flow].mean()
    std = df[flow].std()
    threshold = mean + 5 * std
    outliers = df[flow] > threshold

    # Replace the value of the outliers with the normal mean
    normal_mean = df.loc[np.invert(outliers), flow].mean()
    df.loc[outliers, flow] = normal_mean
    # Switch value is binary and is correlated with the flow value
    df.loc[outliers, switch] = 0

    return df


def normalize(df: pd.DataFrame, replace_abnormal=True, scale=True) -> pd.DataFrame:
    # Drop att_flag and datetime column
    # Ignore errors as the testing dataset does not contain att_flag
    df = df.drop(['datetime', 'att_flag'], axis=1, errors='ignore')

    # Replace abnormalities with normal values
    if replace_abnormal:
        # Replace values of Pump 6 that are to high
        df = replace_pump_outliers(df, 'pu6')

        # Replace values of Pump 11 that are to high
        df = replace_pump_outliers(df, 'pu11')

    # Normalize the data
    if scale:
        scale = StandardScaler()
        scaled_df = scale.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
        return scaled_df

    return df


def get_magnitude(df: pd.DataFrame) -> pd.Series:
    magnitude = lambda x: (x ** 2).sum()
    return df.apply(magnitude, axis=1)


def train(df_n):
    # Normalize data and remove abnormalities
    df_n = normalize(df_n)

    # Do PCA decomposition
    pca = PCA()
    pca.fit(df_n)

    # Use k-1 principal components for reconstruction
    pcs = pca.components_[0: -1]
    var = sum(pca.explained_variance_ratio_[0:-1])

    print(f"# PCA principal components: {len(pcs)}")
    print(f"# PCA explained variance: {var}")

    # Define projections
    P = np.transpose(np.array(pcs))  # make the pcs column vectors
    C = np.matmul(P, np.transpose(P))
    I = np.eye(C.shape[0])

    def pca_project_fn(df_proj):
        # Save columns that dont have a meaning in PCA
        datetimes = None
        if 'datetime' in df_proj.columns:
            datetimes = df_proj['datetime']

        flags = None
        if 'att_flag' in df_proj.columns:
            flags = df_proj['att_flag']

        # Normalize but dont remove abnormalities
        df_proj = normalize(df_proj, replace_abnormal=False)

        # Project the data to the PCA space
        y = df_proj.to_numpy()
        # yhat = np.apply_along_axis(lambda v: np.matmul(C, v), 1, y)  # axis=1 is over each row
        yres = np.apply_along_axis(lambda v: np.matmul((I - C), v), 1, y)

        # Convert back to dataframes for easier data handling
        # df_yhat = pd.DataFrame(yhat, index=df_proj.index, columns=df_proj.columns)
        df_yres = pd.DataFrame(yres, index=df_proj.index, columns=df_proj.columns)

        # Set vector magnitudes
        df_proj['sq_mag'] = get_magnitude(df_proj)
        df_yres['sq_mag'] = get_magnitude(df_yres)

        # Restore columns
        if datetimes is not None:
            # df_yhat['datetime'] = datetimes
            df_yres['datetime'] = datetimes
            df_proj['datetime'] = datetimes
        if flags is not None:
            # df_yhat['att_flag'] = flags
            df_yres['att_flag'] = flags
            df_proj['att_flag'] = flags

        return df_proj, df_yres

    return pca_project_fn


def detect_with_pca(df_test, fn_project):
    # Project to the PCA space
    df_proj, df_yres = fn_project(df_test)

    # Threshold which gave low amount of false positives on training2
    threshold = 0.075

    # Plot normal signal
    plt.figure()
    sns.lineplot(df_test.index, df_proj['sq_mag'], label="magnitude")
    plt.title("Signal vector magnitude", fontweight="bold")
    plt.ylabel("Magnitude")
    plt.xlabel("Hours from T=0")
    plt.show()

    # Plot residual error
    plt.figure()
    sns.lineplot(df_test.index, df_yres['sq_mag'], label='residual')
    sns.lineplot(df_test.index, threshold, label='threshold', dashes=True)
    plt.title("Residual vector magnitude", fontweight="bold")
    plt.ylabel("Magnitude")
    plt.xlabel("Hours from T=0")
    plt.show()

    # Return the attacks
    attacks = df_yres['sq_mag'] > threshold
    return df_test[attacks]


def plot_abnormalities():
    # Define data
    df_n = parse_to_df(path_training_1)
    df_n = normalize(df_n, replace_abnormal=False, scale=False)

    # Abnormality notes
    # - Pump 11 at 3000-400 is at +50 instead of 0
    # - Pump 6 -- always 0 but > 20 at abnormalities

    df_n['sq_mag'] = get_magnitude(df_n)

    plt.figure()
    sns.lineplot(df_n.index, df_n['sq_mag'], label="Signal magnitude")
    plt.title("Magnitude of state vector", fontweight="bold")
    plt.xlabel("Hours from T=0")
    plt.ylabel("State vector")
    plt.show()

    plt.figure()
    sns.lineplot(df_n.index, df_n['f_pu6'], label='Flow P6')
    sns.lineplot(df_n.index, df_n['s_pu6'], label='Suction P6')
    sns.lineplot(df_n.index, df_n['f_pu11'], label='Flow P11')
    sns.lineplot(df_n.index, df_n['s_pu11'], label='Suction P11')
    plt.title("Abnormalities in training data", fontweight="bold")
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")
    plt.show()


if __name__ == '__main__':
    sns.set()
    # Plot the abnormalities in the training data which are set to the normal mean of the data
    plot_abnormalities()

    # Define data
    df_normal = parse_to_df(path_training_1)
    df_a = parse_to_df(path_training_2)
    df_test = parse_to_df(path_testing)

    # Detect anomalies with PCA
    project_fn = train(df_normal)
    trn_att = detect_with_pca(df_a, project_fn)
    tst_att = detect_with_pca(df_test, project_fn)
