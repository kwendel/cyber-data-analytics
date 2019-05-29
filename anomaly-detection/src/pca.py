import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import parse_to_df

if __name__ == '__main__':
    df_n = parse_to_df('../data/BATADAL_training1.csv')
    df_a = parse_to_df('../data/BATADAL_training2.csv')

    # Select a signal
    signal = ['f_pu10', 'f_pu11', 'l_t7']
    df_n = df_n[signal]
    df_a = df_a[signal]

    # Normalize the data
    scale = StandardScaler()
    df_n = pd.DataFrame(scale.fit_transform(df_n), columns=df_n.columns)
    df_a = pd.DataFrame(scale.fit_transform(df_a), columns=df_a.columns)

    # Now do PCA on the training data
    pca = PCA()
    pca.fit(df_n)
    pcs = pca.components_

    # Define projections
    P = np.transpose(np.array(pcs))  # make the pcs column vectors
    C = P * np.transpose(P)
    I = np.eye(C.shape[0])

    # Project all of the data
    df_proj = df_a

    y = df_proj.to_numpy()

    def __proj_y(y):
        return np.matmul(C, y)


    def __proj_res(y):
        return np.matmul((I - C), y)


    yhat = np.apply_along_axis(__proj_y, 1, y)  # axis=1 is over each row
    yres = np.apply_along_axis(__proj_res, 1, y)

    # Convert back to dataframes for easier plotting
    df_yhat = pd.DataFrame(yhat, index=df_proj.index, columns=df_proj.columns)
    df_yres = pd.DataFrame(yres, index=df_proj.index, columns=df_proj.columns)

    # Plot normal signal
    plt.figure()
    # sns.lineplot(df_proj.index, df_proj['f_pu10'], label='Signal P10')
    # sns.lineplot(df_proj.index, df_yhat['f_pu10'], label='Model P10')
    sns.lineplot(df_proj.index, df_proj['f_pu11'], label='Signal P11')
    sns.lineplot(df_proj.index, df_yhat['f_pu11'], label='Model P11')
    # sns.lineplot(df_proj.index, df_proj['l_t7'], label='Signal T7')
    # sns.lineplot(df_proj.index, df_yhat['l_t7'], label='Model T7')
    plt.show()

    # Plot residual error
    plt.figure()
    # sns.lineplot(df_proj.index, df_yres['f_pu10'], label='Residual error P10')
    sns.lineplot(df_proj.index, df_yres['f_pu11'], label='Residual error P11')
    # sns.lineplot(df_proj.index, df_yres['l_t7'], label='Residual error T7')
    plt.show()
