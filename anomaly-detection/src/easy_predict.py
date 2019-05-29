from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data import parse_to_df
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def forward_validation(train, test, p):
    # Walk forward validation
    history = [x for x in train]
    predictions = list()

    for i in range(len(test)):
        # Prediction is the historical value at p timesteps back
        yhat = history[-p]
        predictions.append(yhat)
        # Add observation
        history.append(test[i])

    return predictions


def window_regression(X, y, window, end):
    time_step = window  # This will start the rolling window when the window is defined at 0
    preds = list()
    true = list()
    reg = LinearRegression()

    while time_step < end:
        # Select previous data of size window (0-indexing so its upto time_step)
        X_trn = X.iloc[time_step - window: time_step, :]
        y_trn = y.iloc[time_step - window: time_step]
        X_tst = X.iloc[time_step, :].to_numpy().reshape(1, -1)  # reshape because it is only a single sample
        y_tst = y.iloc[time_step].to_numpy().reshape(1, -1)

        # Train LR model on the data in the window
        reg.fit(X_trn, y_trn)

        # Predict the next step and save
        yhat = reg.predict(X_tst)
        preds.append(yhat.item())
        true.append(y_tst.item())

        # Go the next timestep
        time_step += 1

    return true, preds


def find_window_value(X, y, end):
    sizes = range(1, 50)
    scores = list()

    for w in sizes:
        true, preds = window_regression(X, y, w, end)

        # Report performance
        rmse = sqrt(mean_squared_error(true, preds))
        scores.append(rmse)
        print('w=%d RMSE:%.3f' % (w, rmse))

    plt.figure()
    plt.plot(sizes, scores)
    plt.title("RMSE vs rolling window size")
    plt.xlabel("Window size")
    plt.ylabel("RMSE")
    plt.show()


def find_persistence_value(train, test):
    p_vals = range(1, 50)
    scores = list()

    for p in p_vals:
        predictions = forward_validation(train, test, p)

        # Report performance
        rmse = sqrt(mean_squared_error(test, predictions))
        scores.append(rmse)
        print('p=%d RMSE:%.3f' % (p, rmse))

    plt.figure()
    plt.plot(p_vals, scores)
    plt.title("RMSE vs persistence value")
    plt.xlabel("Persistence value")
    plt.ylabel("RMSE")
    plt.show()


def plot_predictions(x, test, predictions, title="Enter title"):
    # plot predictions vs observations
    plt.figure()
    sns.lineplot(x, test, label='Signal')
    sns.lineplot(x, predictions, label='Prediction')
    plt.title(title)
    plt.xlabel("Hours from T=0")
    plt.ylabel("Signal value")
    plt.show()


def persistence_prediction():
    df_n = parse_to_df('../data/BATADAL_training1.csv')

    # Pick sensor signal and use first 2/3 as trn, rest as test
    column = 'f_pu4'
    start = 0
    train_s = 700
    test_s = 300

    # Create splits
    series = pd.Series(df_n[column], index=df_n.index)
    X = series.values
    train, test = X[start:train_s], X[train_s:train_s + test_s]
    test_range = range(train_s, train_s + test_s)

    # Plot the RMSE for multiple persistence values
    find_persistence_value(train, test)
    # plot shows the lowest RMSE at p=24, which makes sense as the patterns can repeat every day

    # Now show the actual value with the predicted value
    predictions = forward_validation(train, test, 24)
    plot_predictions(test_range, test, predictions, title="Persistence prediction")


def regression_prediction():
    df_n = parse_to_df('../data/BATADAL_training1.csv')

    # Regression prediction
    # Previous plots showed that signals are heavenly correlated
    # This suggests that we can predict a signal based on other signals that are correlated

    # Select some signals that have correlation
    features = ['s_pu4', 'l_t3']
    label = ['f_pu4']
    X = df_n[features]
    y = df_n[label]

    end = 300
    # Find best rolling window size
    find_window_value(X, y, end)
    # Plot shows low RMSE values after w=40

    # Rolling window parameters
    window = 40
    true, preds = window_regression(X, y, window, end)

    # Report performance
    rmse = sqrt(mean_squared_error(true, preds))
    print(f"Regression RMSE={rmse}")

    plot_predictions(range(window, end), true, preds, title="Linear Regression predictions")


if __name__ == '__main__':
    sns.set()
    persistence_prediction()
    regression_prediction()

