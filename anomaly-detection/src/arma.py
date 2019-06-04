from math import sqrt, nan

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARMA
from tabulate import tabulate
from tqdm import tqdm

try:
    from .data import parse_to_df, label_data
    from .stats import print_stats
except (ImportError, ModuleNotFoundError):
    from data import parse_to_df, label_data
    from stats import print_stats

path_training_1 = '../data/BATADAL_training1.csv'
path_training_2 = '../data/BATADAL_training2.csv'
path_test = '../data/BATADAL_test.csv'


def plot_autocorrelation(ts: pd.Series, prefix=None):
    fig = plt.figure(figsize=(8, 8))

    # Plot autocorrelation
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(ts, lags=plot_lags, ax=ax1)
    if prefix:
        plt.title(f"{prefix} Autocorrelation")

    # Plot partial autocorrelation
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(ts, lags=plot_lags, ax=ax2)


def plot_arma(ts, order):
    # Train ARMA model
    model = ARMA(ts, order=order)
    try:
        model = model.fit(disp=-1)
    except:  # some ARMA orders don't do well and crash, we'll ignore them
        return [order, nan, nan, nan]
    # Get the residual errors
    residuals = pd.Series(model.resid)
    # and plot the distribution of the residual errors
    residuals.plot(kind='kde')
    plt.title(f"{order} Residuals distribution")
    plt.show()
    # and plot the (partial) autocorrelation of the residual errors
    plot_autocorrelation(residuals, prefix=f"{order} Residuals")
    plt.show()

    # Return statistics for the trained model
    res = [order, model.aic, sm.stats.durbin_watson(residuals), residuals.mean()]
    print(res)
    return res


def analyze_arma_order(df, sensor, order):
    # Get the sensor data
    ts = pd.Series(df[sensor], index=df.index).head(2000)
    ts.head(100).plot()
    plt.show()
    # Plot autocorrelation plots for the sensor data
    plot_autocorrelation(ts)
    plt.savefig(f"../plots/autocorrelation_{sensor}")
    plt.show()

    # Gather results from different ARMA configurations
    results = [
        plot_arma(ts, (order[0] + 0, order[1] + 0)),  # +0 AR +0 MA (the given initial order)
        plot_arma(ts, (order[0] + 0, order[1] + 1)),  # +0 AR +1 MA
        plot_arma(ts, (order[0] + 0, order[1] + 2)),  # +0 AR +2 MA
        plot_arma(ts, (order[0] + 1, order[1] + 0)),  # +1 AR +0 MA
        plot_arma(ts, (order[0] + 1, order[1] + 1)),  # +1 AR +1 MA
        plot_arma(ts, (order[0] + 1, order[1] + 2)),  # +1 AR +2 MA
        plot_arma(ts, (order[0] + 2, order[1] + 0)),  # +2 AR +0 MA
        plot_arma(ts, (order[0] + 2, order[1] + 1)),  # +2 AR +1 MA
        plot_arma(ts, (order[0] + 2, order[1] + 2)),  # +2 AR +2 MA
    ]
    print(tabulate(results, headers=['order', 'AIC', 'residuals dw', 'residuals mean']))


def analyze():
    df: pd.DataFrame = parse_to_df(path_training_1)
    # Analyze the sensors with starting parameters gathered from the autocorrelation plots
    analyze_arma_order(df, 'l_t1', (3, 0))
    analyze_arma_order(df, 'l_t4', (3, 0))
    analyze_arma_order(df, 'l_t7', (1, 0))
    analyze_arma_order(df, 'f_pu10', (1, 1))
    analyze_arma_order(df, 'f_pu7', (1, 1))
    analyze_arma_order(df, 'f_pu1', (2, 0))
    analyze_arma_order(df, 'f_pu2', (2, 1))


def test_model(df, sensor, order):
    sensor_values = df[sensor]
    start = 1416  # start at 00h the first of September
    end = len(sensor_values)
    train, test = sensor_values[0:start], sensor_values[start:end]

    # Predict every value for the given sensor step by step
    predictions = pd.Series(index=test.index)
    for t in tqdm(range(start, end), desc=sensor):  # tqdm shows a nice progressbar
        # take the last 1000 values before this point to train an ARMA model
        history = pd.Series(sensor_values[0:t].tail(1000)).reset_index(drop=True)
        # train model
        model = ARMA(history, order=order)
        model_fit = model.fit(disp=-1, method='css')
        # get next predicted value and add it to the predictions
        output = model_fit.forecast()
        yhat = output[0]
        predictions[t] = yhat

    # calculate the error for each prediction
    diff = abs(test - predictions)
    if sensor == 'f_pu1':
        # special threshold for f_pu1 otherwise there were too much false positives
        flags = diff > diff.max() / 2
    else:
        # this worked for other sensors
        flags = diff > diff.mean() + 3 * diff.std()

    # create plot
    x = df[start:end]['datetime']
    fig = plt.figure(figsize=(8, 8))
    register_matplotlib_converters()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    # plot the errors
    ax2.plot(x, diff)
    ax2.set_ylabel('error')
    # highlight errors that are above threshold
    ax2.fill_between(x, 0, 3 * diff.max(), color='r', alpha=0.3,
                     where=flags)

    # plot the sensor values
    ax1.plot(x, test, color='g')
    ax1.set_ylabel('sensor value')
    ax1.set_ylim(ymin=(test.min() - (test.max() - test.min()) / 2))

    # auto format date labels
    fig.autofmt_xdate()
    plt.title(sensor)
    plt.show()

    # print distribution characteristics of the errors
    print(diff.describe())

    # return indices of flagged hours
    return diff[flags].index


def test_models():
    df: pd.DataFrame = parse_to_df(path_training_2)
    label_data(df)

    # get indices of flagged sensor values for each sensor
    results = dict()
    results['l_t1'] = test_model(df, 'l_t1', (3, 0))
    results['l_t4'] = test_model(df, 'l_t4', (3, 1))
    results['l_t7'] = test_model(df, 'l_t7', (2, 1))
    results['f_pu10'] = test_model(df, 'f_pu10', (2, 2))
    results['f_pu7'] = test_model(df, 'f_pu7', (3, 1))
    results['f_pu1'] = test_model(df, 'f_pu1', (2, 1))

    # combine results
    total_detected = pd.Series(False, index=df.index)
    for sensor, flagged in results.items():
        total_detected[flagged] = True
        # print individual sensor values
        print(f"\n\n{sensor}")
        print_stats(df, flagged)

    # print combined results
    print('\n\nCombined stats')
    print_stats(df, total_detected[total_detected].index)


if __name__ == '__main__':
    sns.set(font_scale=1.2)
    plot_lags = 100

    # analyze arma models for sensors
    analyze()

    # run created models on the test data
    test_models()
