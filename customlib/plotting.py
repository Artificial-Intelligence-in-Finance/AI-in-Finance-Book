import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# imports for technical analysis
#import talib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from pandas.plotting import lag_plot
import seaborn as sns
import statsmodels as sm
from sklearn.metrics import r2_score, mean_squared_error

import scipy.stats as stats
from customlib import preprocessing as pre


from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr


# save the images of the plots for the report
images_path = '../../images/'


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    """
    Saves the images of the generated plots as png with a resolution of 300dpi.

    :param fig_id: file name
    :param tight_layout: True or False
    :param fig_extension: file format
    :param resolution: resolution of the plot
    :return: saved version of the plot
    """

    path = os.path.join(images_path, fig_id + '.' + fig_extension)
    print('Saving figure: ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_NANs(data, title):
    """
    Function to plot NAN values.
    :param data: pandas data frame
    :param title: title for the plot
    :return: plot with results
    """
    # Find NaNs and count them, then calutalte their percentage in our dataset.
    total_nans = data.isnull().sum().sort_values(ascending=False)
    percent_nans = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    data_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])

    # plot the results
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(50), data_nans['Percent NaN'].iloc[:50].values.tolist())
    plt.xticks(np.arange(50), data_nans['Percent NaN'].iloc[:50].index.values.tolist(), rotation='90')
    plt.ylabel('NAN-values in %', fontsize=18)
    plt.title(title)
    plt.grid(alpha=0.3, axis='y');


def plot_FFT(fft_df, ylabel, title):
    """
    plots the fast fourier transformation of a time series
    :param fft_df: pandas data frame
    :param ylabel: y-axis label for the plot
    :param title: title for the plot
    :return: FFT plot
    """
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [6, 9, 12, 24, 50]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))

    plt.xlabel('Days')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend();




def plot_pacf(data, title, nlags=30):
    """
    Plot the partial autocorrelation of a time series.
    :param data: pandas data frame
    :param title: plot title
    :param nlags: number of lags
    :return: pacf plot
    """
    pacf=sm.tsa.stattools.pacf(data, nlags=nlags)
    plt.plot(pacf, label='pacf')
    plt.plot([2.58/np.sqrt(len(data))]*nlags, label='99% confidence interval (upper)')
    plt.plot([-2.58/np.sqrt(len(data))]*nlags, label='99% confidence interval (lower)')
    plt.title(title)
    plt.legend();


def plot_correlogram(x, lags=None, title=None):
    """
    This function is adapped from the book "Machine Learning for Algorithmic Trading"
    It plots a time series correlogram.

    :param x: pandas series
    :param lags: lags to use
    :param title: plot title
    :return: correlogram
    """

    lags = min(10, int(len(x)/5)) if lags is None else lags
    with sns.axes_style('whitegrid'):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0], title='Residuals')
        x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=24)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)


def lag_plots(df, title, plot_name):
    """
    Lag plots for comparison of differnt lags.
    :param df: pandas data frame
    :param plot_name: plot file name
    :return: lag plots
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    # The axis coordinates for the plots
    ax_idcs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1)]

    for lag, ax_coords in enumerate(ax_idcs, 1):
        ax_row, ax_col = ax_coords
        axis = axes[ax_row][ax_col]
        lag_plot(df, lag=lag, ax=axis)
        axis.set_title(f"Lag={lag}");

    fig.suptitle(title, fontsize=24)

    save_fig(plot_name)


def plot_heatmap(data, fig_name):
    """
    Heatmap of model metrics comparison
    :param data: model metrics
    :param fig_name: plot file name
    :return: heatmap
    """
    fig, axes = plt.subplots(ncols=3, figsize=(12,4), sharex=True, sharey=True)
    sns.heatmap(data[data.RMSE<25].RMSE.unstack(), fmt='.3f', annot=True, cmap='GnBu', ax=axes[0], cbar=False);
    sns.heatmap(data.BIC.unstack(), fmt='.2f', annot=True, cmap='GnBu', ax=axes[1], cbar=False)
    sns.heatmap(data.AIC.unstack(), fmt='.2f', annot=True, cmap='GnBu', ax=axes[2], cbar=False)
    axes[0].set_title('Root Mean Squared Error')
    axes[1].set_title('Bayesian Information Criterion')
    axes[2].set_title('Akaike Information Criterion')
    fig.tight_layout()
    save_fig(fig_name);


def prediction_plot(x_train, y_train, y_pred, title, figname):

    """

    :param x_train: x_train values from a fit
    :param y_train: y_train values from a fit
    :param y_pred: predited values from a fit
    :param title: title for the plot
    :param figname: figure name to save the file
    :return: plot with predicted and observed values
    """

    fig, ax = plt.subplots()
    ax.plot(x_train[0, :, 0], c='#fde70c', label='Inputs')
    ax.plot(range(10, 15), y_train[0, :], 'o', c='#8c8b8b', markersize=8, label='Labels')
    ax.plot(range(10, 15), y_pred[0, :, 0], 'Xr', markersize=8, label='Predictions')
    legend = ax.legend(loc='upper left', fontsize='x-large')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.title(title)
    plt.ylabel('Price ($)')
    plt.xlabel('Time (lags)');
    save_fig(figname)


def r2_error_plot(y_true, y_pred, n_steps_ahead, title):

    """

    :param y_true: observed values
    :param y_pred: predicted values
    :param n_steps_ahead: prediction horizon
    :param title: title for the plot
    :return: plot with r^2 score for each predicted lag
    """

    r2_lag = []
    plt.grid(True)
    for i in range(5):
        r2_lag.append(r2_score(y_true[:, i], y_pred[:, i]))
        #print(r2_score(y_true[:, i], y_pred[:, i]))
    x_tick = n_steps_ahead + 1
    plt.plot(range(1, 6), r2_lag, 'o', c='#fde70c',
             mec='#8c8b8b', mew='1.5', ms=14)
    #plt.ylim([0, 1])
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(1, x_tick, 1))
    plt.xlabel('Lags', fontsize=16)
    plt.ylabel(r'$R^2$', fontsize=16)
    plt.title(title, fontsize=18)
    print(r2_lag)


def error_boxplot(y_true, y_pred, n_steps_ahead, title):

    """

    param y_true: observed values
    :param y_pred: predicted values
    :param n_steps_ahead: prediction horizon
    :param title: title for the plot
    :return: boxplots with the errors for each lags separately
    """

    error_dict = {}
    for i in range(n_steps_ahead):
        error_dict[i + 1] = ((y_true[:, i] - y_pred[:, i]) ** 2).flatten()

    fig, ax = plt.subplots()

    medianprops = dict(linestyle='-', linewidth=3.5, color='#fde70c')
    flierprops = dict(markeredgecolor='#8c8b8b')
    ax.boxplot(error_dict.values(), medianprops=medianprops,
               flierprops=flierprops)
    ax.set_xticklabels(error_dict.keys())
    title = title
    plt.title(title)
    plt.xlabel('Lags')
    plt.ylabel('Value')


# Function to create a histogram and a Q-Q plot to have a look at the variable distribution
def diagnostic_plots_h_qq(df, variable, fig_title):
    """
     A function to plot a histogram and a Q-Q plot
     side by side, for a certain variable
     Parameter:
     df = your_df
     variable = your_variable_name (the variable of interest)
     fig_title = your_document_name

     """
    # define figures size
    plt.figure(figsize=(15, 6))

    # histogram
    plt.subplot(1, 2, 1)
    sns.distplot(df[variable], bins=30, color='#b1aeae')
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Quantiles')
    save_fig(fig_title);


def turkey_anscombe_plot(title, y_pred, residual, fig_title):
    """
    Creates a Turkey-Anscombe plot.
    The Tukey-Anscombe is a graphical tool: we plot the residuals r (on the y-axis) versus
    the fitted values Y_hat (on the x-axis). A reason to plot against the fitted values Y_hat
    is that the sample correlation between ri and Y_hat is always zero.
    Parameter:
    title = your_plot_title
    y_pred = your_predicted_values
    residual = your_model_residuals
    fig_title = your_figure_title

    """
    plt.figure(figsize=(12, 9))
    plt.title(title, fontsize=18)
    plt.scatter(x=y_pred, y=residual, color='black', alpha=0.3)
    plt.grid(color='#dcdcdc', linestyle='-', linewidth=.5)
    xmin = min(y_pred)
    xmax = max(y_pred)
    plt.hlines(y=0, xmin=xmin * 0.9, xmax=xmax * 1.1, color='#fde70c', linestyle='-', lw=3)
    plt.xlabel('Fitted Value')
    plt.ylabel('Residual')
    save_fig(fig_title)
    plt.show()


def error_plot(y_true, y_pred, n_steps_ahead, title):

    """

    :param y_true: observed values
    :param y_pred: predicted values
    :param n_steps_ahead: prediction horizon
    :param title: title for the plot
    :return: error plot with time line for each step ahead seperate
    """

    mse_lag = []
    for i in range(n_steps_ahead):
        mse_lag.append(mean_squared_error(y_true[:, i], y_pred[:, i]))

    plt.plot(range(1, n_steps_ahead+1), mse_lag, 'o', c='#fde70c', markeredgecolor='#8c8b8b', markersize=14)
    plt.xticks(np.arange(1, n_steps_ahead + 1, 1))
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel('Lags')
    plt.ylabel('MSE')
    plt.title(title);
    print(mse_lag)
    #save_fig(figname)


def prediction_vs_observed_plot(compare, params, n_steps_ahead, y_train, y_test, title, trainplot=True):

    """

    :param compare: the models keys to compare i. e. 'lstm'
    :param params: models saved parameter
    :param n_steps_ahead: prediction horizon i. e. 5
    :param y_train: train set
    :param y_test: test set
    :param title: title of the plot
    :param trainplot: if it should be a plot for the training or testing performance
    :return: predicted vs. observed plot for each step ahean
    """


    fig = plt.figure(figsize=(10, 20))
    # if training set plot different
    if trainplot:
        x_vals = np.arange(len(y_train))
    else:
        x_vals = len(y_train) + np.arange(len(y_test))

    # iterate over the lags
    for i in range(n_steps_ahead):
        plt.subplot(n_steps_ahead, 1, i+1)
        # compare each value in each lag
        for key in compare:
            # if training set plot different
            if trainplot:
                y_vals = params[key]['pred_train'][:, i]
                label = params[key]['label'] + \
                    ' (train MSE: %.2e)' % params[key]['MSE_train steps ahead: '+str(i+1)]
            else:
                y_vals = params[key]['pred_test'][:, i]
                label = params[key]['label'] + \
                    ' (test MSE: %.2e)' % params[key]['MSE_test steps ahead:' + str(i+1)]

            plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=2)

        # if training set plot different
        if trainplot:
            plt.plot(x_vals, y_train[:, i], c="k", label="Observed", lw=2)

        else:
            plt.plot(x_vals, y_test[:, i],
                     c="k", label="Observed", lw=2)

        plt.xlim(x_vals.min(), x_vals.max())
        plt.xlabel('Time (ticks)', fontsize=14)
        plt.ylabel('$\hat{Y}$', rotation=0, fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.title(f'{i+1}' + title, fontsize=16)

    fig.tight_layout(pad=3.0)


def error_plot_timeline(compare, params, n_steps_ahead, y_train, y_test, title, trainplot=True):

    """

    :param compare: the models keys to compare i. e. 'lstm'
    :param params: models saved parameter
    :param n_steps_ahead: prediction horizon i. e. 5
    :param y_train: train set
    :param y_test: test set
    :param title: title of the plot
    :param trainplot: if it should be a plot for the training or testing performance
    :return: plot of the error timeline for each step ahean
    """

    fig = plt.figure(figsize=(10, 20))

    if trainplot:
        x_vals = np.arange(len(y_train))

    else:
        x_vals = len(y_train) + np.arange(len(y_test))

    for i in range(n_steps_ahead):
        plt.subplot(n_steps_ahead, 1, i + 1)
        for key in compare:
            if trainplot:
                y_vals = params[key]['pred_train'][:, i] - y_train[:, i]
                label = params[key]['label'] + ' (train MSE: %.2e)' % params[key][
                    'MSE_train steps ahead: ' + str(i + 1)]

            else:
                y_vals = params[key]['pred_test'][:, i] - y_test[:, i]
                label = params[key]['label'] + \
                    ' (test MSE: %.2e)' % params[key]['MSE_test steps ahead:' + str(i + 1)]
            plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=2)

        plt.axhline(0, linewidth=0.8)
        plt.xlim(x_vals.min(), x_vals.max())
        plt.xlabel('Time (ticks)', fontsize=16)
        plt.ylabel('$\hat{Y}-Y$', fontsize=16)
        plt.legend(loc="best", fontsize=12)
        plt.title(f'{i+1} ' + title, fontsize=16)

    fig.tight_layout(pad=3.0)


def plot_CV_histogram(results_df, key, ticker, n_steps, n_steps_ahead):

    """

    :param results_df: results from cross-validation as pandas data frame
    :param key: the network to evaluate i. e. 'rnn'
    :param ticker: the ticker which was evaluated i. e. TMO
    :param n_steps: input lags, integer i. e. 10
    :param n_steps_ahead: prediction horizon, integer,  i. e. 5
    :param uni: if univariate or bivariate (with sentiment)
    :return: the cross validation performance plot
    """


    title = 'Univariate CV train and test scores of {} steps ahead of {} for {}'.format(n_steps, key, ticker)
    file_name = 'Histogramm_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)


    plt.hist(results_df['mean_test_score'], label='mean test score')
    plt.hist(results_df['mean_train_score'], label='mean train score')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc=0, fontsize=16)
    save_fig(file_name)


def plot_post_distribution(t_post, ticker, n_steps, n_steps_ahead):


    """

    :param t_post: possterier distribution
    :param ticker: name of asset i. e. TMO
    :param n_steps: input lags, integer, i. e. 10
    :param n_steps_ahead: prediction horizon, integer, i. e. 5
    :param uni: if uni or bivariate data, i.e. with sentiment
    :return: plot for models cross valdidation ditribution perforamnce
    """

    title = 'Posterior distribution for univariate of {} '.format(ticker)
    file_name = 'post_dist_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)

    x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 100)
    plt.plot(x, t_post.pdf(x))
    plt.fill_between(x, t_post.pdf(x), 0, facecolor='#fde70c', alpha=.4)
    plt.ylabel('Probability density')
    plt.xlabel(r'Mean difference ($\mu$)')
    plt.title(title);

    save_fig(file_name)


def stock_price_plot(mu, sigma, n_steps, n_steps_ahead, y_true, y_pred, key, ticker, uni):

    """

    :param mu: mu of scaled data
    :param sigma: sigma of scaled data
    :param n_steps: input lags
    :param n_steps_ahead: prediction horizon
    :param y_true: true values
    :param y_pred: predicted values
    :param key: key of model i. e. 'lstm'
    :param ticker: ticker i. e. TMO
    :param uni: if uni- or bivariate data
    :return: plot with actual vs. predicted
    """



    fig = plt.figure(figsize=(5, 12))

    if uni:
        file_name = 'stock_price_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)
    else:
        file_name = 'stock_price_plot_price_conv_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)

    for i in range(n_steps_ahead):
        plt.subplot(n_steps_ahead, 1, i + 1)
        # just use the mu and sigma of the closing price
        test = pre.reverse_scaling(y_true[:, i], mu[0], sigma[0])
        pred = pre.reverse_scaling(y_pred[:, i], mu[0], sigma[0])

        title = 'Observed vs. predicted with {} lags and {} steps ahead of {} for {}'.format(
            n_steps, i + 1, key, ticker)
        plt.plot(test, label='Observed', linewidth=1.6)
        plt.plot(pred, label='Predicted', linewidth=1.6)
        plt.legend(loc=0, fontsize=14)
        plt.xlabel('Time (observations)', fontsize=8)
        plt.ylabel('Price in $', fontsize=8)
        plt.legend(loc="best", fontsize=8)
        plt.title(title, fontsize=12)

    save_fig(file_name)


def plot_train_val_loss(history, key, ticker, n_steps, n_steps_ahead, name=True):

    """

    :param history: training history
    :param key: key of the funciton e. g. 'rnn'
    :param ticker: asset e. g. TMO
    :param n_steps: input lags, integer, e. g. 10
    :param n_steps_ahead: prediction horizon, e. .g. 5 (integer)
    :param uni: if uni-or bivariate data
    :return: learing curve plot (training and val loss)
    """


    if name:
        file_name = '{}_train_val_los_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(key,ticker, n_steps, n_steps_ahead)

    title = 'Model loss of {} steps ahead of {} for {}'.format(n_steps, key, ticker)

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.xlabel('Epoch',fontsize=14)
    plt.legend(['Train', 'Val'], loc=0, fontsize=16)

    save_fig(file_name)


def scaled_observed_vs_predicted_plot(data, key, ticker, data_train, data_val,
                                      data_test, mu, sigma, params, n_steps, n_steps_ahead):

    """

    :param data1: the splitted dataframe from the cut-off (special COVID-19 data cut-off) (first plit)
    :param data2: the splitted dataframe from the cut-off (special COVID-19 data cut-off) (second split)
    :param key: key for the model, i. e. 'rnn'
    :param ticker: assets ticker i. e. TMO
    :param data_train: training data split
    :param data_val: validaiton data spit
    :param data_test: testing data split
    :param mu: mu of the transformed data
    :param sigma: sigmma of the transformed data
    :param params: models funcitons params
    :param n_steps: input lags, integer, i.e. 10
    :param n_steps_ahead: prediction horizon, integer, i. e. 5
    :param uni: if bi- or univariate data to handle
    :return: scaled prediction plot
    """


    for i in range(n_steps_ahead):
        fig = plt.figure(figsize=(10, 20))
        plt.subplot(n_steps_ahead, 1, i + 1)

        train = pre.reverse_scaling(
            params[key]['pred_train'][:, i], mu[0], sigma[0])
        val = pre.reverse_scaling(
            params[key]['pred_val'][:, i], mu[0], sigma[0])
        test = pre.reverse_scaling(
            params[key]['pred_test'][:, i], mu[0], sigma[0])
        title = 'Observed vs. predicted with {} lags and {} steps ahead of {} for {}'.format(
            n_steps, i + 1, key, ticker)

        look_back = (n_steps + n_steps_ahead)-1
        data3 = pd.concat([data])

        file_name = '{}_stock_price_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(key, ticker, n_steps,
                                                                                          i+1)

        plt.plot(data.close, 'k', label="observed", lw=1.5)
        plt.plot(data_train.index[look_back:], train,
                 c='#767a76', lw=1.5, label="train")
        plt.plot(data_val.index[look_back:], val,
                 lw=1.5, c='#12a506', label="val")
        plt.plot(data_test.index[look_back:], test,
                 lw=1.5, c='#fde70c', label="test")
        plt.fill_between(
            [data_train.index[-1], data_val.index[0]], 0, np.max(data3.close))
        plt.xlabel('Date')
        plt.ylabel('Price in $')
        plt.title(title, fontsize=16)
        plt.legend(fontsize=16)

        save_fig(file_name)


def rediduals_plot(predicted, residual, y_test, title):

    """

    :param predicted: predicted values
    :param residual: residuals of the prediction
    :param y_test: y_test data set for the model
    :param title: title of the plot
    :return: residuals plot with Turkey-Anscombe and predicted vs. actual plots
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

    # turkey anscombe
    axes[0].scatter(x=predicted, y=residual, color='black', alpha=0.3)
    axes[0].grid(color='#dcdcdc', linestyle='-', linewidth=.5)
    xmin = min(predicted)
    xmax = max(predicted)
    axes[0].hlines(y=0, xmin=xmin * 0.9, xmax=xmax * 1.1, color='#fde70c', linestyle='-', lw=3)
    axes[0].set_xlabel('Fitted Value')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Turkey-Anscombe Plot', fontsize=18)

    # actual vs. predicted
    axes[1].plot(y_test, predicted, 'ok', alpha=0.3)
    p1 = max(max(predicted), max(y_test))
    p2 = min(min(predicted), min(y_test))
    axes[1].plot([p1, p2], [p1, p2], color='#fde70c')
    axes[1].set_title('Actual vs. Predicted Plot', fontsize=18)
    axes[1].set_xlabel('Fitted Values')
    axes[1].set_ylabel('Actual Values')
    plt.suptitle(title, fontsize=22)
    fig.tight_layout(pad=1.0);


def plot_metrics_eval(eval_df, title, fig_name, number=True):
    df_rename = eval_df.copy()
    df_renamed = eval_df.copy()

    rename_ = {df_rename.index[0]: 'RNN train',
              df_rename.index[1]: 'RNN val',
              df_rename.index[2]: 'RNN test',
              df_rename.index[3]: 'LSTM train',
              df_rename.index[4]: 'LSTM val',
              df_rename.index[5]: 'LSTM test',
              df_rename.index[6]: 'alpha_t_RIM train',
              df_rename.index[7]: 'alpha_t-RIM val',
              df_rename.index[8]: 'alpha_t-RIM test'
              }
    df_renamed.rename(
        index=rename_, inplace=True)

    ax = df_renamed.plot.bar(rot=45)
    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='large')
    ax.legend(fontsize=16)
    ax.set_ylabel('Metric value')
    plt.title(title)

    if number:
        # annotate
        plt.bar_label(ax.containers[0], label_type='edge', fontsize=12)
        plt.bar_label(ax.containers[1], label_type='edge', fontsize=12);
    else:
        pass

    save_fig(fig_name)


def error_distribution_plot(error_dict, title, fig_name):
    multiplied_list = [element * 100 for element in error_dict]
    std_dev = 'Std-Dev: ' + str(np.std(multiplied_list).round(4))
    mu = 'Mean: ' + str(np.mean(multiplied_list).round(4))
    sns.distplot(multiplied_list)
    plt.xlabel('MAPE')

    plt.text((min(multiplied_list)-0.2), 0.30, std_dev, fontsize=12)
    plt.text((min(multiplied_list)-0.2), 0.15, mu, fontsize=12)
    plt.title(title);

    save_fig(fig_name)


def heatmap_plot(dataset, titel, image_name):
    #colormap = plt.cm.RdBu
    colormap = plt.cm.Greens
    plt.figure(figsize=(12, 12))
    plt.title(titel, y=1.05, size=16)

    mask = np.zeros_like(dataset.corr())
    mask[np.triu_indices_from(mask)] = True

    svm = sns.heatmap(dataset.corr().round(2), mask=mask, linewidths=0.1, vmax=1.0,
                      square=True, cmap=colormap, linecolor='white', annot=True)

    save_fig(fig_id=image_name)
