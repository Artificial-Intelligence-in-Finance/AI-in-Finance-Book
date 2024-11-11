from scipy.stats import t
import numpy as np
import pandas as pd
from customlib import preprocessing as pre
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import statsmodels.api as sm
import statsmodels.stats.api as sms


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : int
        Variance-corrected standard deviation of the set of differences.
    """
    n = n_train + n_test
    corrected_var = (
        np.var(differences, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def get_rescaled_mape_r2(y_test, predicted, mu, sigma, n_steps_ahead):

    """
    Rescales the data and performs a MAPE and R^2 metrics of given data-

    :param y_test: testing data
    :param predicted: predicted data
    :param mu: mu of the transformed data
    :param sigma: sigma of the transformed data
    :param n_steps_ahead: prediction horizon
    :return: rescaled metrics for the data
    """

    mape_list = []
    r2_list = []
    n_steps_ahead_list = []
    for i in range(n_steps_ahead):
        n_steps_ahead_list.append(i+1)
        test = pre.reverse_scaling(y_test[:, i], mu[0], sigma[0])
        pred = pre.reverse_scaling(predicted[:, i], mu[0], sigma[0])
        mape_list.append((mean_absolute_percentage_error(test, pred)*100))
        r2_list.append(r2_score(test, pred))

    # create dataframe
    metrics_df = pd.DataFrame(list(zip(n_steps_ahead_list, mape_list, r2_list)),
                              columns=['lag', 'MAPE', '$R^2$'])

    # round and remove trailing zeros
    metrics_df['MAPE'] = metrics_df['MAPE'].round(4).astype(str)
    metrics_df['$R^2$'] = metrics_df['$R^2$'].round(4).astype(str)

    # use the lags as index
    metrics_df = metrics_df.set_index('lag')

    return metrics_df


def white_noise_test(residual, n_steps):
    """

    :param residual: models residuals
    :param n_steps: input lags
    :return: performed white noise test results
    """

    lb, p = sm.stats.diagnostic.acorr_ljungbox(residual, lags=n_steps, boxpierce=False)
    n_steps_list = []

    # get lags for dataframe
    for i in range(n_steps):
        n_steps_list.append(i + 1)

    n_steps_arr = np.array(n_steps_list)

    values = np.array(list(zip(n_steps_arr, lb, p)))

    # create dataframe
    white_noise_df = pd.DataFrame(values, columns=['Lag', 'LB-stats', 'P-value'])

    white_noise_df['Lag'].astype(int)

    # round and remove trailing zeros
    white_noise_df['Lag'] = white_noise_df['Lag'].astype(int).astype(str)
    white_noise_df['LB-stats'] = white_noise_df['LB-stats'].astype(str)
    white_noise_df['P-value'] = white_noise_df['P-value'].astype(str)

    # use the lags as index
    white_noise_df = white_noise_df.set_index('Lag')

    return white_noise_df


def get_noise_metrics_df(model, ticker, n_steps, uni=True):

    """
    Funciton to get the model noise test results of one model.

    :param model: model to get the metric from
    :param ticker: assets ticker i. e. AMZN
    :param n_steps: input lags
    :param uni: if uni- or bivriate data
    :return: noise results from test for one model.
    """

    analysis_path = '../model_analysis/'
    analysis_folder = '{}/'.format(ticker)

    if uni:
        df = pd.read_csv(analysis_path + analysis_folder + model +
                         '_white_noise_df-price-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)
    else:
        df = pd.read_csv(analysis_path + analysis_folder + model +
                         '_white_noise_df-price_conv-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)
    return df


def get_model_metrics_df(model, ticker, n_steps, uni=True):

    """
    Funciton to get the model metrics of one model.

    :param model: model to get the metric from
    :param ticker: assets ticker i. e. AMZN
    :param n_steps: input lags
    :param uni: if uni- or bivriate data
    :return: model metrics data frame
    """

    analysis_path = '../model_analysis/'
    analysis_folder = '{}/'.format(ticker)

    if uni:
        df = pd.read_csv(analysis_path + analysis_folder + model +
                         '_metrics_df-price-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)
    else:
        df = pd.read_csv(analysis_path + analysis_folder + model +
                         '_metrics_df-price_conv-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)

    return df


def get_concat_df(ticker, model_1, model_2, model_3, n_steps, model_list, noise=True, uni=True):

    """
    Function to easily concat the data frames form different models.

    :param ticker: assets i. e. TMO
    :param model_1: first model
    :param model_2: second model
    :param model_3: third model
    :param n_steps: input lags
    :param model_list: concated list of models
    :param noise: if noise test or not
    :param uni: if bi- or univariate
    :return: concated data frame
    """

    if uni:
        if noise:
            df_1 = get_noise_metrics_df(model_1, ticker, n_steps, uni=True).T
            df_2 = get_noise_metrics_df(model_2, ticker, n_steps, uni=True).T
            df_3 = get_noise_metrics_df(model_3, ticker, n_steps, uni=True).T
            df_list = [df_1[1:2], df_2[1:2], df_3[1:2]]
            df_ = pd.concat(df_list, keys=model_list,
                            axis=0).reset_index(level=1)

        else:
            df_1 = get_model_metrics_df(model_1, ticker, n_steps, uni=True).T
            df_2 = get_model_metrics_df(model_2, ticker, n_steps, uni=True).T
            df_3 = get_model_metrics_df(model_3, ticker, n_steps, uni=True).T
            df_list = [df_1, df_2, df_3]
            df_ = pd.concat(df_list, keys=model_list,
                            axis=0).reset_index(level=1)
    else:
        if noise:
            df_1 = get_noise_metrics_df(model_1, ticker, n_steps, uni=False).T
            df_2 = get_noise_metrics_df(model_2, ticker, n_steps, uni=False).T
            df_3 = get_noise_metrics_df(model_3, ticker, n_steps, uni=False).T
            df_list = [df_1[1:2], df_2[1:2], df_3[1:2]]
            df_ = pd.concat(df_list, keys=model_list,
                            axis=0).reset_index(level=1)

        else:
            df_1 = get_model_metrics_df(model_1, ticker, n_steps, uni=False).T
            df_2 = get_model_metrics_df(model_2, ticker, n_steps, uni=False).T
            df_3 = get_model_metrics_df(model_3, ticker, n_steps, uni=False).T
            df_list = [df_1, df_2, df_3]
            df_ = pd.concat(df_list, keys=model_list,
                            axis=0).reset_index(level=1)

    return df_.T


def get_eval_df(model_performance):

    """
    Transorms the eval dictionary into a pandas data frame.

    :param model_performance: models performance dictionary
    :return: metrics as data frame
    """

    metrics_evaluate = pd.DataFrame.from_dict(model_performance).T

    metrics_evaluate = metrics_evaluate.drop(columns=0)
    metrics_evaluate = metrics_evaluate.rename(columns={1: 'MSE', 2: 'MAE'})

    return metrics_evaluate


def get_clean_CV_df(df):


    """

    Gets the cross-validation resutls

    :param df: data frame
    :return: resuls
    """

    cols = df.columns.tolist()
    cols.remove('rank_test_score')
    cols.insert(0, 'rank_test_score')

    unwanted = {'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                'mean_test_score', 'std_test_score', 'split0_train_score',
                'split1_train_score', 'split2_train_score', 'mean_train_score',
                'std_train_score', 'params'}
    cols = [e for e in cols if e not in unwanted]

    df = df[cols]
    df = df.sort_values(by=['rank_test_score'])
    return df


def get_model_eval_df(model, ticker, n_steps, uni=True):

    """
    Function to get the models evaluation metrics.

    :param model: the chosen model to get the metrics
    :param ticker: the ticker of the asset i. e. TMO
    :param n_steps: input lags, integer
    :param uni: if uni- or bivariate data
    :return: models evaluation metrics as data frame
    """

    eval_path = '../model_evaluate/'
    eval_folder = '{}/'.format(ticker)

    if uni:
        df = pd.read_csv(eval_path + eval_folder + model +
                         '_evaluate_df-price-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)
    else:
        df = pd.read_csv(eval_path + eval_folder + model +
                         '_evaluate_df-price_conv-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)

    return df


def get_concat_eval_df(ticker, model_1, model_2, model_3, n_steps, uni=True):

    """
    Funciton to concat the results from the eval function from TensorFlow.

    :param ticker: asseet name i.e. TMO
    :param model_1: model one
    :param model_2: model two
    :param model_3: model three
    :param n_steps: input lags, integer value
    :param uni: if uni- or bivariate data
    :return: concated data frame of evals function
    """

    if uni:
        df_1 = get_model_eval_df(model_1, ticker, n_steps, uni=True)
        df_2 = get_model_eval_df(model_2, ticker, n_steps, uni=True)
        df_3 = get_model_eval_df(model_3, ticker, n_steps, uni=True)
        frames = [df_1, df_2, df_3]
        result = pd.concat(frames)

    else:
        df_1 = get_model_eval_df(model_1, ticker, n_steps, uni=False)
        df_2 = get_model_eval_df(model_2, ticker, n_steps, uni=False)
        df_3 = get_model_eval_df(model_3, ticker, n_steps, uni=False)
        frames = [df_1, df_2, df_3]
        result = pd.concat(frames)

    result['MSE'] = result['MSE'].round(decimals=4)
    result['MAE'] = result['MAE'].round(decimals=4)

    return result

def get_model_params_df(model, ticker, n_steps, uni=True):

    """
    Function to get the models parameter.

    :param model: the chosen model to get the params
    :param ticker: the ticker of the asset i. e. TMO
    :param n_steps: input lags, integer
    :param uni: if uni- or bivariate data
    :return: models params as data frame
    """

    params_path = '../params/'
    params_folder = '{}/'.format(ticker)

    if uni:
        df = pd.read_csv(params_path + params_folder + model +
                         '_params_df-price-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)
    else:
        df = pd.read_csv(params_path + params_folder + model +
                         '_params_df-price_conv-' + ticker + '-' + n_steps + '-n_steps.csv', index_col=0)

    return df

def get_concat_params_df(ticker, model_1, model_2, model_3, n_steps, uni=True):

    """
    Function to concat the models pramameter as a data frame.

    :param ticker: assets ticker name i.e. AMZN
    :param model_1: model one
    :param model_2: model two
    :param model_3: model three
    :param n_steps: input lags
    :param uni: if uni- or bivariate
    :return: the models parmas as data frame
    """

    if uni:
        df_1 = get_model_params_df(model_1, ticker, n_steps, uni=True).T
        df_2 = get_model_params_df(model_2, ticker, n_steps, uni=True).T
        df_3 = get_model_params_df(model_3, ticker, n_steps, uni=True).T
        frames = [df_1, df_2, df_3]
        result = pd.concat(frames)

    else:
        df_1 = get_model_params_df(model_1, ticker, n_steps, uni=False).T
        df_2 = get_model_params_df(model_2, ticker, n_steps, uni=False).T
        df_3 = get_model_params_df(model_3, ticker, n_steps, uni=False).T
        frames = [df_1, df_2, df_3]
        result = pd.concat(frames)

    result = result.drop(columns={'color', 'model','label','network'})
    result = result.replace(np.nan, '-')

    return result.T


def print_residuals_test(y_test, x_test, predicted, n_steps, n_steps_ahead, sig_level=0.05, asset=None, uni=True):

    """
    Funiton to test the residuals of a regresssion.
    :param y_test: y_test data
    :param x_test: x_test data
    :param predicted: predicted values
    :param n_steps: input lags
    :param n_steps_ahead: prediction horizon
    :param sig_level: significance level
    :param asset: asset i. e. TMO
    :param uni: if uni- or bivariate data
    :return: tst results as print-out
    """

    if uni:
        name = 'pricing data'
    else:
        name = 'pricing and sentiment data'

    print(' Homoscedastic Test of {} input lags for {} on {}'.format(n_steps, asset, name))
    print('=' * 62, '\n')

    for i in range(n_steps_ahead):
        residual = y_test[:, i] - predicted[:, i]
        test = sms.het_goldfeldquandt(residual, x_test[:, i], split=20)
        if test[1] > sig_level:
            print('Residuals of {} steps ahead for are Homoscedastic'.format(i + 1))
        else:
            print('Residuals of {} steps ahead are Heteroscedastic'.format(i + 1))

        print('F statistic: \t\t\t\t{}'.format(test[0]))
        print('P-value: \t\t\t\t{}'.format(test[1]))
        print('-' * 62, '\n')


def get_lagged_features_bootstrapping(df, n_steps, n_steps_ahead):
    """
    df: pandas DataFrame of time series to be lagged
    n_steps: number of lags, i.e. sequence length
    n_steps_ahead: forecasting horizon
    """
    lag_list = []
    for lag in range(n_steps + n_steps_ahead - 1, n_steps_ahead - 1, -1):
        lag_list.append(df.shift(lag))

    lag_array = np.dstack([i[n_steps + n_steps_ahead - 1:] for i in lag_list])

    # We swap the last two dimensions so each slice along the first dimension
    # is the same shape as the corresponding segment of the input time series
    lag_array = np.swapaxes(lag_array, 1, -1)
    return lag_array


def print_error_confidence_interval(error_dict, n_steps_ahead, ticker, key, n_steps):

    header = ' lags and {} steps prediction horizon for {} stock'.format(n_steps_ahead, ticker)

    print(' 95% Confidence interval for {} for {} input'.format(key,n_steps))
    print(header)
    print('='*54)

    print('50th percentile (median): \t\t\t{}'.format((np.median(error_dict)*100).round(4)))

    # compute 95% confidence intervals (100 - alpha)
    alpha = 5.0

    # compute lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    lower = max(0.0, np.percentile(error_dict, lower_p))
    print('2.5th percentile: \t\t\t\t{}'.format(((lower*100).round(4))))

    # compute upper percentile
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = min(1.0, np.percentile(error_dict, upper_p))
    print('97.5th percentile: \t\t\t\t{}'.format(((upper*100).round(4))))
