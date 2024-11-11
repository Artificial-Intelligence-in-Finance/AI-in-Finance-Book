import numpy as np
import pandas


def prepare_data(df, train_weight=0.8, log_=True):


    """

    :param df: pandas data fame
    :param train_weight: training weight to split the data frame
    :param log_: if log should be applied or not
    :return: splitted data
    """

    if log_:
        df = np.log(df)
    # split into train and test sets
    split = int(len(df)*train_weight)
    df_train, df_test = df.iloc[:split], df.iloc[split:]

    # get training mu and sigma
    mu, sigma = df_train.mean(), df_train.std()

    df_train = (df_train - mu) / sigma
    # scale test with sigma and mu of the train set
    df_test = (df_test - mu) / sigma

    return mu, sigma, df_train,  df_test


def prepare_data_val_no_split(df, train_weight=0.7, test_weight=0.9, log_=True):

    """

    :param df: pandas data frame to split
    :param train_weight: training weight
    :param test_weight: testing weight
    :param log_: if log should be applied to the data frame
    :return: splitted data
    """

    if log_:
        df = np.log(df)
    # split into train, val and test sets
    n = len(df)
    df_train = df[0:int(n * train_weight)]
    df_val = df[int(n * train_weight):int(n * test_weight)]
    df_test = df[int(n * test_weight):]

    # get training mu and sigma
    mu, sigma = df_train.mean(), df_train.std()

    df_train = (df_train - mu) / sigma
    # scale val and test with sigma and mu of the train set
    df_val = (df_val - mu) / sigma
    df_test = (df_test - mu) / sigma

    return mu, sigma, df_train, df_val, df_test


def prepare_data_val(df1, df2, weight=0.6, log_=True):

    """
    This is a special fucntion for the cutted data frames to exclude the COVID 19 data.

    :param df1: first data frame
    :param df2: second data frame (after the cut-off)
    :param weight: weight to split the data
    :param log_: if log should be applied or not
    :return: splitted data frame
    """

    if log_:
        df1 = np.log(df1)
        df2 = np.log(df2)
    # split into train, val and test sets
    n = len(df2)
    df_train = df1
    split = int(len(df2)*weight)
    df_val, df_test = df2.iloc[:split], df2.iloc[split:]

    # get training mu and sigma
    mu, sigma = df_train.mean(), df_train.std()

    df_train = (df_train - mu) / sigma
    # scale val and test with sigma and mu of the train set
    df_val = (df_val - mu) / sigma
    df_test = (df_test - mu) / sigma

    return mu, sigma, df_train, df_val, df_test


def reverse_scaling(data, mu, sigma, exp_=True):

    """

    :param data: arrays of the data to transform back
    :param mu: mu values from the split
    :param sigma: sigma values from the split
    :param exp_: if log was applied to the data before and therefore expo needs to be applied in turn.
    :return:
    """

    data_ = data * sigma + mu
    if exp_:
        data_reversed = np.exp(data_)
    else:
        data_reversed = data_

    return data_reversed


def get_lagged_features(df, n_steps, n_steps_ahead):
    """
    df: pandas DataFrame of time series to be lagged
    n_steps: number of lags, i.e. sequence length
    n_steps_ahead: forecasting horizon
    """
    lag_list = []
    for lag in range(n_steps + n_steps_ahead - 1, n_steps_ahead - 1, -1):
        lag_list.append(df.shift(lag))
    lag_array = np.dstack([i[n_steps+n_steps_ahead-1:] for i in lag_list])
    # We swap the last two dimensions so each slice along the first dimension
    # is the same shape as the corresponding segment of the input time series
    lag_array = np.swapaxes(lag_array, 1, -1)
    return lag_array


def get_lagged_dataset(data, n_steps, n_steps_ahead):

    """

    :param data: data frame
    :param n_steps: input lags
    :param n_steps_ahead: prediction horizon
    :return: lagged dataset
    """

    x_df = get_lagged_features(data, n_steps, n_steps_ahead)
    y_df = get_lagged_features(data.iloc[n_steps:, :1], n_steps_ahead, 0)
    return x_df, y_df



