#imports for smoothing
from tsmoothie.smoother import *
from pykalman import KalmanFilter

# other imports
import pandas as pd
import numpy as np



def compute_FFT(data, value):
    """
    Computes the fast fourier trasnform of a given value
    :param data: pandas data fame
    :param value: column value
    :return: computed FFT
    """
    data_FT = data
    fft_ = np.fft.fft(np.asarray(data_FT[value].tolist()))
    fft_df = pd.DataFrame({'fft': fft_})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    return fft_df


def conv_smoothing(data, window_len=126, window_type='ones'):
    """
    Function to smooth the given dataframe with convolution smoothing
    This function is based on the tsmoothie library: https://github.com/cerlymarco/tsmoothie
    for more info about convolutional smoothing pls visit:
    http://www.cs.umd.edu/~djacobs/CMSC828seg/SmoothingConvolution.pdf

    Parameter
    ---------
    data = pandas dataframe
    window_len(int) = Greater than equal to 1. The length of the window used to compute the convolutions.
    window_type (str) = The type of the window used to compute the convolutions.
                        Supported types are: 'ones', 'hanning', 'hamming', 'bartlett', 'blackman'.

    """

    # operate smoothing
    smoother = ConvolutionSmoother(window_len=window_len, window_type=window_type)
    smoother.smooth(data)

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval')
    smoothed = smoother.smooth_data[0]

    data_smoothed = data
    data_smoothed['smooth'] = smoothed

    # return just the smoothed df
    data_smoothed.drop(data_smoothed.columns[[0]], axis=1, inplace=True)

    return data_smoothed


def compute_kalman(sentiment):
    """
    Smmothes the e.g. the raw sentiment score with the hlep of
    the pykalman library.
    Example input: calulate_kalman(apple_sent, 'AAPL_sentiment')

    :param Sentiment: column of the pandas data frame to be smoothed.
    :return: smoothed values
    """

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)
    df = sentiment.to_frame()
    state_means, _ = kf.filter(df)
    df['kalman'] = state_means

    return df['kalman']


def conv_smoother(sentiment):
    """
    Computes the smoothing of e.g. the raw senitment data with the help of
    the tsmoothie library.

    :param Sentiment: column to be smoothed
    :return: smoothed values
    """
# operate smoothing
    smoother = ConvolutionSmoother(window_len=126, window_type='ones')
    df = sentiment.to_frame()
    smoother.smooth(df)

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval')
    smoothed = smoother.smooth_data[0]
    df['conv_filer'] = smoothed
    return df['conv_filer']



