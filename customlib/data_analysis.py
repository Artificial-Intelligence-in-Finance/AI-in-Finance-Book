# standard imports
import pandas as pd
from statsmodels.tsa.stattools import acf, q_stat, adfuller, kpss
import matplotlib.pyplot as plt
# smoothing
from tsmoothie.smoother import *
from pykalman import KalmanFilter
from numpy.fft import *
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from .plotting import save_fig
from scipy import stats
import seaborn as sns
import statsmodels.tsa.api as tsa
from statsmodels.tsa.api import VAR

# other imports
from sklearn.metrics.cluster import normalized_mutual_info_score


def identify_outliers(data, col_name, title, plot_name):
    """
    Function to identify/depict outliers in a plot for easy identification.

    :param data: pandas data frame
    :param col_name: the name of the column to be investigated
    :param title: plot title
    :param plot_name: name to save the plot
    :return: plot
    """

    data['simple_rtn'] = data[col_name].pct_change()

    data_rolling = data[['simple_rtn']].rolling(window=20).agg(['mean', 'std'])
    data_rolling.columns = data_rolling.columns.droplevel()

    data_outliers = data.join(data_rolling)
    data_outliers.dropna(inplace=True)

    data_outliers['outlier'] = data_outliers.apply(find_outliers,
                                                   axis=1)
    outliers = data_outliers.loc[data_outliers['outlier'] == True,
                                 ['simple_rtn']]

    plt.plot(data_outliers.index, data_outliers.simple_rtn, label='Normal')
    plt.scatter(outliers.index, outliers.simple_rtn,
                color='red', label='Anomaly', s=100);
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc=0);
    save_fig(plot_name)


def find_outliers(row):
    '''
    Function for finding the outliers using the 3 sigma rule:
    https://www.investopedia.com/terms/t/three-sigma-limits.asp
    Three-sigma limits (3-sigma limits) is a statistical calculation that
    refers to data within three standard deviations from a mean.
    The row must contain the following columns/indices: simple_rtn, mean, std.

    Parameters
    ----------
    input: dataframe containing the following columns: simple return, mean and std.
    sigmas : The number (integer value) of standard deviations above or below the mean,
    which will be used to detect outliers.

    Outlier validates to True,

    '''
    return_ = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']

    if (return_ > (mu + 3 * sigma)) | (return_ < mu - (3 * sigma)):
        return True
    else:
        return False


# test for unit roots
def stationary_test(df):
    return df.apply(lambda x: f'{pd.Series(adfuller(x)).iloc[1]:.2%}').to_frame('p-value')


# test stationarity of data and print results
def print_stationarity(data, significance=0.05, asset=None, name=None):
    """
    Perform a stationary test an print the results
    :param data: pandas data frame
    :param significance: significance level
    :param name: which column in the data frame e. g. adj_close
    :return: print out of test
    """
    test = adfuller(data, autolag='BIC')
    p = {'test_statistics': round(test[0], 2),
         'p_value': round(test[1], 2),
         'n_lags': round(test[2], 2),
         'n_obs': test[3]}

    p_val = p['p_value']

    def adjust(val, length=10):
        return str(val).ljust(length)

    print(' Augmented Dickey-Fuller Test of {} on {}'.format(asset, name))
    print('=' * 50)
    print(' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(' Significance level: \t\t{}'.format(significance))
    print(' Test statistics: \t\t{}'.format(p['test_statistics']))
    print(' Set lags to: \t\t\t{}'.format(p['n_lags']))

    for key, val in test[4].items():
        print(f' Critical value {adjust(key)} \t {round(val, 2)}')

    if p_val <= significance:
        print(' P-value ist {} Reject Null Hypothesis'.format(p_val))
        print(' {} is Stationary.'.format(name))

    else:
        print(' P-value is {} Cannot reject the Null Hypothesis.'.format(p_val))
        print(' {} is Non-Stationary.'.format(name))


def test_cointegration(data, alpha=0.05):
    """
    Perform Johensen's Cointegration Test

    :param data: data frame
    :param alpha: alpha value
    :return: test results
    """

    coint_t = coint_johansen(data, -1, 5)
    sigif = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = coint_t.lr1
    cvts = coint_t.cvt[:, sigif[str(1 - alpha)]]
    #col_names = list(data.columns)
    def adjust(val, length=15):
        return str(val).ljust(length)

    # Summary
    print(' Name   \t Test results \tcoint > 95% \tsignificant ')
    print('=' * 70)
    #print('', str(col_names[0]), '\t\t', (traces[0]).round(2), '\t\t\t\t', (traces[0] > cvts[0]))
    #print('', str(col_names[1]), '\t', (traces[1]).round(2), '\t\t\t\t', (traces[1] > cvts[1]))
    for col, trace, cvt in zip(data.columns, traces, cvts):
        print(adjust(col), adjust(round(trace, 2), 15),  adjust(cvt,16), trace > cvt)


def stats_tests(series):
    """

    :param series: pandas series
    :return: if series is normal distributed
    """

    # get test statistic and p-value from the normaltest
    k2, p = stats.normaltest(series)
    alpha = 1e-3
    print('Statistic tests results')
    print('='*44)
    print(' P-Value {:35.3f}'.format(p))
    print(' Kurtosis of normal distribution: {:10.3f}'.format(stats.kurtosis(series)))
    print(' Skewness of normal distribution: {:10.3f}'.format(stats.skew(series)))
    print('-' * 44)
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print(' Data does not look Gaussian (reject H0)')
    else:
        print(' Data looks Gaussian (fail to reject H0)')


def identify_collinear(X, correlation_threshold):
    """
    Finds collinear features based on the correlation coefficient between features.
    For each pair of features with a correlation coefficient greather than `correlation_threshold`,
    only one of the pair is identified for removal.

    Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

    Parameters

    Value of the Pearson correlation cofficient for identifying correlation features

    """
    # Dictionary to hold removal operations
    correlated = {}
    correlation_threshold = correlation_threshold

    corr_matrix = X.corr()
    corr_matrix = corr_matrix

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

    # Iterate through the columns to drop to record pairs of correlated features
    for column in to_drop:
        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

        # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]

        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                          'corr_feature': corr_features,
                                          'corr_value': corr_values})

        # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index=True)

    record_collinear = record_collinear
    correlated['collinear'] = to_drop

    print('%d features with a correlation magnitude greater than %0.2f.\n' % (
    len(correlated['collinear']), correlation_threshold))
    return correlated


def corr_plot_pearson(data, title, filename):
    """
    Pearson correlation plot
    :param data: pandas data frame
    :param title: plot title
    :param filename: plots file name
    :return: correlation plots
    """
    corr = data.corr(method='pearson').round(3)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 20))
    ax.set_title(title, fontsize=28)
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmin= -1, vmax=1, center=0,
                square=True, linewidths=.8,annot=True, cbar_kws={"shrink": .6});
    save_fig(filename)


def corr_plot_spearman(data, title, filename):
    """
    Spearman correlation plot
    :param data: pandas data frame
    :param title: plot title
    :param filename: plots file name
    :return: correlation plots
    """
    corr = data.corr(method='spearman').round(3)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 20))
    ax.set_title(title, fontsize=28)
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmin= -1, vmax=1, center=0,
                square=True, linewidths=.8,annot=True, cbar_kws={"shrink": .6});
    save_fig(filename)


def indviduell_corr_pearson(data, ticker, title, filename):
    """
    Pearson correlation plot (single asset)
    :param data: pandas data frame
    :param title: plot title
    :param filename: plots file name
    :return: correlation plots
    """
    single_asset = data.loc[data['ticker'] == ticker]
    corr_plot_pearson(single_asset, title, filename)


def indviduell_corr_spearman(data, ticker, title, filename):
    """
    Spearman correlation plot (single asset)
    :param data: pandas data frame
    :param title: plot title
    :param filename: plots file name
    :return: correlation plots
    """
    single_asset = data.loc[data['ticker'] == ticker]
    corr_plot_spearman(single_asset, title, filename)


def get_tickers_with_high_correlations_rank(data, rank, corr, article_more, kal=True):
    """
    :param data: pivoted data frame
    """

    if kal:
        article = data[(data.dollar_vol_rank < rank) &
                       (data.total_article > article_more) & (data.corr_kal > corr)]
        lst = article.index.get_level_values('ticker').unique().to_list()
        cnt = article.index.get_level_values('ticker').nunique()
        print(
            'There are {} ticker which have a Kalman-correlation of {}, more than {} articles with a rank over {} the tickers are: {}.'.format(
                cnt, corr, article_more, rank, ', '.join(lst)))
    else:
        article = data[(data.dollar_vol_rank < rank) &
                       (data.total_article > article_more) & (data.corr_conv > corr)]
        lst = article.index.get_level_values('ticker').unique().to_list()
        cnt = article.index.get_level_values('ticker').nunique()
        print(
            'There are {} ticker which have a convolutional-correlation of {}, more than {} articles with a rank over {} the tickers are: {}.'.format(
                cnt, corr, article_more, rank, ', '.join(lst)))


def get_tickers_with_high_correlations(data,corr, article_no, kal=True, small=True, verbose=True):
    """
    :param data: pivoted data frame
    """

    if kal:
        if small:
            article = data[(data.total_article < article_no) & (data.corr_kal > corr)]
            lst = article.index.get_level_values('ticker').unique().to_list()
            cnt = article.index.get_level_values('ticker').nunique()

            if verbose:
                print('There are {} ticker which have a Kalman-correlation of {} and lesser than {} articles the tickers are: {}.'.format(cnt, corr, article_no, ', '.join(lst)))
            return lst
        else:
            article = data[(data.total_article > article_no) & (data.corr_kal > corr)]
            lst = article.index.get_level_values('ticker').unique().to_list()
            cnt = article.index.get_level_values('ticker').nunique()

            if verbose:
                print('There are {} ticker which have a Kalman-correlation of {} and more than {} articles the tickers are: {}.'.format(cnt, corr, article_no,  ', '.join(lst)))
            return lst

    else:
        if small:
            article = data[(data.total_article < article_no) & (data.corr_conv > corr)]
            lst = article.index.get_level_values('ticker').unique().to_list()
            cnt = article.index.get_level_values('ticker').nunique()

            if verbose:
                print('There are {} ticker which have a convolutional-correlation of {} and lesser than {} articles the tickers are: {}.'.format(cnt, corr, article_no, ', '.join(lst)))

            return lst

        else:
            article = data[(data.total_article > article_no) & (data.corr_conv > corr)]
            lst = article.index.get_level_values('ticker').unique().to_list()
            cnt = article.index.get_level_values('ticker').nunique()

            if verbose:
                print('There are {} ticker which have a convolutional-correlation of {} and more than {} articles the tickers are: {}.'.format(cnt, corr, article_no,  ', '.join(lst)))
            return lst


def seasonal_decomposition(data, filename, title, period=63):

    """

    :param data: pandas dataframe
    :param filename: filename for saving the plot
    :param title: title for the plot
    :param period: period for the seasonal decomposition
    :return: seasonal decomposed plots
    """

    components = tsa.seasonal_decompose(data, model='additive', period=period)

    ts = (data.to_frame('Original')
          .assign(Trend=components.trend)
          .assign(Seasonality=components.seasonal)
          .assign(Residual=components.resid))
    with sns.axes_style('white'):
        ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], legend=False)
        plt.suptitle(title, fontsize=24)
        sns.despine()
        plt.tight_layout()
        plt.subplots_adjust(top=.91);
        save_fig(filename)


def getWeights(d, lags):

    """

    :param d: pands data frame
    :param lags: lags, integer value i. e. 5
    :return: weights from the series expansion of the differencing operator
             for real orders d and up to lags coefficients
    """

    w = [1]
    for k in range(1, lags):
        w.append(-w[-1] * ((d - k + 1)) / k)
    w = np.array(w).reshape(-1, 1)
    return w


def cutoff_find(order,cutoff,start_lags):

    """

    :param order: order is the dearest d
    :param cutoff: is 1e-5 for us, and start lags is an initial amount of lags in which the loop will start
                    this can be set to high values in order to speed up the algo
    :param start_lags: lags where to start
    :return: the cut off for the data
    """

    val=np.inf
    lags=start_lags
    while abs(val)>cutoff:
        w=getWeights(order, lags)
        val=w[len(w)-1]
        lags+=1
    return lags


def ts_differencing_tau(series, order, tau):

    """

    :param series: pandas series
    :param order: differencing order
    :param tau: tau for differencing the sieres
    :return: return the time series resulting from (fractional) differencing
    """

    lag_cutoff=(cutoff_find(order,tau,1)) #finding lag cutoff with tau
    weights=getWeights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:]


def plotWeights(dRange, lags, numberPlots):

    """

    :param dRange:
    :param lags:
    :param numberPlots:
    :return:
    """

    weights = pd.DataFrame(np.zeros((lags, numberPlots)))
    interval = np.linspace(dRange[0], dRange[1], numberPlots)
    for i, diff_order in enumerate(interval):
        weights[i] = getWeights(diff_order, lags)
    weights.columns = [round(x, 2) for x in interval]
    fig = weights.plot(figsize=(15, 6))
    plt.legend(title='Order of differencing')
    plt.title('Lag coefficients for various orders of differencing')
    plt.xlabel('lag coefficients')
    # plt.grid(False)
    plt.show()


def ts_differencing(series, order, lag_cutoff):

    """

    :param series: pandas series
    :param order: differencing order
    :param lag_cutoff: cut off
    :return: the time series resulting from (fractional) differencing
             or real orders order up to lag_cutoff coefficients
    """

    weights = getWeights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k] * series.shift(k).fillna(0)
    return res[lag_cutoff:]


def differencing_order(df, possible_d, title, value, tau=1e-4):

    """

    :param df: pandas data frame
    :param possible_d: possible differencing value
    :param title: title for plot
    :param value: value to difference
    :param tau: tau for differencing
    :return: differencing order for time series
    """

    df_log = pd.DataFrame(np.log(df[value]), index=df.index)
    original_adf_stat_holder = [None] * len(possible_d)
    log_adf_stat_holder = [None] * len(possible_d)

    for i in range(len(possible_d)):
        original_adf_stat_holder[i] = adfuller(ts_differencing_tau(df[value], possible_d[i], tau))[1]
        log_adf_stat_holder[i] = adfuller(ts_differencing_tau(df_log[value], possible_d[i], tau))[1]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(possible_d, original_adf_stat_holder)
    ax[0].axhline(y=0.01, color='r')
    ax[0].set_title('ADF P-value by differencing order in the original series')
    ax[0].set_xlabel('diff-order')
    ax[0].set_ylabel('p-value')
    fig.suptitle(title, fontsize=24)
    ax[1].plot(possible_d, log_adf_stat_holder)
    ax[1].axhline(y=0.01, color='r')
    ax[1].set_xlabel('diff-order')
    ax[1].set_ylabel('p-value')
    ax[1].set_title('ADF P-value by differencing order in the logarithmic series');


def print_stationarity_kpps(data, significance=0.05, asset=None, name=None):
    """
    Perform a stationary test an print the results
    :param data: pandas data frame
    :param significance: significance level
    :param name: which column in the data frame e. g. adj_close
    :return: print out of test
    """
    # set regression to 'ct', to test around a trend
    test = kpss(data, regression='ct', nlags="auto")
    p = {'test_statistics': round(test[0], 2),
         'p_value': round(test[1], 2),
         'n_lags': round(test[2], 2)}

    p_val = p['p_value']

    def adjust(val, length=10):
        return str(val).ljust(length)

    print(' KPPS Test of {} on {}'.format(asset, name))
    print('=' * 50)
    print(' Null Hypothesis: The process is trend stationary.')
    print(' Significance level: \t\t{}'.format(significance))
    print(' Test statistics: \t\t{}'.format(p['test_statistics']))
    print(' Lags used: \t\t\t{}'.format(p['n_lags']))

    for key, val in test[3].items():
        print(f' Critical value {adjust(key)} \t {round(val, 2)}')

    if p_val >= significance:
        print(' P-value ist {} Reject Null Hypothesis'.format(p_val))
        print(' {} is Trend-stationary.'.format(name))

    else:
        print(' P-value is {} Cannot reject the Null Hypothesis.'.format(p_val))
        print(' {} is Non-Stationary.'.format(name))


def print_mean_variance(data, asset=None, name=None):

    """

    :param data: pandas data frame
    :param asset: name of the stock i. e. Apple
    :param name: name of the column i. e. close
    :return: the mean and variances of the data frame
    """

    X = data
    split = round(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()

    print(' Mean/Variance of {} on {}'.format(asset, name))
    print('=' * 60)
    print('Mean 1 = \t{} \tMean 2 = \t{}'.format(round(mean1, 4), round(mean2, 4)))
    print('Variance 1 = \t{} \tVariance 2 = \t{}'.format(round(var1, 4), round(var2, 4)))


def plot_cross_correlation(data, value_1, value_2, asset=None):

    """

    :param data: pandas data frame
    :param value_1: first column
    :param value_2: second column
    :param asset: stock i.e. Amazon
    :return: cross correlation plot
    """

    # input data
    model = VAR(data[[value_1, value_2]].values)

    # fit a VAR model because we need to use result.plot_acorr function
    result = model.fit(1)

    # use resid=False to plot ACF of sample data
    fig = result.plot_acorr(resid=False)

    # change titles whatever you like
    ax = fig.get_axes()
    ax[0].set_title('{} autocorr'.format(value_1.capitalize()), fontsize=14)
    ax[1].set_title('{} follow {}'.format(value_1.capitalize(), value_2.capitalize()), fontsize=14)
    ax[2].set_title('{} follow {}'.format(value_2.capitalize(), value_1.capitalize()), fontsize=14)
    ax[3].set_title('{} autocorr'.format(value_2.capitalize()), fontsize=14)
    fig.suptitle('Multivariate time series ACF plots for {}'.format(asset), fontsize=20);



def lagged_autocorr(data, lag=1):
    """
    Lag-N autocorrelation

    :param data: pandas.Series object
    :param lag: no of lags to apply before performing autocorrelation
    :return: autocorrelation as float
    """
    return data.corr(data.shift(lag))


def lagged_crosscorr(data_x, data_y, lag=0):
    """
    Lag-N cross-correlation

    :param data_x: pandas.Series object
    :param data_y: pandas.Series object
    :param lag: no of lags to apply before performing crossorrelation
    :return: crosscorrelation as float
    """


def df_derived_by_shift(df,lag=0,NON_DER=[]):

    """

    :param df: pandas data frame
    :param lag: lag for cross-correlation with shifting
    :param NON_DER: which column not to use in the data frame
    :return: pandas data frame with lagged correlations
    """

    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        #df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
        df = pd.concat([df, dfn], axis=1)
    return df

