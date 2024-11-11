'''
This classes and functions are modified to fit the purpose of this project and
based on the following library: https://github.com/cerlymarco/tsmoothie
'''

import numpy as np


def _id_nb_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the none overlapping block bootstrap
    ('nbb') strategy given the number of observations in a timeseries and
    the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    blocks = np.random.permutation(
        np.arange(0, n_obs, block_length)
    ).reshape(-1, 1)

    _id = (blocks + nexts).ravel()[:n_obs]

    return _id


def _id_mb_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the moving block bootstrap
    ('mbb') strategy given the number of observations in a timeseries
    and the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    last_block = n_obs - block_length
    blocks = np.random.randint(0, last_block, (n_blocks, 1))
    _id = (blocks + nexts).ravel()[:n_obs]

    return _id



def _id_cb_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the circular block bootstrap
    ('cbb') strategy given the number of observations in a timeseries
    and the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    last_block = n_obs
    blocks = np.random.randint(0, last_block, (n_blocks, 1))
    _id = np.mod((blocks + nexts).ravel(), n_obs)[:n_obs]

    return _id


def _id_s_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the stationary bootstrap
    ('sb') strategy given the number of observations in a timeseries
    and the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    random_block_length = np.random.poisson(block_length, n_obs)
    random_block_length[random_block_length < 3] = 3
    random_block_length[random_block_length >= n_obs] = n_obs
    random_block_length = random_block_length[random_block_length.cumsum() <= n_obs]
    residual_block = n_obs - random_block_length.sum()
    if residual_block > 0:
        random_block_length = np.append(random_block_length, residual_block)

    n_blocks = random_block_length.shape[0]
    nexts = np.zeros((n_blocks, random_block_length.max() + 1))
    nexts[np.arange(n_blocks), random_block_length] = 1
    nexts = np.flip(nexts, 1).cumsum(1).cumsum(1).ravel()
    nexts = (nexts[nexts > 1] - 2).astype(int)

    last_block = n_obs - random_block_length.max()
    blocks = np.zeros(n_obs, dtype=int)
    if last_block > 0:
        blocks = np.random.randint(0, last_block, n_blocks)
        blocks = np.repeat(blocks, random_block_length)
    _id = blocks + nexts

    return _id


'''
Define Bootstrapping class.
'''


class BootstrappingWrapper:
    """BootstrappingWrapper generates new timeseries samples using
    specific algorithms for sequences bootstrapping.
    The BootstrappingWrapper handles single timeseries. Firstly, the
    smoothing  of the received series is computed. Secondly, the residuals
    of the smoothing operation are generated and randomly partitionated
    into blocks according to the choosen bootstrapping techniques. Finally,
    the residual blocks are sampled in random order, concatenated and then
    added to the original smoothing curve in order to obtain a bootstrapped
    timeseries.
    The supported bootstrap algorithms are:
     - none overlapping block bootstrap ('nbb')
     - moving block bootstrap ('mbb')
     - circular block bootstrap ('cbb')
     - stationary bootstrap ('sb')
    Parameters
    ----------
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother
        (except for WindowWrapper). It computes the smoothing on the series
        received.
    bootstrap_type : str
        The type of algorithm used to compute the bootstrap.
        Supported types are: none overlapping block bootstrap ('nbb'),
        moving block bootstrap ('mbb'), circular block bootstrap ('cbb'),
        stationary bootstrap ('sb').
    block_length : int
        The shape of the blocks used to sample from the residuals of the
        smoothing operation and used to bootstrap new samples.
        Must be an integer in [3, timesteps).
    Attributes
    ----------
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother
        (except for WindowWrapper) that was passed to BootstrappingWrapper.
        It as the same properties and attributes of every Smoother.
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_seasonal_data
    >>> from tsmoothie.bootstrap import BootstrappingWrapper
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_seasonal_data(n_series=1, timesteps=200,
    ...                          freq=24, measure_noise=10)
    >>> bts = BootstrappingWrapper(
    ...     ConvolutionSmoother(window_len=8, window_type='ones'),
    ...     bootstrap_type='mbb', block_length=24)
    >>> bts_samples = bts.sample(data, n_samples=100)
    """

    def __init__(self, timeseries, bootstrap_type, block_length):
        self.timeseries = timeseries
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length

    def __repr__(self):
        return "<tsmoothie.bootstrap.{}>".format(self.__class__.__name__)

    def __str__(self):
        return "<tsmoothie.bootstrap.{}>".format(self.__class__.__name__)

    def sample(self, data, n_samples=1):
        """Bootstrap timeseries.
        Parameters
        ----------
        data : array-like of shape (timesteps, n)
            timeseries to bootstrap.
            The data are assumed to be in increasing time order.
        n_samples : int, default=1
            How many bootstrapped series to generate.
        Returns
        -------
        bootstrap_data : array of shape (n_samples, timesteps, n)
            Bootstrapped samples
        """

        # bootstrap_types = ['nbb', 'mbb', 'cbb', 'sb']
        bootstrap_types = ['mbb']

        if self.bootstrap_type not in bootstrap_types:
            raise ValueError(
                "'{}' is not a supported bootstrap type. "
                "Supported types are {}".format(
                    self.bootstrap_type, bootstrap_types))

        if self.block_length < 3:
            raise ValueError("block_length must be >= 3")

        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        data = np.asarray(data)

        nobs = data.shape[0]
        n_features = data.shape[1]

        if self.bootstrap_type == 'mbb':
            bootstrap_func = _id_mb_bootstrap

        bootstrap_data = np.empty((n_samples, nobs, n_features))
        for i in np.arange(n_samples):
            bootstrap_id = bootstrap_func(nobs, self.block_length)
            bootstrap_data[i] = data[bootstrap_id]

        return bootstrap_data
