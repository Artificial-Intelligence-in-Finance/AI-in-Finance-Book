import logging

import matplotlib.pyplot as plt
import numpy as np
from kats.consts import TimeSeriesData, TimeSeriesChangePoint
from kats.detectors.detector import Detector
from scipy.stats import norm, zscore
from kats.detectors.trend_mk import MKDetector
from customdslib.plotting import save_fig

from typing import List, Tuple


"""
All the classes and funciton here are a modificaiotn of the kats functions and classes from kats, facebook
to fit the needs for this project.
"""


class RobustStatMetadata:
    def __init__(self, index: int, metric: float) -> None:
        self._metric = metric
        self._index = index

    @property
    def metric(self):
        return self._metric

    @property
    def index(self):
        return self._index


class RobustStatDetector(Detector):

    def __init__(self, data: TimeSeriesData) -> None:
        super(RobustStatDetector, self).__init__(data=data)
        if not self.data.is_univariate():
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self,
                p_value_cutoff: float = 1e-2,
                smoothing_window_size: int = 5,
                comparison_window: int = -2
                 ) -> List[Tuple[TimeSeriesChangePoint, RobustStatMetadata]]:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()
        data_df = data_df.set_index(time_col_name)

        df_ = data_df.loc[:, val_col_name].rolling(window=smoothing_window_size)
        df_ = (
            # Smooth
            df_.mean()
            .fillna(method="bfill")
            # Make spikes standout
            .diff(comparison_window)
            .fillna(0)
        )

        # pyre-fixme[16]: Module `stats` has no attribute `zscore`.
        y_zscores = zscore(df_)
        p_values = norm.sf(np.abs(y_zscores))
        ind = np.where(p_values < p_value_cutoff)[0]

        if len(ind) == 0:
            return []  # empty list for no change points

        change_points = []

        prev_idx = -1
        for idx in ind:
            if prev_idx != -1 and (idx - prev_idx) < smoothing_window_size:
                continue

            prev_idx = idx
            cp = TimeSeriesChangePoint(
                start_time=data_df.index.values[idx],
                end_time=data_df.index.values[idx],
                confidence=1 - p_values[idx])
            metadata = RobustStatMetadata(index=idx, metric=float(df_.iloc[idx]))

            change_points.append((cp, metadata))

        return change_points

    def plot(self,
            change_points: List[Tuple[TimeSeriesChangePoint, RobustStatMetadata]]
             ) -> None:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        plt.plot(data_df[time_col_name], data_df[val_col_name])

        if len(change_points) == 0:
            logging.warning('No change points detected!')

        for change in change_points:
            plt.axvline(x=change[0].start_time, color='red', alpha=0.25)

        plt.show()


from typing import Any, Dict, List, Optional, Tuple, Union

class RobustStatDetector_(RobustStatDetector):
    def __init__(self, data: TimeSeriesData) -> None:
        super(RobustStatDetector, self).__init__(data=data)
        if not self.data.is_univariate():
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

        def plot(self,
            change_points: List[Tuple[TimeSeriesChangePoint, RobustStatMetadata]]
             ) -> None:
            time_col_name = self.data.time.name
            val_col_name = self.data.value.name

            data_df = self.data.to_dataframe()

            plt.plot(data_df[time_col_name], data_df[val_col_name])

            if len(change_points) == 0:
                logging.warning('No change points detected!')

            for change in change_points:
                plt.axvline(x=change[0].start_time, color='red', alpha=0.25)

            plt.show()




class MKDetector_(MKDetector):
    def __init__(
        self,
        data: Optional[TimeSeriesData] = None,
        threshold: float = 0.8,
        alpha: float = 0.05,
        multivariate: bool = False,
    ) -> None:
        # pyre-fixme[6]: Expected `TimeSeriesData` for 1st param but got
        #  `Optional[TimeSeriesData]`.
        super(MKDetector_, self).__init__(data=data)

        self.threshold = threshold
        self.alpha = alpha
        self.multivariate = multivariate
        self.__subtype__ = "trend_detector"

        # Assume univariate but multivariate data is detected
        if self.data is not None:
            if not self.data.is_univariate() and not self.multivariate:
                logging.warning("Using multivariate MK test for univariate data.")
                self.multivariate = True
            # Assume multivariate but univariate data is detected
            elif self.data.is_univariate() and self.multivariate:
                logging.warning("Using univariate MK test on multivariate data.")

    def plot_(self,
        change_points: List[Tuple[TimeSeriesChangePoint, RobustStatMetadata]]
            ) -> None:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        plt.plot(data_df[time_col_name], data_df[val_col_name])

        if len(change_points) == 0:
            logging.warning('No change points detected!')

        for change in change_points:
            plt.axvline(x=change[0].start_time, color='red', alpha=0.1)

        plt.show()


def robust_stat_detector(data, ticker, close=True):
    if close:
        tsd = TimeSeriesData(data[['time', 'close']])
        title = 'RobostStatDetector change points of closing price of {}'.format(ticker)
        fig_name = '{}_robust_stat_detector_plot_price'.format(ticker)

    else:
        tsd = TimeSeriesData(data[['time', 'conv_filter']])
        title = 'RobostStatDetector change points of sentiment score of {}'.format(ticker)
        fig_name = '{}_robust_stat_detector_plot_sent'.format(ticker)


    detector = RobustStatDetector(tsd)
    change_points = detector.detector()

    plt.xticks(rotation=45, fontsize=14)
    plt.ylabel('Price in $', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.title(title, fontsize=24)
    detector.plot(change_points);
    save_fig(fig_name)

    print('\nNumber of total detected change points: ', len(change_points))

    print('\n')
    print('Detected change points of {} :'.format(ticker))
    for i in range(len(change_points)):
        change_point, metadata = change_points[i]
        print(change_point)




def mk_stat_detector(data, ticker, window, down=True, close=True):
    if close:
        tsd = TimeSeriesData(data[['time', 'close']])
        if down:
            title = 'Downwards trend detection of closing price of {}'.format(ticker)
            fig_name = '{}_mk_down_plot_price'.format(ticker)
        else:
            title = 'Upwards trend detection of closing price of {}'.format(ticker)
            fig_name = '{}_mk_up_plot_price'.format(ticker)
    else:
        tsd = TimeSeriesData(data[['time', 'conv_filter']])
        if down:
            title = 'Downwards trend detection of cof sentiment score of {}'.format(ticker)
            fig_name = '{}_mk_down_plot_sent'.format(ticker)
        else:
            title = 'Upwards trend detection of cof sentiment score of {}'.format(ticker)
            fig_name = '{}_mk_up_plot_sent'.format(ticker)



    # init the detector
    detector = MKDetector_(data=tsd, threshold=.9)

    if down:
        # run detector
        detected_time_points = detector.detector(direction='down', window_size=window, freq='weekly')
        trend = 'upwards'
    else:
        # run detector
        detected_time_points = detector.detector(direction='up', window_size=window, freq='weekly')
        trend = 'downwards'


    plt.xticks(rotation=45, fontsize=14)
    plt.ylabel('Price in $', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.title(title, fontsize=24)
    # plot the results
    detector.plot_(detected_time_points);
    save_fig(fig_name)

    print('\nNumber of total detected change points: ', len(detected_time_points))

    print('\n')
    print('Detected {} trend change points of {} :'.format(trend, ticker))
    for i in range(len(detected_time_points)):
        change_point, metadata = detected_time_points[i]
        print(change_point)


