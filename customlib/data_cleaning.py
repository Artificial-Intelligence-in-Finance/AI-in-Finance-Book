
import pandas as pd

import holidays

import datetime


def show_weekdays(data, date_value):
    """

    :param data: pandas data frame
    :param date_value: column of date value e.g. "date"
    :return: new dataframe with weekdays in it
    """

    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

    data[date_value] = pd.to_datetime(data[date_value])
    data['Weekday'] = data[date_value].dt.weekday
    data['Weekday'] = data['Weekday'].replace(weekdays)

    return data.groupby('Weekday').count()


def delete_holidays(data, date_value, years, other_holidays):
    us_market_holidays = []

    for date in holidays.UnitedStates(years=years).items():
        us_market_holidays.append(str(date[0]))

    us_market_holidays.extend(other_holidays)

    # drop the dates, which are market holidays
    data = data.drop(data[
                         (data[date_value].isin(us_market_holidays))].index)

    return data


def find_NANs(data):
    """
    Returns a pandas dataframe denoting the total number of NAN values and the percentage of
    the NANs in each column.
    The column names are noted on the index.

    :param data: dataframe
    :return: data frame with null/nans
    """

    # pandas series denoting features and the sum of their null values
    null_sum = data.isnull().sum()  # instantiate columns for missing data
    total = null_sum.sort_values(ascending=False)
    percent = (((null_sum / len(data.index)) * 100).round(2)).sort_values(ascending=False)

    # concatenate the columns to create the new dataframe with missing values
    df_NANs = pd.concat([total, percent], axis=1, keys=['Number of NANs', 'Percent NANs'])

    # drop rows that don't have any missing data
    df_NANs = df_NANs[(df_NANs.T != 0).any()]

    return df_NANs


def date_range(start_date, end, step):

        d = start_date
        while d < end:
            yield d
            d += step


def find_missing_dates(data, value_name, missing_days):
    """
    finds consecutive days missing
    :param data: pandas data frame
    :param value_name: data column e.g. "date"
    :param missing_days: numer of missing days to inspect e.g 2
    :return: missing dates
    """
    dates = data[value_name].to_list()

    DAY = datetime.timedelta(missing_days)
    # missing dates: a list of [start_date, end)
    missing = [(d1 + DAY, d2) for d1, d2 in zip(dates, dates[1:]) if (d2 - d1) > DAY]

    missing_dates = [d for d1, d2 in missing for d in date_range(d1, d2, DAY)]

    return missing_dates
