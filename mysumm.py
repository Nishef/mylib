import numpy as np
import pandas as pd


def calculate_count(series):
    length = len(series)
    return length


def calculate_unique_count(series):
    return series.nunique()


def calculate_missing_count(series):
    return series.isnull().sum()


def calculate_percentage(value, total):
    return f"{value / total * 100:.1f}%"


def calculate_statistics(series):
    if np.issubdtype(series.dtype, np.number):
        length = calculate_count(series)
        stats = {
            "Count": length,
            "Unique_Count": calculate_unique_count(series),
            "Unique_Count (%)": calculate_percentage(
                calculate_unique_count(series), length
            ),
            "Missing": calculate_missing_count(series),
            "Missing (%)": calculate_percentage(
                calculate_missing_count(series), length
            ),
            "Skewness": series.skew(),
            "Kurtosis": series.kurt(),
            "Mean": series.mean(),
            "Minimum": series.min(),
            "Maximum": series.max(),
            "Zeros": (series == 0).sum(),
            "Zeros (%)": calculate_percentage((series == 0).sum(), length),
            "Negative": (series < 0).sum(),
            "Negative (%)": calculate_percentage((series < 0).sum(), length),
        }
        return stats
    return None


def calculate_quantile_statistics(series):
    if np.issubdtype(series.dtype, np.number):
        quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        data_range = series.max() - series.min()
        iqr = quantiles[0.75] - quantiles[0.25]

        stats = {
            "5-th percentile": quantiles[0.05],
            "Q1": quantiles[0.25],
            "Median": quantiles[0.5],
            "Q3": quantiles[0.75],
            "95-th percentile": quantiles[0.95],
            "Range": data_range,
            "Interquartile range (IQR)": iqr,
        }

        return stats
    return None


def calculate_non_numeric_statistics(series):
    if np.issubdtype(series.dtype, object):
        length = calculate_count(series)
        stats = {
            "Count": length,
            "Unique_Count": calculate_unique_count(series),
            "Unique_Count (%)": calculate_percentage(
                calculate_unique_count(series), length
            ),
            "Missing": calculate_missing_count(series),
            "Missing (%)": calculate_percentage(
                calculate_missing_count(series), length
            ),
        }
        return stats
    return None


def calculate_column_statistics(df, column_name):
    series = df[column_name]
    if np.issubdtype(series.dtype, np.number):
        stats = calculate_statistics(series)
        quantile_stats_result = calculate_quantile_statistics(series)

        if stats:
            stats.update(quantile_stats_result)
            return stats
    else:
        stats = calculate_non_numeric_statistics(series)
        return stats


def create_summary_dataframe(df):
    summary_data = {}
    for column_name in df.columns:
        stats = calculate_column_statistics(df, column_name)
        if stats:
            summary_data[column_name] = stats

    summary_df = pd.DataFrame(summary_data)
    return summary_df
