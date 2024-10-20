import json
import os
import warnings
from typing import Dict, Optional, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_number_steps_from_freq

from .timeseries_analysis import analyze_data

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

METRICS_LIST = [  # Metrics list is the minimum set of metrics needed to use the classifier in any configuration.
    "n_data_points",
    'first_valid_timestamp',
    "n_data_points_from_first_valid",
    "n_data_points_trailing_invalid",
    "percentage_nans",
    # "percentage_zeros",
    "percentage_negatives",
    "valid_percentage_nans",
    # "valid_percentage_zeros",
    "recent_percentage_nans",
    "percentage_coverage_from_first_valid",
    # "recent_percentage_zeros",
    'mean_scale',
    'recent_mean_scale',
    'std',
    'recent_std',
    'recent_yoy_meanscale_shift',
    "lumpiness",
    "recent_trend_shift_down",
    "recent_trend_shift_up",
    "autocorrelation_daily_lag",
    "non_seasonal_volatility",
    'scaled_interdecile_range',
    'mstl_trend_strength',
    'recent_mstl_trend_strength',
    # 'mstl_seasonality_strength_hour_of_day',
    'mstl_seasonality_strength_day_of_week',
    'mstl_seasonality_strength_day_of_month',
    'mstl_seasonality_strength_day_of_year',
    'mstl_seasonality_strength_business_day',
    'residuals_right_left_tail_ratio',
    'residuals_skewness',
    'autocorrelation_1_lag',
    'daily_volatility'
]


class Classifier:
    """
    Class to implement the timeseries classifier, classifying timeseries based on historical sales metrics.
    Attributes
    ----------
    historical_data: Dict | None
        Historical data to compute the classification with.
        Might not be present if and only if
        classification has been loaded from a file and not computed directly.
    freq: str | 'D'
        Frequency in the historical data
    timeseries_id_cols: List[str] | None
        List of columns that are used to define each time series in the dataframe
    weeks_before_cold_start: int | None
        Minimum number of data points (in weeks) below which a time series will be classified as cold start
    weeks_before_luke_warm_start: int | None
        Minimum number of data points (in weeks) below which a time series will be classified as luke warm start
    timeseries_classification: Dict
        Dict of {timeseries_class_name: List[timeseries_name]} representing the classification.
        Only present after the classification has been computed / loaded.
    timeseries_metrics: Dict
        Dict of {timeseries_name: {metric_name: metric_value for metric_name in METRICS}}.
        Only present after the classification has been computed / loaded.
    df_time_series_metrics: pd.DataFrame
        DataFrame representation of the metrics for each timeseries.
    stage_1_algorithm: str
        Which algorithm to use for the stage 1 classification (determining whether the timeseries is fit for DL model).
        Can be either 'DecisionTree' or 'RandomForest'.
    threshold_ignore_intermittent: float
        Threshold of intermittency (percentage zeros) above which we do not compute all metrics for a particular time series.
    Methods
    -------
    compute_time_series_metrics(n_jobs: int | None) -> None
        Computes the time series metrics for all time series in historical data, and stores them in `df_time_series_metrics` attribute.
    compute_classification(n_jobs: int | None) -> None
        Computes the timeseries classification and saves it in `timeseries_classification` attribute.
        If `df_time_series_metrics` is None, the function will call `compute_time_series_metrics`.
    load_classification(classification_path) -> None
        Loads the classification from a saved dictionary.
    save_classification(classification_saving_path) -> None
        Saves the classification to the `classification_saving_path`.
    get_timeseries_classes() -> Dict
        Returns a list of unique timeseries classes in the data.
    get_ts_in_timeseries_class(timeseries_class) -> List
        Returns the list of timeseries names in the `timeseries_class`.
    get_ts_id_to_timeseries_classes() -> Dict
        Returns a dict of {timeseries_name: timeseries_class}, the inverse dictionary of the one saved in `timeseries_classification`.
    """
    def __init__(
        self,
        historical_data,
        freq,
        timeseries_id_cols,
        timeseries_timestamp_col,
        threshold_ignore_intermittent,
        ) -> None:
        self.historical_data = historical_data
        self.freq = freq
        self.timeseries_id_cols = timeseries_id_cols 
        self.timeseries_timestamp_col = timeseries_timestamp_col
        self.threshold_ignore_intermittent = threshold_ignore_intermittent
        
    def compute_time_series_metrics(self, n_points_recency_cutoff: Optional[int] = None, n_jobs: Optional[int] = None):
        """
        Computes the timeseries metrics based on the historical data.
        Parameters
        ----------
            n_jobs Optional[int]: number of jobs to use to multiprocess the computation of the metrics. Defaults to None.
        Returns
        -------
        """
        # Get the analysis
        assert (self.historical_data is not None), "Historical data is None but trying to compute the classification"

        print("Analyzing the data ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.df_time_series_metrics = analyze_data(
                self.historical_data,
                self.timeseries_id_cols,
                self.timeseries_timestamp_col,
                metrics_list=METRICS_LIST,
                n_jobs=n_jobs,
                threshold_ignore_intermittent=self.threshold_ignore_intermittent,
                n_points_recency_cutoff=n_points_recency_cutoff,
            )
    
    def load_classification(self, saved_classification_path):
        if self.historical_data is not None:
            warnings.warn("Classifier: historical_data is not used since use_saved_classification set to True.")
        if saved_classification_path is None:
            raise ValueError("saved_classification_path is None but use_saved_classification is True: where are we suppposed to load the classification from?")
        with open(saved_classification_path) as f:
            timeseries_classifier = json.load(f)
        if 'timeseries_metrics' in timeseries_classifier.keys():
            # Then the object contains the timeseries metrics, we don't need that. It's the result of the AFP container.
            self.timeseries_classification = timeseries_classifier['timeseries_classification']
        else:
            self.timeseries_classification = timeseries_classifier

    def save_classification(self, classification_saving_path):
        if classification_saving_path is None:
            raise ValueError("classification_saving_path is None but save_classification is True: where are we suppposed to save the classification?")
        with open(classification_saving_path, 'w') as fp:
            json.dump(self.timeseries_classification, fp)

    def get_timeseries_classes(self):
        return list(self.timeseries_classification.keys())

    def get_ts_in_timeseries_class(self, timeseries_class):
        return self.timeseries_classification[timeseries_class]

    def get_ts_id_to_timeseries_classes(self):
        return {ts_id: timeseries_class for (timeseries_class, timeseries_class_list) in self.timeseries_classification.items() for ts_id in timeseries_class_list}

    
class HeuristicClassifier(Classifier):
    def __init__(self) -> None:
        pass
    
class TreeBaseClassifier(Classifier):
    """
    Class to implement the timeseries classifier, classifying timeseries based on historical sales metrics.
    Attributes
    ----------
    historical_data: Dict | None
        Historical data to compute the classification with.
        Might not be present if and only if
        classification has been loaded from a file and not computed directly.
    freq: str | 'D'
        Frequency in the historical data
    timeseries_id_cols: List[str] | None
        List of columns that are used to define each time series in the dataframe
    weeks_before_cold_start: int | None
        Minimum number of data points (in weeks) below which a time series will be classified as cold start
    weeks_before_luke_warm_start: int | None
        Minimum number of data points (in weeks) below which a time series will be classified as luke warm start
    timeseries_classification: Dict
        Dict of {timeseries_class_name: List[timeseries_name]} representing the classification.
        Only present after the classification has been computed / loaded.
    timeseries_metrics: Dict
        Dict of {timeseries_name: {metric_name: metric_value for metric_name in METRICS}}.
        Only present after the classification has been computed / loaded.
    df_time_series_metrics: pd.DataFrame
        DataFrame representation of the metrics for each timeseries.
    stage_1_algorithm: str
        Which algorithm to use for the stage 1 classification (determining whether the timeseries is fit for DL model).
        Can be either 'DecisionTree' or 'RandomForest'.
    threshold_ignore_intermittent: float
        Threshold of intermittency (percentage zeros) above which we do not compute all metrics for a particular time series.
    Methods
    -------
    compute_time_series_metrics(n_jobs: int | None) -> None
        Computes the time series metrics for all time series in historical data, and stores them in `df_time_series_metrics` attribute.
    compute_classification(n_jobs: int | None) -> None
        Computes the timeseries classification and saves it in `timeseries_classification` attribute.
        If `df_time_series_metrics` is None, the function will call `compute_time_series_metrics`.
    load_classification(classification_path) -> None
        Loads the classification from a saved dictionary.
    save_classification(classification_saving_path) -> None
        Saves the classification to the `classification_saving_path`.
    get_timeseries_classes() -> Dict
        Returns a list of unique timeseries classes in the data.
    get_ts_in_timeseries_class(timeseries_class) -> List
        Returns the list of timeseries names in the `timeseries_class`.
    get_ts_id_to_timeseries_classes() -> Dict
        Returns a dict of {timeseries_name: timeseries_class}, the inverse dictionary of the one saved in `timeseries_classification`.
    """

    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        freq: str = 'D',
        timeseries_id_cols: Optional[List[str]] = None,
        timeseries_timestamp_col: Optional[str] = None,
        timeseries_target_col: Optional[str] = None,
        weeks_recency_cutoff: Optional[int] = None,
        weeks_before_invalid: Optional[int] = None,
        # weeks_before_cold_start: Optional[int] = None,
        stage_1_algorithm: str = 'DecisionTree',
        threshold_ignore_intermittent: Optional[float] = None
    ) -> None:
        """
        The init function saves the data and the cold_start_cut_date
        Parameters
        ----------
        historical_data: Optional[Dict] = None
            Dict of historical data (for example, 10 weeks of data)
        weeks_before_invalid: Optional[int] = None
            Minimum number of data points (in weeks) below which a time series will be classified as invalid.
            For example, if weeks_before_invalid = 1 and data frequency is 'D',
            then time series with less than 7 points
            will be classified as invalid.
        weeks_before_cold_start: Optional[int] = None
            Minimum number of data points (in weeks) below which a time series will be classified as cold start.
            For example, if weeks_before_cold_start = 1 and data frequency is 'D',
            then time series with less than 7 points
            will be classified as cold start.
            Ignored if classification is loaded from a saved point
        weeks_before_luke_warm_start: Optional[int] = None
            Minimum number of data points (in weeks) below which a time series will be classified as luke warm start.
            For example, if weeks_before_luke_warm_start = 10 and data frequency is 'D',
            then time series with less than 70 points
            will be classified as luke warm start.
            Ignored if classification is loaded from a saved point
        stage_1_algorithm: str = 'DecisionTree'
            Determines the algorithm that is used in the first stage of the classifier.
            Can be either 'DecisionTree' or 'RandomForest'.
            Defaults to 'DecisionTree'.
        threshold_ignore_intermittent: Optional[float] = None
            To speed up the classifier, we do not compute metrics for intermittent time series.
            Time series with a % of zeros higher than `threshold_ignore_intermittent` will be ignored
            in the full metrics computation, and only simple metrics will be computed for them (percentage_zeros, ...).
        Returns
        -------
        """
        # currently only support indogeneous feature
        self.historical_data = historical_data[timeseries_id_cols + [timeseries_timestamp_col, timeseries_target_col] ]
        self.freq = freq
        self.timeseries_id_cols = timeseries_id_cols
        self.timeseries_timestamp_col = timeseries_timestamp_col
        self.timeseries_target_col = timeseries_target_col
        self.weeks_recency_cutoff = weeks_recency_cutoff
        self.weeks_before_invalid = weeks_before_invalid
        # self.weeks_before_cold_start = weeks_before_cold_start
        # self.weeks_before_luke_warm_start = weeks_before_luke_warm_start
        self.timeseries_classification = {}
        self.timeseries_metrics = {}
        self.df_time_series_metrics = pd.DataFrame()
        assert stage_1_algorithm in ['DecisionTree', 'RandomForest']
        self.stage_1_algorithm = stage_1_algorithm
        self.threshold_ignore_intermittent = threshold_ignore_intermittent

    def compute_classification(self, n_jobs: Optional[int] = None):
        """
        Computes the timeseries classification, based on time series characteristics.
        If `self.compute_time_series_metrics` has not been called yet, the function will call it
        and compute the time series metrics at the beginning.
        Parameters
        ----------
            n_jobs Optional[int]: number of jobs to use to multiprocess the computation of the metrics. Defaults to None.
        Returns
        -------
        """
        # assert (self.weeks_before_cold_start is not None), "weeks_before_cold_start is None but trying to predict"
        
        freq = self.freq
        n_points_recency_cutoff = self.weeks_recency_cutoff * get_number_steps_from_freq(freq, 'W')
        
        # assert (self.weeks_before_luke_warm_start is not None), "weeks_before_luke_warm_start is None but trying to predict"
        # Get the analysis
        if self.df_time_series_metrics is None or len(self.df_time_series_metrics) == 0:
            self.compute_time_series_metrics(n_points_recency_cutoff=n_points_recency_cutoff, n_jobs=n_jobs)

        # To add the metrics metadata
        for _, r in self.df_time_series_metrics.iterrows():
            if len (self.timeseries_id_cols) > 1:
                self.timeseries_metrics[str(list(r.ts_id.values()))] = r[METRICS_LIST].to_dict()
            else:
                self.timeseries_metrics[str(r.ts_id)] = r[METRICS_LIST].to_dict()

        # Getting the frequency of the data. If historical_data is empty then set arbitrary freq since it doesn't matter


        #= next(iter(self.historical_data.values())).index.freq if (self.historical_data is not None and len(self.historical_data) > 0) else 'D'
        n_points_invalid = self.weeks_before_invalid * get_number_steps_from_freq(freq, 'W')
        # n_points_cold_start = self.weeks_before_cold_start * get_number_steps_from_freq(freq, 'W')
        # n_points_luke_warm_start = self.weeks_before_luke_warm_start * get_number_steps_from_freq(freq, 'W')

        df_metrics = self.df_time_series_metrics.copy()  # To avoid modifying the time series results

        # Compute the classification based on the analysis, i.e. implementing the tree based structure

        # Tier 1 classifications
        general_seasonal = "(mstl_seasonality_strength_day_of_week > 0.6) |  (mstl_seasonality_strength_day_of_month > 0.6) | (mstl_seasonality_strength_day_of_year > 0.9)" # & mstl_seasonality_strength_hour_of_day > 0.29
        highly_seasonal = "(mstl_seasonality_strength_day_of_week > 0.75) |  (mstl_seasonality_strength_day_of_month > 0.75) | (mstl_seasonality_strength_day_of_year > 0.95)" # & mstl_seasonality_strength_hour_of_day > 0.29
        validity = f"(n_data_points >= {n_points_invalid})"
        dropout = "(n_data_points_trailing_invalid > (0.5 * n_data_points_from_first_valid))"
        generic_cold_start = "((percentage_coverage_from_first_valid < 0.9) & (n_data_points_from_first_valid < 365 * 1.9))"
        extremely_intermittent = "(valid_percentage_nans > 0.75)"
        highly_intermittent = "((valid_percentage_nans > 0.5) and (valid_percentage_nans <= 0.75))"
        intermittent = "((valid_percentage_nans > 0.25) and (valid_percentage_nans <= 0.5))"
        low_intermittentcy = "(valid_percentage_nans <= 0.25)"
        recent_level_shift = "((recent_yoy_meanscale_shift >= 1) | (recent_yoy_meanscale_shift <= -1)) & (((recent_yoy_meanscale_shift) < 5) | ((recent_yoy_meanscale_shift) > -5) | ((mstl_trend_strength > 0.9) & (recent_mstl_trend_strength <= 0.9)))"
        recent_level_extreme_shift = "(((recent_yoy_meanscale_shift) >= 5) | ((recent_yoy_meanscale_shift) <= -5) | ((mstl_trend_strength > 0.9) & (recent_mstl_trend_strength > 0.9)))"
        
        
        print(f"n_points_invalid: {n_points_invalid}")
        # print(f"n_points_luke_warm_start: {n_points_luke_warm_start}")
        invalid_ts_query = f"not({validity})"
        
        dropout_query = f"{dropout}"
        
        generic_cold_start_query = f"{generic_cold_start}"
        
        extremely_intermittent_recent_dropout_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {extremely_intermittent} & (recent_percentage_nans > 0.9)"
        extremely_intermittent_recent_decay_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {extremely_intermittent} & (recent_percentage_nans > 0.75 & recent_percentage_nans <= 0.9)"
        extremely_intermittent_cold_start_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {extremely_intermittent} & (recent_percentage_nans <= 0.25) & not({general_seasonal})"
        extremely_intermittent_cold_start_seasonal_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {extremely_intermittent} & (recent_percentage_nans <= 0.25) & ({general_seasonal})"
        extremely_intermittent_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {extremely_intermittent} & not({extremely_intermittent_cold_start_query}) & not({extremely_intermittent_cold_start_seasonal_query}) & not({extremely_intermittent_recent_dropout_query}) & not({extremely_intermittent_recent_decay_query})"
        
        
        highly_intermittent_recent_dropout_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & (recent_percentage_nans > 0.75)"
        highly_intermittent_recent_decay_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & (recent_percentage_nans > 0.5 & recent_percentage_nans <= 0.75)"
        highly_intermittent_warm_start_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & (recent_percentage_nans <= 0.1) & not({general_seasonal})"
        highly_intermittent_warm_start_seasonal_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & (recent_percentage_nans <= 0.1) & ({general_seasonal})"
        highly_intermittent_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & not({highly_intermittent_warm_start_query}) & not({highly_intermittent_warm_start_seasonal_query}) & not({highly_intermittent_recent_decay_query}) & not({highly_intermittent_recent_dropout_query}) & not({general_seasonal})"  # Do not(x >= value) instead of x < value to get True for NaN values
        highly_intermittent_seasonal_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & not({highly_intermittent_warm_start_query}) & not({highly_intermittent_warm_start_seasonal_query}) & not({highly_intermittent_recent_decay_query}) & ({general_seasonal}) & not({recent_level_shift}) & not({recent_level_extreme_shift})"  # Do not(x >= value) instead of x < value to get True for NaN values
        highly_intermittent_seasonal_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & not({highly_intermittent_warm_start_query}) & not({highly_intermittent_warm_start_seasonal_query}) & not({highly_intermittent_recent_decay_query}) & ({general_seasonal}) & ({recent_level_shift})"  # Do not(x >= value) instead of x < value to get True for NaN values
        highly_intermittent_seasonal_extreme_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {highly_intermittent} & not({highly_intermittent_warm_start_query}) & not({highly_intermittent_warm_start_seasonal_query}) & not({highly_intermittent_recent_decay_query}) & ({general_seasonal}) & ({recent_level_extreme_shift})"  # Do not(x >= value) instead of x < value to get True for NaN values
        
        intermittent_recent_dropout_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {intermittent} & (recent_percentage_nans > 0.75)"
        intermittent_recent_decay_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {intermittent} & (recent_percentage_nans > 0.25 & recent_percentage_nans <= 0.75)"
        intermittent_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {intermittent} & not({general_seasonal}) & not({intermittent_recent_dropout_query}) & not({intermittent_recent_decay_query})"
        intermittent_seasonal_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {intermittent} & ({general_seasonal}) & not({intermittent_recent_dropout_query}) & not({intermittent_recent_decay_query}) & not({recent_level_shift}) & not({recent_level_extreme_shift})"
        intermittent_seasonal_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {intermittent} & ({general_seasonal}) & not({intermittent_recent_dropout_query}) & not({intermittent_recent_decay_query}) & ({recent_level_shift})"
        intermittent_seasonal_extreme_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {intermittent} & ({general_seasonal}) & not({intermittent_recent_dropout_query}) & not({intermittent_recent_decay_query}) & ({recent_level_extreme_shift})"

        # # Building the DL predictable query
        # if self.stage_1_algorithm == 'DecisionTree':
        #     print("Stage 1 of the classifier is a Decision Tree")
        #     dl_unpredictable_c1 = "percentage_negatives > 0.14"
        #     dl_unpredictable_c2 = "mean_scale <= 12"
        #     dl_unpredictable_c3 = "lumpiness > 1.85"
        #     dl_unpredictable_c4 = "percentage_negatives > 0.06"
        #     dl_unpredictable_query = f"({dl_unpredictable_c1}) | \
        #         (not({dl_unpredictable_c1}) & {dl_unpredictable_c2} & {dl_unpredictable_c3}) | \
        #             (not({dl_unpredictable_c1}) & {dl_unpredictable_c2} & not({dl_unpredictable_c3}) & {dl_unpredictable_c4})"
        # else:
        #     print("Stage 1 of the classifier is a Random Forest")
        #     # Load the random forest file
        #     stage_1_random_forest = joblib.load(f"{os.path.dirname(__file__)}/stage_1_random_forest_py38.joblib")  # We have 2 random forest files, will use the py38 file once we upgrade the package version

        #     df_metrics['percentage_negatives_growth'] = df_metrics['percentage_negatives']

        #     X = df_metrics[stage_1_random_forest.feature_names_in_].copy()
        #     X.fillna(0, inplace=True)  # Fillna with 0s for inference.
        #     df_metrics['random_forest_prediction'] = pd.Series(stage_1_random_forest.predict(X), index=df_metrics.index)
        #     dl_unpredictable_query = "random_forest_prediction > 0.228"
        # lumpy_query = f"not({intermittent_query}) & ({dl_unpredictable_query})"

        # Building the seasonal / non_seasonal query
        seasonal_c1 = "(recent_trend_shift_down > 0.1) | (recent_trend_shift_up > 0.1) | (recent_trend_shift_down < -0.1) | (recent_trend_shift_up < -0.1)"
        seasonal_c2 = "(autocorrelation_daily_lag > 0.3)"
        seasonal_c3 = "(non_seasonal_volatility > 1.1)"
        
        # highly_seasonal_query = f"({low_intermittentcy} & {highly_seasonal} & not({seasonal_c1} | {seasonal_c2} | {seasonal_c3})) "
        regular_seasonal_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & ({low_intermittentcy} & {general_seasonal} & not({seasonal_c1} | {seasonal_c2} | {seasonal_c3})) & not({recent_level_shift}) & not({recent_level_extreme_shift})"
        regular_seasonal_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & ({low_intermittentcy} & {general_seasonal} & not({seasonal_c1} | {seasonal_c2} | {seasonal_c3})) & ({recent_level_shift})"
        regular_seasonal_extreme_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & ({low_intermittentcy} & {general_seasonal} & not({seasonal_c1} | {seasonal_c2} | {seasonal_c3})) & ({recent_level_extreme_shift})"
        seasonal_volatile_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & ({low_intermittentcy} & {general_seasonal} & ({seasonal_c1} | {seasonal_c2} | {seasonal_c3})) "
        
        # seasonal_only = f"({seasonal_c1}) | (not({seasonal_c1}) & {seasonal_c2}) | (not({seasonal_c1}) & not({seasonal_c2}) & {seasonal_c3})"
        # seasonal_query = f"not({lumpy_query}) & ({seasonal_only})"
        # seasonal_query = f"({seasonal_only})"
        non_seasonal_recent_dropout_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {low_intermittentcy} & not({regular_seasonal_query} | {seasonal_volatile_query}) & (recent_percentage_nans > 0.75)"
        non_seasonal_recent_decay_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {low_intermittentcy} & not({regular_seasonal_query} | {seasonal_volatile_query}) & (recent_percentage_nans > 0.25 & recent_percentage_nans <= 0.75)"
        non_seasonal_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {low_intermittentcy} & not({regular_seasonal_query} | {seasonal_volatile_query}) & not({non_seasonal_recent_dropout_query}) & not({non_seasonal_recent_decay_query}) & not({recent_level_shift}) & not({recent_level_extreme_shift})"
        non_seasonal_shifted_query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {low_intermittentcy} & not({regular_seasonal_query} | {seasonal_volatile_query}) & not({non_seasonal_recent_dropout_query}) & not({non_seasonal_recent_decay_query}) & ({recent_level_shift})"
        non_seasonal_extreme_shifted__query = f"{validity} & not({dropout}) &not({generic_cold_start}) & {low_intermittentcy} & not({regular_seasonal_query} | {seasonal_volatile_query}) & not({non_seasonal_recent_dropout_query}) & not({non_seasonal_recent_decay_query}) & ({recent_level_extreme_shift})"

        timeseries_class_queries = {
            'invalid': invalid_ts_query,
            'dropout': dropout_query,
            'generic_cold_start': generic_cold_start_query,
            'extremely_intermittent_recent_dropout': extremely_intermittent_recent_dropout_query,
            'extremely_intermittent_recent_decay': extremely_intermittent_recent_decay_query,
            'extremely_intermittent_cold_start': extremely_intermittent_cold_start_query,
            'extremely_intermittent_cold_start_seasonal': extremely_intermittent_cold_start_seasonal_query,
            'extremely_intermittent': extremely_intermittent_query,
            'highly_intermittent_recent_dropout': highly_intermittent_recent_dropout_query,
            'highly_intermittent_recent_decay': highly_intermittent_recent_decay_query,
            'highly_intermittent_warm_start': highly_intermittent_warm_start_query,
            'highly_intermittent_warm_start_seasonal': highly_intermittent_warm_start_seasonal_query,
            'highly_intermittent': highly_intermittent_query,
            'highly_intermittent_seasonal': highly_intermittent_seasonal_query,
            'highly_intermittent_seasonal_shifted': highly_intermittent_seasonal_shifted_query,
            'highly_intermittent_seasonal_extreme_shifted': highly_intermittent_seasonal_extreme_shifted_query,
            'intermittent_recent_dropout': intermittent_recent_dropout_query,
            'intermittent_recent_decay': intermittent_recent_decay_query,
            'intermittent': intermittent_query,
            'intermittent_seasonal': intermittent_seasonal_query,
            'intermittent_seasonal_shifted': intermittent_seasonal_shifted_query,
            'intermittent_seasonal_extreme_shifted': intermittent_seasonal_extreme_shifted_query,
            'regular_seasonal': regular_seasonal_query,
            'regular_seasonal_shifted': regular_seasonal_shifted_query,
            'regular_seasonal_extreme_shifted': regular_seasonal_extreme_shifted_query,
            'seasonal_volatile': seasonal_volatile_query,
            'non_seasonal_recent_dropout': non_seasonal_recent_dropout_query,
            'non_seasonal_recent_decay': non_seasonal_recent_decay_query,
            'non_seasonal': non_seasonal_query,
            'non_seasonal_shifted': non_seasonal_shifted_query,
            'non_seasonal_extreme_shifted': non_seasonal_extreme_shifted__query,
        }

        df_metrics['timeseries_class'] = pd.Series(
            np.select(
                [df_metrics.eval(query) for query in timeseries_class_queries.values()],
                list(timeseries_class_queries.keys()),
                default='non_seasonal'
            ),
            index=df_metrics.index
        )

        # Construct the output dictionary
        for timeseries_class in timeseries_class_queries:
            self.timeseries_classification[timeseries_class] = list(df_metrics[df_metrics['timeseries_class'] == timeseries_class].ts_id.astype(str).unique())

    

class ClassifierHeatMap:
    def __init__(self, gc1: Classifier, gc2: Classifier) -> None:
        self.gc1 = gc1
        self.gc2 = gc2
        self.cm = None

    def compute_heat_map(self):
        """
        classifiers must be trained before
        """
        ts_id_to_bucket1 = self.gc1.get_ts_id_to_timeseries_classes()
        ts_id_to_bucket2 = self.gc2.get_ts_id_to_timeseries_classes()
        list_ts = list(set(ts_id_to_bucket1.keys()).intersection(set(ts_id_to_bucket2.keys())))
        max_n_ts = max(len(ts_id_to_bucket1.keys()), len(ts_id_to_bucket2.keys()))
        if len(list_ts) != max_n_ts:
            print(f"Warning: computing classification on {len(list_ts)} time series, but one of the classifier had {max_n_ts} time series.")
        cm = pd.DataFrame(index=self.gc1.get_timeseries_classes(), columns=self.gc2.get_timeseries_classes())
        cm.fillna(0, inplace=True)
        for ts_id in list_ts:
            cm.loc[ts_id_to_bucket1[ts_id], ts_id_to_bucket2[ts_id]] += 1
        self.cm = cm
        return self.cm

    def visualize_heat_map(
        self,
        gc_names=None,
        group_names=None,
        categories='auto',
        count=True,
        normalize=None,
        cbar=True,
        xyticks=True,
        xyplotlabels=True,
        sum_stats=True,
        figsize=None,
        cmap='Blues',
        title=None
    ):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        gc_names:      Names of the timeseries classifiers in order
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     Show the proportions for each category. Defaults to None. Options are `horizontal`, `vertical`, `total`.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                    Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                    See http://matplotlib.org/examples/color/colormaps_reference.html
        title:         Title for the heatmap. Default is None.
        '''

        if self.cm is None:
            print("Confusion Matrix is None. Running `compute_heat_map` method first.")
            self.compute_heat_map()
        cf = self.cm.to_numpy()

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if normalize is not None:
            if normalize == 'horizontal':
                percent_matrix = cf / cf.sum(axis=1, keepdims=True)
            elif normalize == 'vertical':
                percent_matrix = cf / cf.sum(axis=0, keepdims=True)
            elif normalize == 'total':
                percent_matrix = cf / np.sum(cf)
            else:
                raise ValueError("`normalize` has wrong value.")
            group_percentages = ["{0:.1%}".format(value) for value in percent_matrix.flatten()]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[1, 1] / sum(cf[:, 1])
                recall = cf[1, 1] / sum(cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize is None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks is False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(
            pd.DataFrame(percent_matrix, index=self.cm.index, columns=self.cm.columns) if normalize is not None else self.cm, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories
        )

        if xyplotlabels and gc_names is not None:
            plt.ylabel(gc_names[0])
            plt.xlabel(gc_names[1] + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)