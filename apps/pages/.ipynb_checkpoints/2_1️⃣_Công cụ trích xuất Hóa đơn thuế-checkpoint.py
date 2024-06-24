import streamlit as st
import sys
import os

sys.path.append("/home/jovyan/work/scripts")
sys.path.append("/home/jovyan/work/config")
sys.path.append("/home/jovyan/work/dung_da")
sys.path.append("/home/jovyan/work/dung_da/Project_TradeDictionary/apps")
from Streamlit_utils import *
from Streamlit_io import *


import math
import numpy as np
import pandas as pd
import operator
import warnings
import sqlite3
import random
from collections import Counter
import json
from db_utils import get_cursor
from datetime import datetime, time
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import sqlalchemy as sa
import pytz
import glob

from scipy.optimize import minimize
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.cbook import boxplot_stats
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from tqdm import tqdm_notebook
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib_venn import venn2, venn2_circles
import matplotlib.mlab as mlab
import mplcursors
import mpldatacursor
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns

sns.set_theme(style="dark")
sns.set_style("dark")
plt.style.use("Solarize_Light2")

from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import HoverTool
from bokeh.models import BoxAnnotation
from bokeh.palettes import Category10

# from bokeh.io import output_notebook
# output_notebook();
from hand_book.utils import print_full
from PIL import Image
import json
from statsmodels.api import OLS
import time


st.set_page_config(
    page_title="Marketing Campaign Dictionary", page_icon="💡", layout="wide"
)

st.markdown(
    """
## :blue[📗 Marketing Campaign Dictionary]
"""
)

static_load = read_static_path()
role_setting = read_role_path()
role_this_page = ["admin", "ns", "ss", "su", "brand"]


def main():
    covid_range = [("2020-01-01", "2020-04-30"), ("2021-05-01", "2021-10-31")]
    role_level_I = st.session_state["role"]
    role_level_II = st.session_state["role_level_II"]
    setting_load = read_widgets_setting(role_level_II)

    setting_info = f"{setting_load['brand_selected']} - Region: {setting_load['region_selected']} & SBU: {setting_load['sbu_selected']}"
    with st.sidebar:
        st.markdown(
            f"""
        ###### 🎯 Cấu hình hiện tại:
        ###### - Region: {setting_load['region_selected']} & SBU: {setting_load['sbu_selected']}
        ###### - Brand : {setting_load['brand_selected']}
        """
        )

    scheme_actual_data = [
        "ShiftDate",
        "SAP_code",
        "TC_Actual",
        "Sale_Actual",
        "Bill_Actual",
    ]
    
    ## Daily run etl_restaurant_daily_sale.py to update data from 2024-now
    list_df_tc_actual = []
    for year_i in static_load["year_template_path"]:
        df_tc_actual_cached_i = pd.read_csv(
            base_path + static_load["restaurant_daily_sale_template_path"].replace("'query_time'", year_i), sep="\t"
        )[scheme_actual_data]
        list_df_tc_actual.append(df_tc_actual_cached_i)

    df_tc_actual = pd.concat(list_df_tc_actual, axis=0)
    df_tc_actual = df_tc_actual[["ShiftDate", "SAP_code", "TC_Actual", "Sale_Actual"]]

    # df_tc_actual["ShiftDate"] = df_tc_actual["ShiftDate"].apply(
    #     lambda x: datetime.strptime(x, "%Y-%m-%d")
    # )
    
    ## Join Restaurants information
    df_res_info = pd.read_csv(base_path + static_load["restaurant_info_path"], sep="\t")
    df_res_info["SAP_code"] = df_res_info["SAP_code"].astype(str)
    df_tc_actual["SAP_code"] = df_tc_actual["SAP_code"].astype(str)
    df_tc_actual_info = pd.merge(df_tc_actual, df_res_info, how="left", on="SAP_code")

    try:
        # Location
        Brand_name = setting_load["brand_selected"]
        Region_name = setting_load["region_selected"]
        Restaurant_name = setting_load["restaurant_selected"]
        SBU_name = setting_load["sbu_selected"]
        lower_time_series = setting_load["lower_time_series"]
        upper_time_series = setting_load["upper_time_series"]
        avg_aggregation_selected = setting_load["avg_aggregation_selected"]
        TC_Sale_type = setting_load["tc_sale_type"]
        is_avg_aggregation = (
            True if avg_aggregation_selected == "Average Store" else False
        )

        covid_range = [("2020-01-01", "2020-04-30"), ("2021-05-01", "2021-10-31")]
        df_tc_brand = df_tc_actual_info[
            (df_tc_actual_info["BrandName"] == Brand_name)
            & (df_tc_actual_info["RegionName"] == Region_name)
            & (df_tc_actual_info["SBU"] == SBU_name)
        ]
        df_tc_brand = df_tc_brand[
            (df_tc_brand["ShiftDate"] >= lower_time_series)
            & (df_tc_brand["ShiftDate"] <= upper_time_series)
        ]
        # Data Type
        data_col = "TC_Actual" if TC_Sale_type == "TC" else "Sale_Actual"
        if is_avg_aggregation == False:
            df_tc_agg = (
                df_tc_brand.groupby(["ShiftDate", "BrandName"])[data_col]
                .sum()
                .reset_index()
            )
        else:
            df_tc_agg = (
                df_tc_brand.groupby(["ShiftDate", "BrandName"])[data_col]
                .mean()
                .reset_index()
            )
            df_tc_agg[data_col] = df_tc_agg[data_col].apply(lambda x: round(x))
        # df_tc_agg["ShiftDate"] = df_tc_agg["ShiftDate"].apply(
        #     lambda x: datetime.strftime(x, "%Y-%m-%d")
        # )
        df_tc_agg = df_tc_agg[["ShiftDate", data_col]]
        df_tc_agg.fillna(0, inplace=True)
    except Exception as e:
        import traceback

        st.error(error_setting_message + str(e), icon="🚨")
        is_error = True
        st.error(f"**Copy this error code for DS:** {traceback.format_exc()}")

    df_decom = df_tc_agg.copy()
    df_decom["ShiftDate"] = pd.to_datetime(df_decom["ShiftDate"]).dt.date
    df_decom = df_decom.set_index("ShiftDate").asfreq("D")
    df_decom.fillna(0, inplace=True)
    model = "ad"
    result_decom = seasonal_decompose(df_decom, model=model, two_sided=False)
    observed = result_decom.observed
    trend = result_decom.trend
    seasonal = result_decom.seasonal
    residual = result_decom.resid

    component_type = "Trend"
    long_term_range_day = 30
    short_term_range_day = 7
    number_of_K = 5
    start_time_series = setting_load["lower_time_series"]
    current_event_day = setting_load["upper_time_series"]

    if component_type == "Trend":
        df_kse = trend
        col_name = "trend"
    elif component_type == "Residual":
        df_kse = residual
        col_name = "residual"
    else:
        df_kse = df_decom
        col_name = "TC_Actual"
    df_kse = df_kse[df_kse.index >= start_time_series]
    df_kse.fillna(0, inplace=True)
    st.markdown(
        """
    > ###### :blue[1.1 K-SE | Methodology & Processing]:
    """
    )
    with st.expander("👉 Mở rộng để xem chi tiết", expanded=False):
        col_image_1, col_image_2 = st.columns([1, 1])
        with col_image_1:
            image1 = Image.open(base_path + "apps/picture/kse_description.png")
            st.image(
                image1,
                caption="Hình mô tả thuật toán K-Same Event",
                use_column_width="never",
                width=500,
            )
        with col_image_2:
            image1 = Image.open(
                base_path + "apps/picture/time_series_decomposition_description.PNG"
            )
            st.image(
                image1,
                caption="Hình mô tả bóc tách Trend cho tính toán khoảng cách sự kiện",
                use_column_width="never",
                width=420,
            )
        st.markdown(
            f"""
            - :red[Idea:] Sử dụng thuật toán theo ý tưởng của mô hình phân loại K-Nearest Neighbor. Tại thời điểm hiện tại, tách thành sự kiện được biểu diễn dưới dạng ma trận **TxF** trong đó :
                + T: chiều thời gian (là số nguyên được tách theo 2 loại long term và short term)
                + F: chiều features (gồm feature về :blue[Component Type = **Trend**] và các feature timeseries về đặc điểm như weekend, holiday, weather,...)

            - Để phù hợp với mục tiêu bài toán của Golden Gate, chia thành 2 sự kiện : Long Term (TL) và Short Term (TS) để đảm bảo cover được đặc điểm trend của sự kiện hiện tại theo thời gian dài (:blue[Long Term Range = {long_term_range_day}]) và thời gian ngắn (:blue[Short Term Range = {short_term_range_day}]). Khi đó DL sẽ biểu diễn dưới dạng 2 ma trận **TLxF** và **TSxF**
            
            - Trong việc bóc tách :blue[Trend], DL dạng chuỗi thời gian tách được thành 3 yếu tố:
                + **:blue[Trend:]** (Xu hướng) thể hiện thiên hướng của dữ liệu, thường trong một khoảng thời gian sẽ thể hiện rõ ràng tính tăng hay giảm, mặc dù DL thô là rất phức tạp. Việc phân tích Trend cho ta DL mịn hơn và không bị ảnh hưởng bởi các yếu tố cố định và bất thường còn lại: Seasonality & Irregular, được mô tả dưới đây.
                + **:blue[Seasonality:]** (Tính mùa) là một khái niệm quan trọng trong DL chuỗi thời gian, thường các DL như TC/Sales sẽ có tính lặp lại nếu theo dõi trong thời gian đủ dài, mô hình thống kê sẽ tự động phát hiện tính mùa trong DL, có thể theo tuần, tháng, năm hoặc một khung thời gian cố định tùy theo tính chất của DL.
                + **:blue[Irregular/Residual:]** (Dư nhiễu) đây là thành phần dư lại sau khi lọc bỏ tính Trend và Seasonality, Residual được hiểu là các pattern bất thường xuất hiện trong DL, các mô hình thống kê sẽ tập trung vào phân tích thành phần này, còn được gọi là Statistical Noise việc giữ hay loại bỏ sự nhiễu sẽ tùy thuộc vào bài toán cần giải quyết. 

            - :red[Stepping:] thuật toán KSE: 
                + B1. Tracking toàn bộ sự kiện từ 1 mốc trong quá khứ (:blue[Start = {start_time_series}]) đến hiện tại theo Window *T*, tổng hợp feature **Fi** tương ứng
                Ta có N bộ sự kiện (N hàng xóm) **TxFi** với các label về MKT Trigger cũng như MKT KPI theo DL Trade MKT
                + B2. Tổng hợp feature cho sự kiện hiện tại **TxF**
                + B3. Tính toán khoảng cách giữa các ma trận: *distance(**TxF**, **TxFi**)* được một tập khoảng cách S (Euclidean Distance)
    """
        )
        st.latex(
            """
        distance(TF, TFi) = \sqrt{\sum_{p=1}^{P} \sum_{q=1}^{Q}(TF_{pq} - TFi_{pq})^2}
        """
        )
        st.markdown(
            f"""
            -
                + B4. Pick top k (:blue[K Events = {number_of_K}]) sự kiện gần với sự kiện hiện tại nhất, ta được các mốc Event: **E=[E1,E2, ..., Ek]** 
                    + Chạy theo sự kiện Long term và Short term
                    + Lấy điểm trung bình Long term và Short term để đưa ra mốc top k Event **E**
                + B5. Dựa vào label của **E** đánh giá với features và DL MKT đưa ra các insights về action phù hợp cho sự kiện hiện tại"""
        )

        end_event_day = current_event_day
        start_long_term_event_date = datetime.strptime(
            end_event_day, "%Y-%m-%d"
        ) - timedelta(days=long_term_range_day)
        start_short_term_event_date = datetime.strptime(
            end_event_day, "%Y-%m-%d"
        ) - timedelta(days=short_term_range_day)

        start_long_term_event_day = datetime.strftime(
            start_long_term_event_date, "%Y-%m-%d"
        )
        start_short_term_event_day = datetime.strftime(
            start_short_term_event_date, "%Y-%m-%d"
        )

        st.markdown(
            f"""
        ###### Current Event to KSE Model:
        - :red[Long Term:] **{start_long_term_event_day} - {end_event_day}**
        - :red[Short Term:] **{start_short_term_event_day} - {end_event_day}**
        """
        )

        df_kse_long_term = df_kse[
            (df_kse.index > start_long_term_event_day) & (df_kse.index <= end_event_day)
        ]
        df_kse_short_term = df_kse[
            (df_kse.index > start_short_term_event_day)
            & (df_kse.index <= end_event_day)
        ]
        arr_long_term = np.array(df_kse_long_term.values)
        if len(arr_long_term) < long_term_range_day:
            arr_long_term = list(arr_long_term) + [0] * (
                long_term_range_day - len(arr_long_term)
            )
        arr_short_term = np.array(df_kse_short_term.values)
        if len(arr_long_term) < short_term_range_day:
            arr_long_term = list(arr_long_term) + [0] * (
                short_term_range_day - len(arr_long_term)
            )
        start_time_series_date = datetime.strptime(start_time_series, "%Y-%m-%d")
        end_time_series_date = start_long_term_event_date - timedelta(
            days=long_term_range_day
        )
        end_time_series_day = datetime.strftime(end_time_series_date, "%Y-%m-%d")
        list_events = []
        list_events_date = []
        start_point = start_time_series_date

        total_event = 0
    st.toast(f":red[STEP 1:] **:blue[Loaded !]** Data of {setting_info} ")
    with st.spinner(f"Load dữ liệu và tính toán các mốc sự kiện"):
        # LOAD FULL TRANSACTIONS - version 4 Brands
        # transaction_raw_path = base_path + "dataset/raw_transaction_SBU_HN.csv"
        # transaction_raw_path = base_path + static_load["transactions_path"]
        # df_transactions_load = pd.read_csv(transaction_raw_path, sep="\t")
        
        # LOAD FULL TRANSACTIONS - version full Brands
        list_df_transaction = []
        for year_i in static_load["year_template_path"]:
            path_transaction_i = base_path + static_load['marketing_transaction_template_path'].replace("'query_brand'", setting_load['brand_selected']).replace("'query_time'", year_i)
            if os.path.isfile(path_transaction_i):
                list_df_transaction_i = pd.read_csv(base_path + static_load['marketing_transaction_template_path']
                                                                .replace("'query_brand'", setting_load['brand_selected'])
                                                                .replace("'query_time'", year_i), sep="\t")
                list_df_transaction.append(list_df_transaction_i)
        df_transactions_load = pd.concat(list_df_transaction, axis=0)

        
        
        # FILTER BASED ON SETTING LOAD
        df_transactions = load_transaction_data(df_transactions_load, setting_load)
        while True:
            start_point_day_i = datetime.strftime(start_point, "%Y-%m-%d")
            end_point_date_i = start_point - timedelta(days=-long_term_range_day)
            end_point_day_i = datetime.strftime(end_point_date_i, "%Y-%m-%d")

            df_kse_long_term_i = df_kse[
                (df_kse.index >= start_point_day_i) & (df_kse.index < end_point_day_i)
            ]
            list_events.append(np.array(df_kse_long_term_i.values))
            list_events_date.append((start_point_day_i, end_point_day_i))
            total_event += 1
            if start_point_day_i >= end_time_series_day:
                break

            start_point = start_point - timedelta(days=-1)
        st.toast(
            f":red[STEP 2:] **:blue[Done !]** Calculated Feature of {total_event} Events"
        )
        time.sleep(0.1)

    with st.spinner(f"Tính toán độ tương đồng của sự kiện với hiện tại"):
        total_pairs = 0

        def min_max_normalize_data(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        def z_score_standardization(data):
            mean = np.mean(data)
            std = np.std(data)
            scaled_arr = (data - mean) / std
            return scaled_arr

        def cal_distance(event_1, event_2):
            event_1_scaled = np.array(z_score_standardization(event_1))
            event_2_scaled = np.array(z_score_standardization(event_2))
            dist = np.linalg.norm(event_1_scaled - event_2_scaled)
            if np.isnan(dist):
                dist = 10e6
            return round(dist)

        list_distances = []
        for event_i in list_events:
            ## event_i & arr_long_term
            ## st.write(f"{list_events}")
            if len(event_i) == len(arr_long_term):
                list_distances.append(cal_distance(event_i, arr_long_term))
                total_pairs += 1
        st.toast(
            f":red[STEP 3:] **:blue[Done !]** Calculated Distance of {total_pairs} pairs"
        )
        time.sleep(0.1)

    # with st.spinner(f'Tính toán các thành phần cua'):
    #     st.toast(f":red[STEP 4:] **:blue[Done !]** Calculated sale components")
    #     time.sleep(0.1)

    # st.dataframe(df_transactions_agg_sale.head(10))
    with st.spinner(f"Tổng hợp {number_of_K} sự kiện giống với hiện tại"):
        Holiday = Holiday_Check()
        dict_holiday_2023 = Holiday.get_dict_holiday_yearly(2023)
        dict_holiday_2022 = Holiday.get_dict_holiday_yearly(2022)
        list_holiday_2022_2023 = list(dict_holiday_2022.keys()) + list(
            dict_holiday_2022.keys()
        )

        def cal_sale_components(x):
            result = {}
            gross_amount = x["PriceSum"]
            group_size = x["GuestCount"]
            group_size_greater_0 = group_size[group_size > 0]
            gross_amount_greater_0 = gross_amount[group_size > 0]
            result["Sale"] = np.sum(gross_amount)
            result["NetSale"] = np.sum(x["PaySum"])
            result["TC"] = np.sum(group_size_greater_0)
            result["TA"] = np.mean(gross_amount_greater_0/group_size_greater_0)
            
            # result["Sale_2"] = np.sum(gross_amount[~is_event])
            return pd.Series(result)

        df_transactions_agg_sale = (
            df_transactions.groupby("ShiftDate")
            .apply(cal_sale_components)
            .reset_index()
        )
        df_transactions_agg_sale["isHoliday"] = df_transactions_agg_sale[
            "ShiftDate"
        ].apply(lambda x: x in list_holiday_2022_2023)
        df_transactions_agg_sale["isWeekend"] = df_transactions_agg_sale[
            "ShiftDate"
        ].apply(lambda x: get_day_of_week(str(x)) in ["Sunday", "Saturday"])
        arr_distances = np.array(list_distances)
        arr_distances_loop = arr_distances
        top_same_event = []
        dx = 0
        number_of_K_same = 100
        for event_i in range(number_of_K_same):
            # get top event
            # idx = np.argpartition(arr_distances_loop, 1)
            idx = np.argsort(arr_distances_loop)
            start_ind = 0
            while True:
                if idx[start_ind] not in list_events_date:
                    top_same_event.append(list_events_date[idx[start_ind]])
                    break
                start_ind += 1
            # remove range around the top N to find nearest top N+1 next
            lower_idx_remove = max(idx[0] - 7, 0)
            upper_idx_remove = min(idx[0] + 7, len(arr_distances_loop))
            for ind_remove in range(lower_idx_remove, upper_idx_remove, 1):
                arr_distances_loop[ind_remove] = 10e6 + dx
                dx += 1
        st.toast(f":red[STEP 4:] **:blue[Done !]** Picked Top K same events")
        time.sleep(0.1)

    st.markdown(
        """
    > ###### :blue[1.2 K-SE | Event Model & Result]:
    """
    )
    with st.expander("👉 Cấu hình mô hình sự kiện", expanded=True):
        col_131, _, col_132 = st.columns([1, 0.1, 1])
        with col_131:
            # Setting get campaign during event or after event
            st.markdown(f"""*:blue[Cấu hình thời gian của mô hình sự kiện]*""")
            check_campaign_after_options = ["Trong sự kiện", "Sau sự kiện"]
            check_campaign_after = st.radio(
                "🎯 Lựa chọn thời điểm thống kê MKT Campaign:",
                options=check_campaign_after_options,
                captions=[
                    "Quan sát thời điểm sự kiện giống với hiện trạng",
                    "Quan sát thời điểm ngay sau sự kiện giống với hiện trạng (Recommend)",
                ],
                help="Giúp MKT team quyết định với sự kiện tương tự trong quá khứ thì **trong đó/sau đó** sử dụng chương trình MKT gì?",
                index=1,
            )
            is_check_campaign_after = (
                True
                if check_campaign_after == check_campaign_after_options[1]
                else False
            )
            # duration_after = st.slider(label='Number of day after Same Event:', min_value=7, max_value=30, value=14, step=1, help="Duration of event: cover MKT Campaign's decision")
        with col_132:
            st.markdown(f"""*:blue[Cấu hình mục tiêu của mô hình sự kiện]*""")
            check_event_target_options = [
                "Không cần mục tiêu",
                "Sự kiện tăng Sale",
                "Sự kiện tăng TA",
                "Sự kiện tăng TC",
                "Sự kiện tăng Brand Awareness",
            ]
            check_event_target_options_map = ["", "Sale", "TA", "TC", "BA"]
            check_event_target = st.radio(
                "🎯 Lựa chọn mục tiêu sự kiện:",
                options=check_event_target_options,
                captions=[
                    "Chỉ quan tâm đến sự kiện giống với hiện trạng :green[(Đã phát triển)]",
                    "Thống kê sự kiện có hiệu quả tăng Sale :green[(Đã phát triển)]",
                    "Thống kê sự kiện có hiệu quả tăng TA :green[(Đã phát triển)]",
                    "Thống kê sự kiện có hiệu quả tăng TC :green[(Đã phát triển)]",
                    "Thống kê sự kiện có hiệu quả truyền thông cho Brand :gray[(Chưa phát triển)]",
                ],
                index=1,
                help="Giúp MKT team quyết định sự kiện tương tự có hiệu quả MKT như thế nào?",
            )
            target_increase_type = dict(
                zip(check_event_target_options, check_event_target_options_map)
            )[check_event_target]
        col_133, _, col_134 = st.columns([1, 0.1, 1])
        with col_133:
            if is_check_campaign_after:
                st.markdown("""---""")
                check_event_duration_options = ["1-2 tuần", "4 tuần", "6 tuần"]
                check_event_duration = st.radio(
                    "🎯 Lựa chọn khoảng thời gian phân tích :",
                    options=check_event_duration_options,
                    captions=[
                        "Chương trình Flash MKT  (chạy trong khoảng 1 tuần tiếp theo)",
                        "Chương trình MKT thông thường (chạy trong khoảng 4 tuần tiếp theo)",
                        "Chương trình MKT thông thường dài (chạy trong khoảng 6 tuần tiếp theo)",
                    ],
                    index=1,
                    help="Giúp MKT team quyết định kết quả phân tích hiệu quả MKT khi nào?",
                )
                duration_after = (
                    14
                    if check_event_duration == check_event_duration_options[0]
                    else (
                        28
                        if check_event_duration == check_event_duration_options[1]
                        else 42
                    )
                )
        with col_134:
            if target_increase_type not in ["", "BA"]:
                st.markdown("""---""")
                increase_sale_target_options = [
                    "105%",
                    "125%",
                    "133%",
                    "150%",
                    "175%",
                ]
                increase_sale_target_options_map = [1.05, 1.25, 1.33, 1.50, 1.75]
                target_increase_sale = st.radio(
                    "🎯 Lựa chọn ngưỡng tăng Sale/TC/TA mong muốn",
                    options=increase_sale_target_options,
                    index=0,
                    help="Giúp MKT lựa chọn các sự kiện đảm bảo tăng thành phần Sale? Chú ý, phụ thuộc vào cấu hình ở giữa setting",
                )
                target_increase_sale = dict(
                    zip(increase_sale_target_options, increase_sale_target_options_map)
                )[target_increase_sale]
            else:
                target_increase_sale = -1

        # Get top_same_event after
        top_same_event_after = []
        for pair_i in top_same_event:
            start_date_i = datetime.strptime(pair_i[1], "%Y-%m-%d") + timedelta(1)
            end_date_i = datetime.strptime(pair_i[1], "%Y-%m-%d") + timedelta(
                1 + long_term_range_day
            )
            start_day_i = datetime.strftime(start_date_i, "%Y-%m-%d")
            end_day_i = min(
                datetime.strftime(end_date_i, "%Y-%m-%d"), current_event_day
            )
            top_same_event_after.append((start_day_i, end_day_i))
        list_views_event_after = [
            list(df_kse[(df_kse.index >= i[0]) & (df_kse.index <= i[1])])
            for i in top_same_event_after
        ]

        # Let's extend your creative to create decor dataframe
        top_same_event_decor = pd.DataFrame(top_same_event)
        top_same_event_decor.columns = ["Start Event", "End Event"]
        top_same_event_decor["Same_i"] = [
            i for i in list(range(1, top_same_event_decor.shape[0] + 1, 1))
        ]
        top_same_event_decor["Same"] = [
            f"TOP {i} ⭐" for i in list(range(1, top_same_event_decor.shape[0] + 1, 1))
        ]

        list_views_event = [
            list(df_kse[(df_kse.index >= i[0]) & (df_kse.index <= i[1])])
            for i in top_same_event
        ]
        top_same_event_decor["Views"] = list_views_event
        top_same_event_decor["Views_after"] = list_views_event_after
        top_same_event_decor["category"] = [
            f"✔️" for i in range(1, top_same_event_decor.shape[0] + 1, 1)
        ]
        top_same_event_decor = top_same_event_decor[top_same_event_decor.apply(lambda x: (len(x['Views']) > 1) & (len(x['Views_after']) > 1), axis=1)]
        
        # st.dataframe(top_same_event_decor)
        current_event = pd.DataFrame(
            {
                "Start Event": [start_long_term_event_day],
                "End Event": [end_event_day],
                "Same": ["Current Event"],
                "category": ["📉 Triggeration; 🔑 TradeMKT Performance"],
            }
        )
        current_event["Views"] = [arr_long_term]

        # Get top_same_event_after increase sale
        if target_increase_sale > 1:

            def get_performance_of_same_event(
                start_event, end_event, duration_after, feature_col
            ):
                # Get same event sale components
                df_trans_same_event = df_transactions_agg_sale[
                    (df_transactions_agg_sale["ShiftDate"] >= start_event)
                    & (df_transactions_agg_sale["ShiftDate"] <= end_event)
                ]
                # Get after event sale components
                end_event_after = datetime.strptime(end_event, "%Y-%m-%d") + timedelta(
                    duration_after
                )
                end_event_after = datetime.strftime(end_event_after, "%Y-%m-%d")
                df_trans_same_event_after = df_transactions_agg_sale[
                    (df_transactions_agg_sale["ShiftDate"] > end_event)
                    & (df_transactions_agg_sale["ShiftDate"] <= end_event_after)
                ]
                # Remove holiday in sale components
                df_trans_same_event = df_trans_same_event[
                    ~df_trans_same_event["isHoliday"]
                ]
                df_trans_same_event_after = df_trans_same_event_after[
                    ~df_trans_same_event_after["isHoliday"]
                ]

                return np.mean(df_trans_same_event[feature_col]), np.mean(
                    df_trans_same_event_after[feature_col]
                )

            if target_increase_type == "Sale":
                current_event["Sale_Event"] = current_event.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "Sale"
                    )[0],
                    axis=1,
                )
                current_event["Sale_Event"] = current_event["Sale_Event"].apply(
                    lambda x: round_money(x, 6)
                )
                top_same_event_decor["Sale_Event"] = top_same_event_decor.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "Sale"
                    )[0],
                    axis=1,
                )
                top_same_event_decor["Sale_Event_After"] = top_same_event_decor.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "Sale"
                    )[1],
                    axis=1,
                )
                top_same_event_decor["Sale_Event"] = top_same_event_decor[
                    "Sale_Event"
                ].apply(lambda x: round_money(x, 6))
                top_same_event_decor["Sale_Event_After"] = top_same_event_decor[
                    "Sale_Event_After"
                ].apply(lambda x: round_money(x, 6))
                top_same_event_decor["%Increase Sale"] = top_same_event_decor.apply(
                    lambda x: round(
                        1.0
                        + (x["Sale_Event_After"] - x["Sale_Event"]) / x["Sale_Event"],
                        2,
                    ),
                    axis=1,
                )
                top_same_event_decor = top_same_event_decor[
                    top_same_event_decor["%Increase Sale"] >= target_increase_sale
                ]
                top_same_event_decor["%Increase Sale"] = top_same_event_decor[
                    "%Increase Sale"
                ].apply(lambda x: f"{round(x*100)}%")
                same_event_increase_columns = [
                    "Sale_Event",
                    "Sale_Event_After",
                    "%Increase Sale",
                ]

            if target_increase_type == "TC":
                current_event["TC_Event"] = current_event.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "TC"
                    )[0],
                    axis=1,
                )
                current_event["TC_Event"] = current_event["TC_Event"].apply(
                    lambda x: round(x)
                )
                top_same_event_decor["TC_Event"] = top_same_event_decor.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "TC"
                    )[0],
                    axis=1,
                )
                top_same_event_decor["TC_Event_After"] = top_same_event_decor.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "TC"
                    )[1],
                    axis=1,
                )
                top_same_event_decor["TC_Event"] = top_same_event_decor[
                    "TC_Event"
                ].apply(lambda x: round(x))
                top_same_event_decor["TC_Event_After"] = top_same_event_decor[
                    "TC_Event_After"
                ].apply(lambda x: round(x))
                top_same_event_decor["%Increase TC"] = top_same_event_decor.apply(
                    lambda x: round(
                        1.0 + (x["TC_Event_After"] - x["TC_Event"]) / x["TC_Event"], 2
                    ),
                    axis=1,
                )
                #st.dataframe(top_same_event_decor)
                top_same_event_decor = top_same_event_decor[
                    top_same_event_decor["%Increase TC"] >= target_increase_sale
                ]
                top_same_event_decor["%Increase TC"] = top_same_event_decor[
                    "%Increase TC"
                ].apply(lambda x: f"{round(x*100)}%")
                same_event_increase_columns = [
                    "TC_Event",
                    "TC_Event_After",
                    "%Increase TC",
                ]
            if target_increase_type == "TA":
                current_event["TA_Event"] = current_event.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "TA"
                    )[0],
                    axis=1,
                )
                current_event["TA_Event"] = current_event["TA_Event"].apply(
                    lambda x: round_money(x, 3)
                )

                top_same_event_decor["TA_Event"] = top_same_event_decor.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "TA"
                    )[0],
                    axis=1,
                )
                top_same_event_decor["TA_Event_After"] = top_same_event_decor.apply(
                    lambda x: get_performance_of_same_event(
                        x["Start Event"], x["End Event"], duration_after, "TA"
                    )[1],
                    axis=1,
                )
                top_same_event_decor["TA_Event"] = top_same_event_decor[
                    "TA_Event"
                ].apply(lambda x: round_money(x, 3))
                top_same_event_decor["TA_Event_After"] = top_same_event_decor[
                    "TA_Event_After"
                ].apply(lambda x: round_money(x, 3))
                top_same_event_decor["%Increase TA"] = top_same_event_decor.apply(
                    lambda x: round(
                        1.0 + (x["TA_Event_After"] - x["TA_Event"]) / x["TA_Event"], 2
                    ),
                    axis=1,
                )
                top_same_event_decor = top_same_event_decor[
                    top_same_event_decor["%Increase TA"] >= target_increase_sale
                ]
                top_same_event_decor["%Increase TA"] = top_same_event_decor[
                    "%Increase TA"
                ].apply(lambda x: f"{round(x*100)}%")
                same_event_increase_columns = [
                    "TA_Event",
                    "TA_Event_After",
                    "%Increase TA",
                ]
            top_same_event_decor.sort_values(
                by=["Same_i"], ascending=True, inplace=True
            )
            top_same_event_decor = top_same_event_decor.iloc[:number_of_K, :]
            same_event_decor_columns = [
                "Start Event",
                "End Event",
                "Same",
                "Same_i",
                "Views",
                "Views_after",
                "category",
            ] + same_event_increase_columns
            top_same_event_decor = top_same_event_decor[same_event_decor_columns]

        # Sort event by similarity (desc)
        top_same_event_decor.sort_values(
             by=["Same_i"], ascending=True, inplace=True
        )
        top_same_event_decor.drop(columns = ['Same_i'], inplace=True)
        
        # Also show current event with same events
        top_same_event_decor_merge = pd.concat(
            [current_event, top_same_event_decor], axis=0
        )
        # top_same_event_decor_merge.sort_values(
        #     by=["Same"], ascending=True, inplace=True
        # )
        #st.dataframe(top_same_event_decor_merge.head(20))
        # config_y_min = (
        #     min([min(i) for i in list_views_event + list_views_event_after ]) * 0.9
        # )
        # config_y_max = (
        #     max([max(i) for i in list_views_event + list_views_event_after]) * 1.0
        # )
        try:
            config_y_min = (
                min([min(i) for i in list_views_event + list_views_event_after ]) * 0.9
            )
        except:
            config_y_min = 0
        try:
            config_y_max = (
                max([max(i) for i in list_views_event + list_views_event_after]) * 1.0
            )
        except:
            config_y_max = max(list_views_event)
        with st.form(key="form"):
            top_same_event_decor_after = st.data_editor(
                top_same_event_decor_merge,
                column_config={
                    "Start Event": "Ngày bắt đầu sự kiện",
                    "End Event": "Ngày kết thúc sự kiện",
                    "Same": st.column_config.TextColumn(
                        "Mức độ giống với sự kiện hiện tại",
                        help="Level of similarity of Event to Current",
                    ),
                    "Views": st.column_config.BarChartColumn(
                        "Xu hướng (Trend) trong sự kiện",
                        width="medium",
                        y_min=config_y_min,
                        y_max=config_y_max,
                    ),
                    "Views_after": st.column_config.BarChartColumn(
                        "Xu hướng (Trend) sau sự kiện",
                        width="medium",
                        y_min=config_y_min,
                        y_max=config_y_max,
                    ),
                    f"{target_increase_type}_Event": st.column_config.NumberColumn(
                        f"Trung Bình {target_increase_type} trong sự kiện",
                        help=f"Trung bình {target_increase_type} các ngày trong sự kiện đã loại trừ các dịp Lễ Holiday",
                    ),
                    f"{target_increase_type}_Event_After": st.column_config.NumberColumn(
                        f"Trung Bình {target_increase_type} sau sự kiện",
                        help=f"Trung bình {target_increase_type} các ngày sau sự kiện đã loại trừ các dịp Lễ Holiday",
                    ),
                    f"%Increase {target_increase_type}": st.column_config.TextColumn(
                        f"% Tăng {target_increase_type}",
                        help="%Tăng của trung bình các ngày sau sự kiện so với trong sự kiện",
                    ),
                    "category": st.column_config.SelectboxColumn(
                        "Lựa chọn sự kiện thống kê chương trình MKT",
                        help="Choose this to show MKT Performance of events",
                        width="large",
                        default="✔️",
                        options=["✔️", "❌"],
                    ),
                },
                hide_index=True,
                key="event_data_editor",
                use_container_width=True,
            )
            submit_button = st.form_submit_button(
                label=":red[SUBMIT] (✔️: Chọn sự kiện thống kê MKT, ❌: Loại sự kiện thống kê MKT)",
                help="Should to pick 1-3 Events to reduce complexity when make decision",
            )
    #     with st.expander("Event Plotting?", expanded=False):
    #         df_kse_plot = df_kse.copy()
    #         df_kse_plot = pd.DataFrame(df_kse_plot)
    #         df_kse_plot.fillna(0, inplace = True)
    #         bokeh_same_event(df_kse_plot=df_kse_plot, col_name=col_name,
    #                          current_event = [(start_long_term_event_day, end_event_day)], same_events = top_same_event,
    #                          is_plot_intervals=False, is_plot_covid=False)

    st.markdown(
        """
    > ###### :blue[1.3 Dictionary]:
    """
    )
    with st.expander("👉 Thông tin về Triggeration & Performance", expanded=False):
        # compare type setting
        column_setting_1, column_setting_2, column_setting_3 = st.columns([2, 2, 4])
        with column_setting_1:
            insight_show_options = ["📉 Triggeration", "🔑 TradeMKT Performance"]
            insight_show_type = st.selectbox(
                "🎯 Nội dung tra cứu",
                insight_show_options,
                help="Choose interface that you want to inspect. Single mode is more easy to get information!",
                index=1,
            )
        with column_setting_2:
            if insight_show_type == "📉 Triggeration":
                compare_type = st.selectbox(
                    "Lựa chọn cách quan sát 📉 MKT Triggeration? ",
                    ["vs Sự kiện ngay trước đó", "vs Sự kiện cùng kỳ năm ngoái"],
                    help="**Last Term's Period**: Comparing current period to period with same length, just before. **Last Year's Period**: Comparing current period to period with same length, last year in Solar Calender. ",
                    index=1,
                )
            else:
                compare_type = "vs Sự kiện ngay trước đó"
        # with column_setting_2:
        #     triggeration_duration_options = [f"Long-term ({long_term_range_day})", f"Short-term ({short_term_range_day})"]
        #     triggeration_show_type = st.selectbox('Duration for 📉 MKT Triggeration  ? ', triggeration_duration_options,
        #                                      help="Choose duration to calculate metrics in  📉 Triggeration", disabled =True, index = 0)

        if top_same_event_decor_after.iloc[0, -1] == "💡 Fact Information":
            st.markdown(f"💡 Fact Information: ...")
        elif top_same_event_decor_after.iloc[0, -1] == "📉 Triggeration":
            st.markdown("📉 Triggeration: ...")
        elif top_same_event_decor_after.iloc[0, -1] == "🔑 TradeMKT Performance":
            st.markdown("🔑 TradeMKT Performance: ...")
        else:
            # tab_insight_1, tab_insight_2 = st.tabs([f"Long-term ({long_term_range_day})", f"Short-term ({short_term_range_day})"])
            # tab_insight_1 = st.tabs([f"Long-term ({long_term_range_day})"])

            def markdown_up_down(
                value_name,
                current_value,
                compare_value,
                round_percentage=1,
                is_currency=True,
            ):
                if current_value <= compare_value:
                    color_compare = "red"
                    word_compare = "DOWN"
                else:
                    color_compare = "green"
                    word_compare = "UP"
                percentage_compare = (
                    f"{round((current_value - compare_value)/compare_value*100, round_percentage)} %"
                    if compare_value != 0
                    else 0
                )
                if is_currency:
                    current_value = currency_format(current_value)
                result = f"{value_name}: :blue[{current_value}] - **:{color_compare}[{word_compare}: {percentage_compare}]**"
                return result

            def get_insight_period_information(
                df_transactions,
                trend,
                start_term_event_day,
                end_term_event_day,
                start_term_compare,
                end_term_compare,
            ):
                df_transactions_event = df_transactions[
                    (df_transactions.ShiftDate > start_term_event_day)
                    & (df_transactions.ShiftDate <= end_term_event_day)
                ]
                df_transactions_compare = df_transactions[
                    (df_transactions.ShiftDate > start_term_compare)
                    & (df_transactions.ShiftDate <= end_term_compare)
                ]

                trend_event = trend[
                    (trend.index > start_term_event_day)
                    & (trend.index <= end_term_event_day)
                ]
                trend_compare = trend[
                    (trend.index > start_term_compare)
                    & (trend.index <= end_term_compare)
                ]

                result = {}

                # TA, AOV, GroupSize, Sale, TC
                result["TA"] = {
                    "TA": round_money(
                        np.mean(
                            np.mean(df_transactions_event["PriceSum"])
                            / np.mean(df_transactions_event["GuestCount"])
                        ),
                        3,
                    ),
                    "TA_compare": round_money(
                        np.mean(
                            np.mean(df_transactions_compare["PriceSum"])
                            / np.mean(df_transactions_compare["GuestCount"])
                        ),
                        3,
                    ),
                }

                result["AOV"] = {
                    "AOV": round_money(np.mean(df_transactions_event["PriceSum"]), 3),
                    "AOV_compare": round_money(
                        np.mean(df_transactions_compare["PriceSum"]), 6
                    ),
                }

                result["GroupSize"] = {
                    "GroupSize": round(np.mean(df_transactions_event["GuestCount"]), 2),
                    "GroupSize_compare": round(
                        np.mean(df_transactions_compare["GuestCount"]), 2
                    ),
                }

                result["Sale"] = {
                    "Sale": round_money(np.sum(df_transactions_event["PriceSum"]), 9),
                    "Sale_compare": round_money(
                        np.sum(df_transactions_compare["PriceSum"]), 9
                    ),
                }

                result["TC"] = {
                    "TC": round(np.sum(df_transactions_event["GuestCount"]), 0),
                    "TC_compare": round(
                        np.sum(df_transactions_compare["GuestCount"]), 0
                    ),
                }

                # Trend - Remove Seasonal, Remove Residual
                ## CAL COUNT UPTREND, DOWN TREND
                count_uptrend = 0
                count_downtrend = 0
                total_trend = 0
                current_trend_i = 0
                current_trend_lift = (
                    0  # 'None': default, uptrend: 'UP': down trend : 'DOWN'
                )
                list_uptrend_date = []
                list_downtrend_date = []
                datetime_format_output = "%d/%m"
                for i, trend_i in enumerate(trend_event):
                    if i == 0:
                        # skip first trend_i
                        current_trend_i = trend_i
                        current_trend_lift = 0
                        continue
                    if (trend_i > current_trend_i) and (current_trend_lift != "UP"):
                        # check change trend down to up/ none to up
                        count_uptrend += 1
                        total_trend += 1
                        list_uptrend_date.append(
                            datetime.strftime(
                                trend_event.index[i], datetime_format_output
                            )
                        )
                        current_trend_lift = "UP"
                        current_trend_i = trend_i
                    elif (trend_i < current_trend_i) and (current_trend_lift != "DOWN"):
                        # check change trend up to down/ none to down
                        count_downtrend += 1
                        total_trend += 1
                        list_downtrend_date.append(
                            datetime.strftime(
                                trend_event.index[i], datetime_format_output
                            )
                        )
                        current_trend_lift = "DOWN"
                        current_trend_i = trend_i
                    else:
                        current_trend_i = trend_i

                ## CAL LONGEST UPLIFT & DOWNLIFT
                def longest_true_subarray(x):
                    max_length = 0  # Initialize the maximum length to 0
                    current_length = 0  # Initialize the current length to 0
                    start_index = (
                        0  # Initialize the starting index of the current subarray
                    )
                    for i in range(len(x)):
                        if x[i]:  # If the current element is True
                            current_length += 1
                        else:
                            current_length = (
                                0  # Reset the current length if the element is False
                            )
                        if current_length > max_length:
                            max_length = current_length
                            start_index = i - max_length + 1
                    longest_subarray = x[
                        start_index : start_index + max_length
                    ]  # Extract the longest subarray
                    return sum(longest_subarray), (
                        start_index,
                        start_index + max_length - 1,
                    )

                list_trend = list(trend_event)
                lift_gap = [
                    list_trend[i + 1] - list_trend[i]
                    for i in range(0, len(list_trend) - 1)
                ]
                lift_gap_up = [(gap_i > 0) for gap_i in lift_gap]
                lift_gap_down = [(gap_i < 0) for gap_i in lift_gap]
                try:
                    longest_uptrend, tuple_longest_uptrend_index = (
                        longest_true_subarray(lift_gap_up)
                    )
                    if longest_uptrend > 0:
                        start_longest_uptrend = datetime.strftime(
                            trend_event.index[tuple_longest_uptrend_index[0]],
                            datetime_format_output,
                        )
                        end_longest_uptrend = datetime.strftime(
                            trend_event.index[tuple_longest_uptrend_index[1]],
                            datetime_format_output,
                        )
                        tuple_longest_uptrend = (
                            start_longest_uptrend,
                            end_longest_uptrend,
                        )
                    else:
                        tuple_longest_uptrend = ("", "")
                    longest_downtrend, tuple_longest_downtrend_index = (
                        longest_true_subarray(lift_gap_down)
                    )
                    if longest_downtrend > 0:
                        start_longest_downtrend = datetime.strftime(
                            trend_event.index[tuple_longest_downtrend_index[0]],
                            datetime_format_output,
                        )
                        end_longest_downtrend = datetime.strftime(
                            trend_event.index[tuple_longest_downtrend_index[1]],
                            datetime_format_output,
                        )
                        tuple_longest_downtrend = (
                            start_longest_downtrend,
                            end_longest_downtrend,
                        )
                    else:
                        tuple_longest_downtrend = ("", "")
                except:
                    longest_uptrend, longest_downtrend = 0, 0
                    tuple_longest_uptrend, tuple_longest_downtrend = ("", ""), ("", "")

                result["DownUpTrend"] = {
                    "UpTrend": count_uptrend,
                    "UpTrendDate": list_uptrend_date,
                    "UpTrendDuration": longest_uptrend,
                    "UpTrendLongestRange": tuple_longest_uptrend,
                    "DownTrend": count_downtrend,
                    "DownTrendDate": list_downtrend_date,
                    "DownTrendDuration": longest_downtrend,
                    "DownTrendLongestRange": tuple_longest_downtrend,
                    "TotalTrend": total_trend,
                    "TrendDuration": len(lift_gap),
                }

                # Holiday, Weekend
                Holiday = Holiday_Check()
                dict_holiday_2024 = Holiday.get_dict_holiday_yearly(2024)
                dict_holiday_2023 = Holiday.get_dict_holiday_yearly(2023)
                dict_holiday_2022 = Holiday.get_dict_holiday_yearly(2022)
                dict_holiday_2021 = Holiday.get_dict_holiday_yearly(2021)
                dict_holiday_2020 = Holiday.get_dict_holiday_yearly(2020)
                dict_holiday_agg = {
                    "2024": dict_holiday_2024,
                    "2023": dict_holiday_2023,
                    "2022": dict_holiday_2022,
                    "2021": dict_holiday_2021,
                    "2020": dict_holiday_2020,
                }
                list_event_shift_date = sorted(
                    list(set(df_transactions_event.ShiftDate))
                )
                list_event_compare_shift_date = sorted(
                    list(set(df_transactions_compare.ShiftDate))
                )
                count_Holiday = 0
                list_Holiday = []
                dict_Holiday_name = {}
                count_Weekend = 0
                list_Weekend = []
                for date_i in list_event_shift_date:
                    year_i = datetime.strftime(
                        datetime.strptime(date_i, "%Y-%m-%d"), "%Y"
                    )
                    dict_this_year = dict_holiday_agg[f"{year_i}"]
                    if date_i in dict_this_year.keys():
                        count_Holiday += 1
                        list_Holiday.append(date_i)
                        dict_Holiday_name[date_i] = dict_this_year[date_i]
                    if get_day_of_week(date_i) in ["Saturday", "Sunday"]:
                        count_Weekend += 1
                        list_Weekend.append(date_i)

                list_Weekend_compare = []
                for date_compare_i in list_event_compare_shift_date:
                    if get_day_of_week(date_compare_i) in ["Saturday", "Sunday"]:
                        list_Weekend_compare.append(date_compare_i)

                if len(list_Holiday) > 0:
                    df_transactions_Holiday = df_transactions_event[
                        df_transactions_event.ShiftDate.isin(list_Holiday)
                    ]
                    df_transactions_not_Holiday = df_transactions_event[
                        ~df_transactions_event.ShiftDate.isin(list_Holiday)
                    ]

                    avg_guest_holiday = np.mean(
                        df_transactions_Holiday.groupby("ShiftDate")["GuestCount"].sum()
                    )
                    avg_guest_not_holiday = np.mean(
                        df_transactions_not_Holiday.groupby("ShiftDate")[
                            "GuestCount"
                        ].sum()
                    )
                    rate_Holiday = round(avg_guest_holiday / avg_guest_not_holiday, 2)
                else:
                    rate_Holiday = 0

                if len(list_Weekend) > 0:
                    df_transactions_Weekend = df_transactions_event[
                        df_transactions_event.ShiftDate.isin(list_Weekend)
                    ]
                    df_transactions_not_Weekend = df_transactions_event[
                        ~df_transactions_event.ShiftDate.isin(list_Weekend)
                    ]

                    avg_guest_weekend = np.mean(
                        df_transactions_Weekend.groupby("ShiftDate")["GuestCount"].sum()
                    )
                    avg_guest_not_weekend = np.mean(
                        df_transactions_not_Weekend.groupby("ShiftDate")[
                            "GuestCount"
                        ].sum()
                    )
                    avg_guest_full = np.mean(
                        df_transactions_event.groupby("ShiftDate")["GuestCount"].sum()
                    )

                    rate_Weekend = round(avg_guest_weekend / avg_guest_full, 2)
                    rate_Weekday = round(avg_guest_not_weekend / avg_guest_full, 2)
                else:
                    rate_Weekend = 0
                    rate_Weekday = 1
                if len(list_Weekend_compare) > 0:
                    df_transactions_compare_Weekend = df_transactions_compare[
                        df_transactions_compare.ShiftDate.isin(list_Weekend_compare)
                    ]
                    df_transactions_compare_not_Weekend = df_transactions_compare[
                        ~df_transactions_compare.ShiftDate.isin(list_Weekend_compare)
                    ]
                    avg_guest_compare_weekend = np.mean(
                        df_transactions_compare_Weekend.groupby("ShiftDate")[
                            "GuestCount"
                        ].sum()
                    )
                    avg_guest_compare_not_weekend = np.mean(
                        df_transactions_compare_not_Weekend.groupby("ShiftDate")[
                            "GuestCount"
                        ].sum()
                    )
                    avg_guest_compare_full = np.mean(
                        df_transactions_compare.groupby("ShiftDate")["GuestCount"].sum()
                    )
                    rate_Weekend_compare = round(
                        avg_guest_compare_weekend / avg_guest_compare_full, 2
                    )
                    rate_Weekday_compare = round(
                        avg_guest_compare_not_weekend / avg_guest_compare_full, 2
                    )
                else:
                    rate_Weekend_compare = 0
                    rate_Weekday_compare = 1
                # Load common rate from df_brand_agg
                # df_brand_reload = pd.read_csv(base_path + "apps/dataset/df_brand_agg.csv", sep = "\t")
                # df_brand_reload = df_brand_reload[df_brand_reload.BrandName == setting_load['brand_selected']]
                rate_Holiday_compare = 1.73
                # rate_Weekend_compare=1.64
                result["Holiday"] = {
                    "Holiday_Days": count_Holiday,
                    "Holiday_Days_Name": dict_Holiday_name,
                    "Holiday_Rate": rate_Holiday,
                    "Holiday_Rate_compare": rate_Holiday_compare,
                }
                result["Weekend"] = {
                    "Weekend_Days": count_Weekend,
                    "Weekend_Rate": rate_Weekend,
                    "Weekend_Rate_compare": rate_Weekend_compare,
                    "Weekday_Rate": rate_Weekday,
                    "Weekday_Rate_compare": rate_Weekday_compare,
                }
                return result

            def get_insight_weather_current_information(
                end_term_event_day, city="Hanoi"
            ):
                result = {}
                # Get latest data
                path_weather_data = base_path + static_load["weather_folder"]
                daily_data_path = max(glob.glob(path_weather_data + "/*_daily.csv"))
                updated_daily = daily_data_path.split("/")[-1].split("_")[0]
                hourly_data_path = max(glob.glob(path_weather_data + "/*_hourly.csv"))
                updated_hourly = hourly_data_path.split("/")[-1].split("_")[0]
                weather_daily = pd.read_csv(daily_data_path, sep=",")
                weather_daily = weather_daily[weather_daily["City"] == city]
                weather_hourly = pd.read_csv(hourly_data_path, sep=",")
                weather_hourly = weather_hourly[weather_hourly["City"] == city]

                result["WhenUpdated"] = max([updated_daily, updated_hourly])
                result["TempDaily"] = dict(
                    zip(weather_daily.Time, weather_daily.avgtemp_c)
                )
                result["TempHourly"] = dict(
                    zip(weather_hourly.Time, weather_hourly.temp_c)
                )
                result["WeatherDaily"] = dict(
                    zip(weather_daily.Time, weather_daily.weather)
                )
                result["WeatherHourly"] = dict(
                    zip(weather_hourly.Time, weather_hourly.weather)
                )
                return result

            # INSIGHT LONG-TERM
            # insight_details_term(term_range_day=long_term_range_day,
            #                      compare_type=compare_type,
            #                      start_term_event_date=start_long_term_event_date,
            #                      columns_insight=[col_insight_11, col_insight_12],
            #                      columns_mode=columns_mode, key_multiselect='km_1')
            # SETTING INTERFACE MODE: BOTH OR SINGLEinsight_show_options
            # if insight_show_type == insight_show_options[0]:
            #     col_insight_11, col_insight_12 = st.columns([1,1])
            #     columns_mode = 'Both'
            if insight_show_type == insight_show_options[0]:
                col_insight_11 = st.container()
                col_insight_12 = None
                columns_mode = "Single I"
            elif insight_show_type == insight_show_options[1]:
                col_insight_12 = st.container()
                col_insight_11 = None
                columns_mode = "Single II"

            term_range_day = long_term_range_day
            compare_type = compare_type
            start_term_event_date = start_long_term_event_date
            columns_insight = [col_insight_11, col_insight_12]
            columns_mode = columns_mode
            key_multiselect = "km_1"
            # def insight_details_term(term_range_day=long_term_range_day, compare_type="vs Last Term's Period", start_term_event_date=start_long_term_event_date,
            #                          columns_insight=[], columns_mode='Both', key_multiselect=''):
            if compare_type == "vs Sự kiện ngay trước đó":
                start_term_previous = datetime.strftime(
                    start_term_event_date - timedelta(term_range_day + 1), "%Y-%m-%d"
                )
                end_term_previous = datetime.strftime(
                    start_term_event_date - timedelta(1), "%Y-%m-%d"
                )
            elif compare_type == "vs Sự kiện cùng kỳ năm ngoái":
                start_term_previous_year = (
                    f"{int(datetime.strftime(start_term_event_date, '%Y')) - 1}"
                )
                start_term_previous = f"{start_term_previous_year}-{datetime.strftime(start_long_term_event_date, '%m-%d')}"

                end_event_date = datetime.strptime(end_event_day, "%Y-%m-%d")
                end_term_previous_year = (
                    f"{int(datetime.strftime(end_event_date, '%Y')) - 1}"
                )
                end_term_previous = f"{end_term_previous_year}-{datetime.strftime(end_event_date, '%m-%d')}"
            else:
                st.error("Wrong Period options", icon="🚨")
            start_term_event_day = datetime.strftime(start_term_event_date, "%Y-%m-%d")
            end_term_event_day = end_event_day
            dict_insight_long_term = get_insight_period_information(
                df_transactions,
                trend,
                start_term_event_day,
                end_term_event_day,
                start_term_previous,
                end_term_previous,
            )
            dict_insight_weather = get_insight_weather_current_information(
                end_term_event_day
            )
            if columns_mode == "Single I":
                with columns_insight[0]:
                    if dict_insight_long_term["Holiday"]["Holiday_Days"] > 0:
                        list_holiday_name = dict_insight_long_term["Holiday"][
                            "Holiday_Days_Name"
                        ].values()
                        list_holiday_date = dict_insight_long_term["Holiday"][
                            "Holiday_Days_Name"
                        ].keys()
                        details_holiday = ", ".join(
                            [
                                f"{i}: {j}"
                                for (i, j) in zip(list_holiday_name, list_holiday_date)
                            ]
                        )
                    else:
                        details_holiday = ""
                    st.markdown(
                        f"""
                    - ##### 📉 Các tín hiệu để Trigger MKT Campaign:
                        - 💰 **Business Metrics Trend**:
                            + {markdown_up_down("TA", dict_insight_long_term["TA"]["TA"], dict_insight_long_term["TA"]["TA_compare"])}
                            + {markdown_up_down("AOV", dict_insight_long_term["AOV"]["AOV"], dict_insight_long_term["AOV"]["AOV_compare"])}
                            + {markdown_up_down("GroupSize", dict_insight_long_term["GroupSize"]["GroupSize"], dict_insight_long_term["GroupSize"]["GroupSize_compare"])}
                            + {markdown_up_down("Sale", dict_insight_long_term["Sale"]["Sale"], dict_insight_long_term["Sale"]["Sale_compare"])}
                            + {markdown_up_down("TC", dict_insight_long_term["TC"]["TC"], dict_insight_long_term["TC"]["TC_compare"], is_currency=False)}
                            + Up-Trend Times: :blue[{dict_insight_long_term["DownUpTrend"]["UpTrend"]}/{dict_insight_long_term["DownUpTrend"]["TotalTrend"]}] - **:green[List of Days: {", ".join(dict_insight_long_term["DownUpTrend"]["UpTrendDate"])}]**
                            + Up-Trend Longest Duration: :blue[{dict_insight_long_term["DownUpTrend"]["UpTrendDuration"]}/{dict_insight_long_term["DownUpTrend"]["TrendDuration"]}] - **:green[From - to: {"-".join(dict_insight_long_term["DownUpTrend"]["UpTrendLongestRange"])}]**
                            + Down-Trend Times: :blue[{dict_insight_long_term["DownUpTrend"]["DownTrend"]}/{dict_insight_long_term["DownUpTrend"]["TotalTrend"]}] - **:red[List of Days: {", ".join(dict_insight_long_term["DownUpTrend"]["DownTrendDate"])}]**
                            + Down-Trend Longest Duration: :blue[{dict_insight_long_term["DownUpTrend"]["DownTrendDuration"]}/{dict_insight_long_term["DownUpTrend"]["TrendDuration"]}] - **:red[From - to: {"- ".join(dict_insight_long_term["DownUpTrend"]["DownTrendLongestRange"])}]**
                        ---
                        - 📅 **Events**:
                            + Number of Holiday: :blue[{dict_insight_long_term["Holiday"]["Holiday_Days"]} days] 
                                + Holiday Date: :blue[{details_holiday}]
                                + Holiday Rate: :blue[{dict_insight_long_term["Holiday"]["Holiday_Rate"]}]
                            + Number of Weekend: :blue[{dict_insight_long_term["Weekend"]["Weekend_Days"]}] days
                                + {markdown_up_down("Weekend Rate", dict_insight_long_term["Weekend"]["Weekend_Rate"], dict_insight_long_term["Weekend"]["Weekend_Rate_compare"],  is_currency=False)}
                                + {markdown_up_down("Weekday Rate", dict_insight_long_term["Weekend"]["Weekday_Rate"], dict_insight_long_term["Weekend"]["Weekday_Rate_compare"],  is_currency=False)}
                            + New Brand/Restaurants: :blue[{False}]
                            + New Items: :blue[{False}]
                            + Media Crisis: :blue[{False}]
                        ---
                        - ⛈️ **Others (Weather, Traffic, ...)**:
                            + Updated Time: **:blue[{dict_insight_weather['WhenUpdated']}]**
                            + Average of Temperature: :blue[{round(np.mean(list(dict_insight_weather['TempDaily'].values())), 1)} oC] 
                            + Mostly Weather :blue[{Counter(dict_insight_weather['WeatherHourly'].values()).most_common(1)[0][0]}]
                        """
                    )

                    column_plot_1, column_plot_2 = st.columns([2, 2], gap="small")
                    with column_plot_1:
                        df_weather_daily_plot = pd.DataFrame(
                            {
                                "Weather": list(
                                    dict_insight_weather["WeatherHourly"].values()
                                )
                            }
                        )
                        df_weather_daily_plot["NoDay"] = 1
                        top_weather_plot = 5
                        list_weather_plot = [
                            i[0]
                            for i in Counter(
                                dict_insight_weather["WeatherHourly"].values()
                            ).most_common(top_weather_plot)
                        ]
                        df_weather_daily_plot = df_weather_daily_plot[
                            df_weather_daily_plot["Weather"].isin(list_weather_plot)
                        ]
                        fig = px.pie(
                            df_weather_daily_plot,
                            values="NoDay",
                            names="Weather",
                            title="Pie Chart of Hourly Weather",
                            hover_data=["Weather"],
                            labels={"Weather": "Kiểu thời tiết"},
                        )
                        fig.update_layout(width=400, height=400, hovermode="x unified")
                        st.plotly_chart(
                            fig, theme="streamlit", use_container_width=True
                        )
                    with column_plot_2:
                        df_temp_daily_plot = pd.DataFrame(
                            {
                                "TempC": list(
                                    dict_insight_weather["TempHourly"].values()
                                ),
                                "DateHour": list(
                                    dict_insight_weather["TempHourly"].keys()
                                ),
                            }
                        )
                        df_temp_daily_plot["Hour"] = df_temp_daily_plot[
                            "DateHour"
                        ].apply(lambda x: x.split(" ")[1])
                        df_temp_daily_agg_plot = (
                            df_temp_daily_plot.groupby("Hour")["TempC"]
                            .mean()
                            .reset_index()
                        )
                        list_hour_plot = [
                            "10:00",
                            "11:00",
                            "12:00",
                            "13:00",
                            "14:00",
                            "18:00",
                            "19:00",
                            "20:00",
                            "21:00",
                            "22:00",
                        ]
                        df_temp_daily_agg_plot = df_temp_daily_agg_plot[
                            df_temp_daily_agg_plot["Hour"].isin(list_hour_plot)
                        ]
                        fig = px.bar(
                            df_temp_daily_agg_plot,
                            x="Hour",
                            y="TempC",
                            title="Bar Chart of Hourly Temperature",
                            text="TempC",
                            color=["Lunch"] * 5 + ["Dinner"] * 5,
                            hover_data=["TempC"],
                            labels={"TempC": "Nhiệt độ"},
                        )
                        fig.update_traces(
                            texttemplate="%{text:.2s}", textposition="inside"
                        )
                        fig.update_layout(
                            uniformtext_minsize=6,
                            uniformtext_mode="hide",
                            hovermode="x unified",
                            height=375,
                        )
                        st.plotly_chart(
                            fig, theme="streamlit", use_container_width=True
                        )
                    return
            if columns_mode == "Single II":
                with columns_insight[1]:
                    top_same_event_decor_selected = top_same_event_decor_after[
                        top_same_event_decor_after["category"] == "✔️"
                    ]
                    if top_same_event_decor_selected.shape[0] == 0:
                        st.error(f"No Event Selected", icon="🚨")
                        st.error(
                            f"Kiểm tra lại cấu hình mô hình sự kiện. Gợi ý: thay đổi mục tiêu hoặc giảm % Increment"
                        )
                        time.sleep(180)
                    list_event_selected = list(
                        zip(
                            top_same_event_decor_selected["Start Event"],
                            top_same_event_decor_selected["End Event"],
                            top_same_event_decor_selected["Same"],
                        )
                    )

                    if not is_check_campaign_after:
                        # Transaction during
                        list_df_event_i = []
                        for event_i in list_event_selected:
                            df_event_i = df_transactions[
                                (df_transactions.ShiftDate >= event_i[0])
                                & (df_transactions.ShiftDate <= event_i[1])
                            ]
                            df_event_i["EventName"] = event_i[2]
                            list_df_event_i.append(df_event_i)
                        df_transactions_event_selected = pd.concat(
                            list_df_event_i, axis=0
                        )
                    else:
                        # Transaction after Event
                        top_same_event_AFTER_selected = (
                            top_same_event_decor_selected.copy()
                        )
                        top_same_event_AFTER_selected[
                            "End Event After"
                        ] = top_same_event_AFTER_selected["End Event"].apply(
                            lambda x: datetime.strftime(
                                datetime.strptime(x, "%Y-%m-%d")
                                - timedelta(-duration_after),
                                "%Y-%m-%d",
                            )
                        )
                        list_event_selected_after = list(
                            zip(
                                top_same_event_AFTER_selected["End Event"],
                                top_same_event_AFTER_selected["End Event After"],
                                top_same_event_AFTER_selected["Same"],
                            )
                        )
                        list_df_event_after_i = []

                        for event_i in list_event_selected_after:
                            df_event_i = df_transactions[
                                (df_transactions.ShiftDate >= event_i[0])
                                & (df_transactions.ShiftDate <= event_i[1])
                            ]
                            df_event_i["EventName"] = event_i[2]
                            list_df_event_after_i.append(df_event_i)
                        df_transactions_event_selected = pd.concat(
                            list_df_event_after_i, axis=0
                        )

                    # Load File Manual
                    try:
                        # if setting_load["brand_selected"] == "Manwah":
                        #     df_manual = pd.read_excel(
                        #         base_path
                        #         + static_load[
                        #             f"mkt_campaign_manual_MW_{setting_load['sbu_selected']}"
                        #         ],
                        #         sheet_name=0,
                        #     )
                        #     if "Mã chương trình" not in df_manual.columns:
                        #         df_manual = pd.read_excel(
                        #             base_path
                        #             + static_load[
                        #                 f"mkt_campaign_manual_MW_{setting_load['sbu_selected']}"
                        #             ],
                        #             sheet_name=1,
                        #         )
                        # elif setting_load["brand_selected"] == "Gogi House":
                        #     df_manual = pd.read_excel(
                        #         base_path
                        #         + static_load[
                        #             f"mkt_campaign_manual_GG_{setting_load['sbu_selected']}"
                        #         ],
                        #         sheet_name=0,
                        #     )
                        #     if "Mã chương trình" not in df_manual.columns:
                        #         df_manual = pd.read_excel(
                        #             base_path
                        #             + static_load[
                        #                 f"mkt_campaign_manual_MW_{setting_load['sbu_selected']}"
                        #             ],
                        #             sheet_name=1,
                        #         )
                        # elif setting_load["brand_selected"] == "CloudPot":
                        #     df_manual = pd.read_excel(
                        #         base_path
                        #         + static_load[
                        #             f"mkt_campaign_manual_CP_{setting_load['sbu_selected']}"
                        #         ],
                        #         sheet_name=0,
                        #     )
                        #     if "Mã chương trình" not in df_manual.columns:
                        #         df_manual = pd.read_excel(
                        #             base_path
                        #             + static_load[
                        #                 f"mkt_campaign_manual_CP_{setting_load['sbu_selected']}"
                        #             ],
                        #             sheet_name=1,
                        #         )
                        # elif setting_load["brand_selected"] == "Crystal Jade":
                        #     df_manual = pd.read_excel(
                        #         base_path
                        #         + static_load[
                        #             f"mkt_campaign_manual_CJ_{setting_load['sbu_selected']}"
                        #         ],
                        #         sheet_name=0,
                        #     )
                        #     if "Mã chương trình" not in df_manual.columns:
                        #         df_manual = pd.read_excel(
                        #             base_path
                        #             + static_load[
                        #                 f"mkt_campaign_manual_CJ_{setting_load['sbu_selected']}"
                        #             ],
                        #             sheet_name=1,
                        #         )
                        df_manual = pd.read_csv(static_load['manual_mkt_input_MIS_path'], sep="\t")
                        campaign_brand_code = static_load["BrandCode_data"][setting_load['brand_selected']]
                        is_manual_contain_campaign_brand_code = any([(campaign_brand_code in i) 
                                                                     for i in list(set(df_manual['Mã chương trình']))]
                                                                   )
                        if is_manual_contain_campaign_brand_code:
                            manual_columns = [
                                "Mã chương trình",
                                "Mã sub campaign",
                                "Loại Chương trình",
                                "Ngày bắt đầu",
                                "Ngày kết thúc",
                                "Target Deduction",
                                "Target Sales (trước cani)",
                                "Cani% target",
                                "% Redemption Target",
                            ]
                            renamed_manual_columns = [
                                "Mã chương trình",
                                "Mã sub campaign",
                                "Loại Chương trình",
                                "Ngày bắt đầu",
                                "Ngày kết thúc",
                                "Target Deduction",
                                "Target Sales",
                                "Target %Cani",
                                "Target %Redemption",
                            ]
                            df_manual = df_manual[manual_columns]
                            df_manual.columns = renamed_manual_columns
                        else:
                            st.error(
                                f"Dictionary is not cover Brand {setting_load['brand_selected']} in this Phase",
                                icon="🚨",
                            )
                    except:
                        st.error(
                            f"No file Manual for Brand: {setting_load['brand_selected']}",
                            icon="🚨",
                        )
                        time.sleep(60)


                    # Handling wrong time value (user input)
                    df_manual["Ngày bắt đầu"] = df_manual["Ngày bắt đầu"].apply(
                        lambda x: handing_wrong_datetime_format(x)
                    )
                    df_manual["Ngày kết thúc"] = df_manual["Ngày kết thúc"].apply(
                        lambda x: handing_wrong_datetime_format(x)
                    )
                    df_manual = df_manual[df_manual["Ngày bắt đầu"] != ""]
                    df_manual = df_manual[df_manual["Ngày kết thúc"] != ""]

                    # Handling null sub.campaign (user input)
                    df_manual["Mã sub campaign"] = df_manual.apply(
                        lambda x: (
                            x["Mã sub campaign"]
                            if str(x["Mã sub campaign"]) not in ["NaN", "nan", ""]
                            else str(x["Mã chương trình"]) + ".1"
                        ),
                        axis=1,
                    )

                    # Handling wrong input code (user input)
                    df_manual_final = df_manual.copy()
                    df_manual_final["Mã chương trình"] = df_manual_final[
                        "Mã chương trình"
                    ].apply(lambda x: get_voucher_code_format_1(x))
                    df_manual_final["Mã sub campaign"] = df_manual_final[
                        "Mã sub campaign"
                    ].apply(lambda x: get_voucher_code_format_2(x))
                    df_manual_final.dropna(
                        subset=["Mã chương trình", "Mã sub campaign"], inplace=True
                    )

                    # Agg MKT campaign in selected event
                    def get_voucher_code_info(x):
                        result = {}
                        event_name = x["EventName"]
                        voucher_name = x["VoucherName"]
                        result["Event Name"] = Counter(event_name).most_common(1)[0][0]
                        result["Voucher Name"] = Counter(voucher_name).most_common(1)[
                            0
                        ][0]
                        return pd.Series(result)

                    # Filter campaign with brand code and list manual only
                    def filter_BrandCode_program(
                        brand_code, voucher_code, list_voucher_code_brand=[]
                    ):
                        list_sub_string = [brand_code] + list_voucher_code_brand
                        list_sub_string = [str(i) for i in list_sub_string]
                        for st_i in list_sub_string:
                            if st_i[1:] in voucher_code:
                                return True
                        return False

                    ## Transform Voucher Code to same format as Manual, remove non Voucher Name
                    df_transactions_event_selected["VoucherCode"] = (
                        df_transactions_event_selected["VoucherCode"].apply(
                            lambda x: get_voucher_code_format_1(x)
                        )
                    )
                    df_MKT_in_event = (
                        df_transactions_event_selected.groupby("VoucherCode")
                        .apply(get_voucher_code_info)
                        .reset_index()
                    )
                    df_MKT_in_event.dropna(
                        subset=["VoucherCode", "Voucher Name"], inplace=True
                    )

                    st.markdown(
                        f"""
                        + **Danh sách chương trình MKT:**:
                    """
                    )
                    list_voucher_code_brand = list(
                        set(df_manual_final["Mã chương trình"])
                    )
                    dict_brand_code = static_load["BrandCode_data"]
                    brand_code = dict_brand_code[setting_load["brand_selected"]]
                    df_MKT_in_event = df_MKT_in_event[
                        df_MKT_in_event.apply(
                            lambda x: filter_BrandCode_program(
                                brand_code, x["VoucherCode"], list_voucher_code_brand
                            ),
                            axis=1,
                        )
                    ]
                    df_MKT_in_event = df_MKT_in_event[
                        df_MKT_in_event.apply(
                            lambda x: filter_staff(x["Voucher Name"]), axis=1
                        )
                    ]

                    df_MKT_in_event["Event Top"] = df_MKT_in_event["Event Name"].apply(
                        lambda x: int(str(x).split(" ")[1])
                    )
                    df_MKT_in_event = df_MKT_in_event.sort_values(
                        by=["Event Top"], ascending=True
                    ).drop(columns=["Event Top"])
                    st.dataframe(df_MKT_in_event, height=240, hide_index=True, use_container_width=True)

                    ## Join MKT Campaign Information in Manual Handbook and actual Transactions
                    df_MKT_in_event_copy = df_MKT_in_event.rename(
                        columns={"VoucherCode": "Mã chương trình"}
                    )

                    def convert_voucher_name(voucher_name):
                        # Remove campaign code from voucher name
                        if "_" in voucher_name:
                            return voucher_name.split("_")[1]
                        else:
                            return voucher_name

                    def group_campaign(x):
                        # Group by campaign Name
                        result = {}
                        result["Mã sub campaign"] = ", ".join(x["Mã sub campaign"])
                        result["Target Deduction"] = np.sum(x["Target Deduction"]) 
                        result["Target Sales"] = np.sum(x["Target Sales"]) 
                        result["Target %Cani"] = np.mean(x["Target %Cani"])
                        result["Target %Redemption"] = np.mean(x["Target %Redemption"])
                        return pd.Series(result)

                    df_MKT_in_event_copy["Tên chương trình"] = df_MKT_in_event_copy[
                        "Voucher Name"
                    ].apply(lambda x: convert_voucher_name(x))
                    df_MKT_in_event_handbook = pd.merge(
                        df_MKT_in_event_copy[
                            ["Mã chương trình", "Tên chương trình", "Event Name"]
                        ],
                        df_manual_final,
                        on="Mã chương trình",
                        how="left",
                    )
                    # Get mkt camp not in hand book information
                    df_MKT_not_in_event_handbook = df_MKT_in_event_handbook[
                        pd.isnull(df_MKT_in_event_handbook["Loại Chương trình"])
                    ]
                    df_MKT_transaction_not_in_event_handbook = (
                        df_transactions_event_selected[
                            df_transactions_event_selected["VoucherCode"].isin(
                                df_MKT_not_in_event_handbook["Mã chương trình"]
                            )
                        ]
                    )
                    df_MKT_not_in_event_handbook_time = (
                        df_MKT_transaction_not_in_event_handbook.groupby("VoucherCode")
                        .agg({"ShiftDate": ["min", "max"]})
                        .reset_index()
                    )
                    df_MKT_not_in_event_handbook_time.columns = [
                        "Mã chương trình",
                        "Ngày bắt đầu",
                        "Ngày kết thúc",
                    ]
                    df_MKT_not_in_event_handbook_time[
                        "Ngày bắt đầu"
                    ] = df_MKT_not_in_event_handbook_time["Ngày bắt đầu"].apply(
                        lambda x: datetime.strptime(
                            str(x) + " 00:00:00", "%Y-%m-%d %H:%M:%S"
                        )
                    )
                    df_MKT_not_in_event_handbook_time[
                        "Ngày kết thúc"
                    ] = df_MKT_not_in_event_handbook_time["Ngày kết thúc"].apply(
                        lambda x: datetime.strptime(
                            str(x) + " 00:00:00", "%Y-%m-%d %H:%M:%S"
                        )
                    )
                    df_MKT_not_in_event_handbook_information = (
                        df_MKT_not_in_event_handbook.drop(
                            columns=["Ngày bắt đầu", "Ngày kết thúc"]
                        )
                    )
                    df_MKT_not_in_event_handbook_information = pd.merge(
                        df_MKT_not_in_event_handbook_information,
                        df_MKT_not_in_event_handbook_time,
                        how="left",
                        on="Mã chương trình",
                    )
                    df_MKT_not_in_event_handbook_information["Loại Chương trình"] = (
                        "NO INFORMATION"
                    )
                    df_MKT_not_in_event_handbook_information["Mã sub campaign"] = (
                        "NO INFORMATION"
                    )
                    df_MKT_not_in_event_handbook_information["Target Deduction"] = (
                        "NO INFORMATION"
                    )
                    df_MKT_not_in_event_handbook_information["Target Sales"] = (
                        "NO INFORMATION"
                    )
                    df_MKT_not_in_event_handbook_information["Target %Cani"] = (
                        "NO INFORMATION"
                    )
                    df_MKT_not_in_event_handbook_information["Target %Redemption"] = (
                        "NO INFORMATION"
                    )

                    # Get only mkt camp in handbook
                    ## Note that: Filted
                    df_MKT_in_event_handbook_information = (
                        df_MKT_in_event_handbook.dropna(
                            subset=[
                                "Mã sub campaign",
                                "Loại Chương trình",
                                "Ngày bắt đầu",
                                "Ngày kết thúc",
                            ]
                        )
                    )
                    # Merge MKT full
                    df_MKT_full_information = pd.concat(
                        [
                            df_MKT_not_in_event_handbook_information,
                            df_MKT_in_event_handbook_information,
                        ],
                        axis=0,
                    )

                    if df_MKT_in_event_handbook_information.shape[0] > 0:
                        df_MKT_in_event_handbook_information_agg = (
                            df_MKT_in_event_handbook_information.groupby(
                                [
                                    "Mã chương trình",
                                    "Tên chương trình",
                                    "Event Name",
                                    "Loại Chương trình",
                                    "Ngày bắt đầu",
                                    "Ngày kết thúc",
                                ]
                            )
                            .apply(group_campaign)
                            .reset_index()
                        )
                    else:
                        df_MKT_in_event_handbook_information_agg = (
                            df_MKT_in_event_handbook_information
                        )
                        st.error(
                            "There is no campaign defined in manual file. Get actual data ..",
                            icon="🚨",
                        )

                    handbook_agg_columns = [
                        "Mã chương trình",
                        "Mã sub campaign",
                        "Tên chương trình",
                        "Loại Chương trình",
                        "Ngày bắt đầu",
                        "Ngày kết thúc",
                        "Target Deduction",
                        "Target Sales",
                        "Target %Cani",
                        "Target %Redemption",
                    ]
                    df_MKT_in_event_handbook_information_agg = (
                        df_MKT_in_event_handbook_information_agg[handbook_agg_columns]
                    )

                    df_MKT_full_information_agg = pd.concat(
                        [
                            df_MKT_in_event_handbook_information_agg,
                            df_MKT_not_in_event_handbook_information[
                                handbook_agg_columns
                            ],
                        ],
                        axis=0,
                    )

                    df_MKT_in_event_detail = df_MKT_full_information_agg[
                        [
                            "Mã chương trình",
                            "Mã sub campaign",
                            "Tên chương trình",
                            "Ngày bắt đầu",
                            "Ngày kết thúc",
                            "Loại Chương trình",
                        ]
                    ]

                    def save_dataframe_to_image(data):
                        fig = ff.create_table(data)
                        fig.update_layout(autosize=False, width=1200, font={"size": 8})
                        fig.write_image(
                            base_path + "apps/picture/campaigns_dictionary.png", scale=2
                        )

                    # save_dataframe_to_image(df_MKT_in_event_detail)
                    df_MKT_in_event_detail.to_excel(
                        base_path + "dataset/SameEvent_MKT_KPI_INFO.xlsx"
                    )

                    def show_campaign_info():
                        df_detail = pd.DataFrame(
                            {
                                "Campain Detail": [
                                    base_path + "apps/picture/campaigns_dictionary.png"
                                ]
                                # "Campain Detail": ["https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/5435b8cb-6c6c-490b-9608-799b543655d3/Home_Page.png"]
                            }
                        )
                        st.dataframe(
                            df_detail,
                            column_config={
                                "Campain Detail": st.column_config.ImageColumn(
                                    "🔎 Campain Info 🔎",
                                    help="Click to see Campaign Information",
                                )
                            },
                            hide_index=True,
                        )

                    # st.markdown(f"""
                    # -
                    #     ---
                    #     + **MKT Campaign - Stats:**
                    # """)
                    # column_plot_mkt_1, column_plot_mkt_2 = st.columns([1,1])
                    # with column_plot_mkt_1:
                    #     st.metric(label="Total MKT Campaign (Data)",value=len(set(df_MKT_in_event_handbook["Mã chương trình"])))
                    #     st.metric(label="Total MKT Sub-Campaign (Data)",value=len(set(df_MKT_in_event_handbook["Mã sub campaign"])))
                    # with column_plot_mkt_2:
                    #     st.metric(label="Total MKT Campaign (HandBook)",value=len(set(df_MKT_in_event_handbook_information["Mã chương trình"])))
                    #     st.metric(label="Total MKT Sub-Campaign (HandBook)",value=len(set(df_MKT_in_event_handbook_information["Mã sub campaign"])))
                    column_plot_mkt_3, column_plot_mkt_4 = st.columns([1, 1])
                    if df_MKT_in_event_handbook_information.shape[0] > 0:
                        with column_plot_mkt_3:
                            df_mkt_information_target = (
                                df_MKT_in_event_handbook_information[
                                    [
                                        "Mã chương trình",
                                        "Mã sub campaign",
                                        "Loại Chương trình",
                                    ]
                                ]
                            )
                            df_mkt_information_target["SL Sub Campaign"] = 1
                            fig = px.pie(
                                df_mkt_information_target,
                                values="SL Sub Campaign",
                                names="Loại Chương trình",
                                title="Pie Chart of MKT Campaign/Sub Campaign | Target",
                                hover_data=["Mã sub campaign"],
                            )
                            fig.update_layout(hovermode="x unified")
                            # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                        with column_plot_mkt_4:
                            import plotly.graph_objects as go

                            top_event_agg = (
                                df_MKT_in_event_handbook_information.groupby(
                                    "Event Name"
                                )
                                .agg(
                                    {
                                        "Mã chương trình": "nunique",
                                        "Mã sub campaign": "nunique",
                                    }
                                )
                                .reset_index()
                            )

                            fig = go.Figure(
                                data=[
                                    go.Bar(
                                        name="Mã campaign",
                                        x=top_event_agg["Event Name"],
                                        y=top_event_agg["Mã chương trình"],
                                    ),
                                    go.Bar(
                                        name="Mã sub campaign",
                                        x=top_event_agg["Event Name"],
                                        y=top_event_agg["Mã sub campaign"],
                                    ),
                                ]
                            )
                            # Change the bar mode
                            fig.update_layout(barmode="group", hovermode="x unified")
                            # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                    else:
                        df_mkt_information_target = (
                            df_MKT_in_event_handbook_information[
                                [
                                    "Mã chương trình",
                                    "Mã sub campaign",
                                    "Loại Chương trình",
                                ]
                            ]
                        )
                        df_mkt_information_target["SL Sub Campaign"] = 1
                        st.error("No manual data! No information to plot !", icon="🚨")

                    df_mkt_information_target_to_pick = (
                        df_mkt_information_target.groupby("Loại Chương trình")
                        .agg(
                            {"Mã chương trình": "nunique", "Mã sub campaign": "nunique"}
                        )
                        .reset_index()
                    )
                    df_mkt_information_target_to_pick.columns = [
                        "Mục tiêu",
                        "SL Campaign",
                        "SL Sub Campaign",
                    ]
                    df_mkt_information_target_to_pick.sort_values(
                        by=["SL Sub Campaign"], ascending=False, inplace=True
                    )
                    df_mkt_information_target_to_pick["Lựa chọn mục tiêu"] = "✔️"

                    st.toast(body="Chọn Mục tiêu", icon="⬇️")
                    with st.form(key=key_multiselect + "form_campaign"):
                        df_mkt_information_target_picked = st.data_editor(
                            df_mkt_information_target_to_pick,
                            column_config={
                                "Lựa chọn mục tiêu": st.column_config.SelectboxColumn(
                                    "Lựa chọn mục tiêu",
                                    help="Chọn ✔️ để lựa chọn thống kê campaign theo mục tiêu",
                                    default="✔️",
                                    options=["✔️", "❌"],
                                )
                            },
                            hide_index=True,
                            use_container_width=True,
                            key=key_multiselect + "target_data_editor",
                        )
                        submit_button = st.form_submit_button(
                            label=":red[SUBMIT] - camp target (✔️: Target to Statistic, ❌: Target to Ignore)"
                        )
                    ## Filter Campaign Target
                    campaign_target_picked = list(
                        df_mkt_information_target_picked[
                            df_mkt_information_target_picked["Lựa chọn mục tiêu"] == "✔️"
                        ]["Mục tiêu"]
                    )
                    if (submit_button == True) or (
                        len(campaign_target_picked)
                        != df_mkt_information_target_to_pick.shape[0]
                    ):
                        st.toast(
                            body=f'Đã chọn Mục tiêu: {", ".join(campaign_target_picked)}',
                            icon="✅",
                        )
                    else:
                        st.toast(
                            body=f"Đã chọn Mục tiêu (*Mặc định toàn bộ mục tiêu*)",
                            icon="✅",
                        )
                    if len(campaign_target_picked) > 0:
                        df_MKT_full_information_agg = df_MKT_full_information_agg[
                            df_MKT_full_information_agg["Loại Chương trình"].isin(
                                campaign_target_picked
                            )
                        ]
                    else:
                        st.error(
                            "No manual data ! Campaign's target can not be selected !",
                            icon="🚨",
                        )
                        df_MKT_full_information_agg = df_MKT_full_information_agg
                    df_MKT_full_information = df_MKT_full_information[
                        df_MKT_full_information["Mã chương trình"].isin(
                            df_MKT_full_information_agg["Mã chương trình"]
                        )
                    ]

                    st.markdown(
                        f"""
                        + **Thông tin chi tiết chương trình MKT đề xuất:**
                    """
                    )
                    st.dataframe(
                        df_MKT_full_information_agg,
                        height=220,
                        hide_index=True,
                        use_container_width=True,
                    )
    with st.spinner(f"Đang tổng hợp các chỉ số KPI ..."):
        with st.expander("👉 Hiệu quả của TradeMKT ", expanded=False):
            st.markdown(
                f"""
                ---
                + **MKT Campaign - KPI:**
            """
            )
            list_kpi = [
                "1.Sales MKT Contribution",
                "2.TC MKT Contribution",
                "3.Incremental Sales",
                "4.Incremental TC",
                "5.Incremental Cost",
                "6.Incremental Profit Margin",
                "7.TA MKT",
                "8.TA W/o MKT",
                "9.Customer Size MKT",
                "10.Customer Size w/o MKT",
                "11.% Deduction",
                "12.%MKT Fee",
                "13.ME/RE",
                "14.MKT Cost/TC",
                "15.Cost per Actions",
            ]
            list_kpi_done = [
                "1.Sales MKT Contribution",
                "2.TC MKT Contribution",
                "3.Incremental Sales",
                "4.Incremental TC",
                "7.TA MKT",
                "8.TA W/o MKT",
                "9.Customer Size MKT",
                "10.Customer Size w/o MKT",
                "11.% Deduction",
                "12.%MKT Fee",
                "13.ME/RE",
                "14.MKT Cost/TC",
            ]

            df_transactions["VoucherCode"] = df_transactions["VoucherCode"].apply(
                lambda x: get_voucher_code_format_1(x)
            )
            kpi_type_choice = st.multiselect(
                label="Select KPI",
                options=list_kpi,
                default=list_kpi_done,
                key=key_multiselect,
            )
            list_kpi_show = [i for i in kpi_type_choice if i in list_kpi_done]
            if len(list_kpi_show) == 0:
                st.error(
                    f"KPI Selected is not cal yet! Choose in this list {list_kpi_done}"
                )
                time.sleep(60)

            tuple_sub_campaign = list(
                zip(
                    df_MKT_full_information["Mã sub campaign"],
                    df_MKT_full_information["Ngày bắt đầu"],
                    df_MKT_full_information["Ngày kết thúc"],
                    df_MKT_full_information["Mã chương trình"],
                )
            )
            tuple_campaign = list(
                zip(
                    df_MKT_full_information["Ngày bắt đầu"],
                    df_MKT_full_information["Ngày kết thúc"],
                    df_MKT_full_information["Mã chương trình"],
                )
            )
            cm_kpi = sns.light_palette("green", as_cmap=True)
            list_df_performance = []
            # 1.Sales MKT Contribution
            if list_kpi[0] in list_kpi_show:
                st.markdown(f"""**:red[1. Sales MKT Contribution]**""")
                dict_performance_1 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Sale MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]
                    df_transaction_mkt_agg_i = (
                        df_transaction_mkt_i.groupby("ShiftDate")["PriceSum"]
                        .sum()
                        .reset_index()
                    )

                    # Sale Brand (in duration of MKT and in day Brand has MKT action)
                    df_transaction_full_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            df_transactions.ShiftDate.isin(
                                df_transaction_mkt_agg_i["ShiftDate"]
                            )
                        )
                    ]
                    df_transaction_full_agg_i = (
                        df_transaction_full_i.groupby("ShiftDate")["PriceSum"]
                        .sum()
                        .reset_index()
                    )

                    # Sale Store (in duration of MKT and in day Store has MKT action)
                    df_transaction_store_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            df_transactions.ShiftDate.isin(
                                df_transaction_mkt_agg_i["ShiftDate"]
                            )
                        )
                        & (
                            df_transactions.RestaurantCode.isin(
                                df_transaction_mkt_i["RestaurantCode"]
                            )
                        )
                    ]
                    df_transaction_store_agg_i = (
                        df_transaction_store_i.groupby("ShiftDate")["PriceSum"]
                        .sum()
                        .reset_index()
                    )

                    if campaign_name_i not in dict_performance_1.keys():
                        total_sale = np.sum(df_transaction_full_agg_i.PriceSum)
                        total_sale_store = np.sum(df_transaction_store_agg_i.PriceSum)
                        total_sale_mkt = np.sum(df_transaction_mkt_agg_i.PriceSum)
                        dict_performance_1[f"{campaign_name_i}"] = {
                            "%Contribute Sale": (
                                round(total_sale_mkt / total_sale, 4)
                                if total_sale != 0
                                else 0
                            ),
                            "%Contribute Sale (Store)": (
                                round(total_sale_mkt / total_sale_store, 4)
                                if total_sale_store != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_1 = pd.DataFrame(dict_performance_1).T
                # st.dataframe(df_performance_plot_1.style.format("{:.2%}").highlight_max(axis=0).background_gradient(cmap=cm_kpi),
                #             use_container_width=True)
                st.dataframe(
                    df_performance_plot_1.sort_values(
                        by=["%Contribute Sale"], ascending=False
                    ).style.format("{:.2%}"),
                    use_container_width=True,
                )
                # show_campaign_info()
                list_df_performance.append(df_performance_plot_1)

            if list_kpi[1] in list_kpi_show:
                st.markdown(f"""**:red[2. TC MKT Contribution]**""")
                dict_performance_2 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Sale MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]
                    df_transaction_mkt_agg_i = (
                        df_transaction_mkt_i.groupby("ShiftDate")["GuestCount"]
                        .sum()
                        .reset_index()
                    )

                    # Sale Brand (in duration of MKT and in day Brand has MKT action)
                    df_transaction_full_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            df_transactions.ShiftDate.isin(
                                df_transaction_mkt_agg_i["ShiftDate"]
                            )
                        )
                    ]
                    df_transaction_full_agg_i = (
                        df_transaction_full_i.groupby("ShiftDate")["GuestCount"]
                        .sum()
                        .reset_index()
                    )

                    # Sale Store (in duration of MKT and in day Store has MKT action)
                    df_transaction_store_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            df_transactions.ShiftDate.isin(
                                df_transaction_mkt_agg_i["ShiftDate"]
                            )
                        )
                        & (
                            df_transactions.RestaurantCode.isin(
                                df_transaction_mkt_i["RestaurantCode"]
                            )
                        )
                    ]
                    df_transaction_store_agg_i = (
                        df_transaction_store_i.groupby("ShiftDate")["GuestCount"]
                        .sum()
                        .reset_index()
                    )

                    if campaign_name_i not in dict_performance_2.keys():
                        total_tc = np.sum(df_transaction_full_agg_i.GuestCount)
                        total_tc_store = np.sum(df_transaction_store_agg_i.GuestCount)
                        total_tc_mkt = np.sum(df_transaction_mkt_agg_i.GuestCount)
                        dict_performance_2[f"{campaign_name_i}"] = {
                            "%Contribute TC": (
                                round(total_tc_mkt / total_tc, 4)
                                if total_tc != 0
                                else 0
                            ),
                            "%Contribute TC (Store)": (
                                round(total_tc_mkt / total_tc_store, 4)
                                if total_tc_store != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_2 = pd.DataFrame(dict_performance_2).T
                st.dataframe(
                    df_performance_plot_2.sort_values(
                        by=["%Contribute TC"], ascending=False
                    ).style.format("{:.2%}"),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_2)

            if list_kpi[2] in list_kpi_show:
                st.markdown(f"""**:red[3. Incremental Sales]**""")
                dict_performance_3 = {}
                dict_brand_code = static_load["BrandCode_data"]
                brand_code = dict_brand_code[setting_load["brand_selected"]]
                sbu_selected = setting_load["sbu_selected"]
                brand_selected = setting_load["brand_selected"]
                # st.write(baseline_brand_path)
                try:
                    # Need to confirm and automation baseline model.
                    ## Currently only accept some brand that caculated baseline
                    baseline_brand_path = static_load[
                        f"baseline_sale_{brand_code}_{sbu_selected}"
                    ]
                    df_sale_baseline_reload = pd.read_csv(
                        base_path + baseline_brand_path, sep=","
                    )
                    if setting_load["restaurant_selected"] == "ALL":
                        is_plot_sale_baseline_chart = st.checkbox(
                            "Plot Baseline and Actual Sale",
                            value=False,
                            key=key_multiselect + "checkbox",
                        )
                        for tuple_i in tuple_sub_campaign:
                            start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                            end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                            sub_campaign_name_i = tuple_i[0]
                            campaign_name_i = tuple_i[3]
                            campaign_name_i_mode = tuple_i[3][1:]
                            if is_plot_sale_baseline_chart:
                                st.markdown(
                                    f"""
                                ###### :red[Sale Baseline Plot {campaign_name_i} - {sub_campaign_name_i}]
                                """
                                )
                            df_sale_baseline_reload_i = df_sale_baseline_reload[
                                (
                                    df_sale_baseline_reload.ShiftDate.astype(str)
                                    >= start_time_i
                                )
                                & (
                                    df_sale_baseline_reload.ShiftDate.astype(str)
                                    <= end_time_i
                                )
                            ]
                            df_transaction_reload_i = df_transactions[
                                (df_transactions.ShiftDate.astype(str) >= start_time_i)
                                & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                                & (
                                    (
                                        df_transactions.VoucherCode.astype(str)
                                        == campaign_name_i
                                    )
                                    | (
                                        df_transactions.VoucherCode.astype(str)
                                        == campaign_name_i_mode
                                    )
                                )
                            ]
                            df_sale_reload_i = df_transactions[
                                (df_transactions.ShiftDate.astype(str) >= start_time_i)
                                & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                            ]

                            # st.dataframe(df_sale_baseline_reload_i)
                            # st.dataframe(df_transaction_reload_i)
                            df_transaction_agg_i = (
                                df_transaction_reload_i.groupby("ShiftDate")["PriceSum"]
                                .sum()
                                .reset_index()
                            )
                            df_transaction_agg_i.sort_values(
                                by=["ShiftDate"], ascending=True
                            )

                            df_transaction_sale_baseline = df_sale_baseline_reload
                            df_sale_i = df_sale_baseline_reload_i[["ShiftDate", "Sale"]]
                            df_sale_i["SaleType"] = "Sale"
                            df_sale_baseline_i = df_sale_baseline_reload_i[
                                ["ShiftDate", "Sale_baseline"]
                            ].rename(columns={"Sale_baseline": "Sale"})
                            df_sale_baseline_i["SaleType"] = "Sale Baseline"
                            df_sale_baseline_i["Sale"] = df_sale_baseline_i[
                                "Sale"
                            ].apply(lambda x: round(x))
                            df_sale_mkt_i = df_transaction_agg_i[
                                ["ShiftDate", "PriceSum"]
                            ].rename(columns={"PriceSum": "Sale"})
                            df_sale_mkt_i["SaleType"] = "Sale MKT"
                            df_sale_mkt_i["Sale"] = df_sale_mkt_i["Sale"].apply(
                                lambda x: round(x)
                            )

                            if is_plot_sale_baseline_chart:
                                df_sale_baseline_pivot_i = pd.concat(
                                    [df_sale_i, df_sale_baseline_i, df_sale_mkt_i],
                                    axis=0,
                                )
                                fig = px.area(
                                    df_sale_baseline_pivot_i,
                                    x="ShiftDate",
                                    y="Sale",
                                    color="SaleType",
                                    category_orders={
                                        "SaleType": [
                                            "Sale",
                                            "Sale Baseline",
                                            "Sale MKT",
                                        ]
                                    },
                                    color_discrete_sequence=[
                                        "#00429d",
                                        "#cdf1e0",
                                        "red",
                                    ],
                                    pattern_shape="SaleType",
                                    pattern_shape_sequence=[".", "+", "-"],
                                )
                                # fig.add_trace(go.Scatter(x=df_transaction_mkt_agg["ShiftDate"], fill=None, y=df_transaction_mkt_agg["TC (Diff)"], name="TC (Diff)"))
                                fig.update_layout(
                                    width=1500,
                                    height=500,
                                    hovermode="x unified",
                                    title="Sale Baseline Plot",
                                )
                                st.plotly_chart(
                                    fig, theme="streamlit", use_container_width=True
                                )
                            if campaign_name_i not in dict_performance_3.keys():
                                total_sale_baseline = round_money(
                                    np.sum(df_sale_baseline_i.Sale), 6
                                )
                                total_sale_mkt = round_money(
                                    np.sum(df_sale_mkt_i.Sale), 6
                                )
                                total_sale = round_money(
                                    np.sum(df_sale_reload_i.PriceSum), 6
                                )
                                dict_performance_3[f"{campaign_name_i}"] = {
                                    "Incremental Sale": round_money(
                                        total_sale - total_sale_baseline, 6
                                    ),
                                    "Baseline Sale": round_money(
                                        total_sale_baseline, 6
                                    ),
                                    "Sale": round_money(total_sale, 6),
                                    "%Incremental Sale": (
                                        round(
                                            total_sale_mkt
                                            / (total_sale_mkt + total_sale_baseline),
                                            4,
                                        )
                                        if total_sale_mkt + total_sale_baseline > 0
                                        else 0
                                    ),
                                }
                except:
                    st.error(
                        f"Increamental Sale/TC Support for Brand Selection Only ! Your settings: {setting_load['brand_selected']} & {setting_load['restaurant_selected']}",
                        icon="🚨",
                    )
                    for tuple_i in tuple_sub_campaign:
                        start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                        end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                        sub_campaign_name_i = tuple_i[0]
                        campaign_name_i = tuple_i[3]
                        campaign_name_i_mode = tuple_i[3][1:]
                        dict_performance_3[f"{campaign_name_i}"] = {
                            "Incremental Sale": np.nan,
                            "Baseline Sale": np.nan,
                            "Sale": np.nan,
                            "%Incremental Sale": np.nan,
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_3 = pd.DataFrame(dict_performance_3).T
                st.dataframe(
                    df_performance_plot_3.sort_values(
                        by=["Incremental Sale"], ascending=False
                    ).style.format(
                        {
                            "%Incremental Sale": "{:.2%}",
                            "Sale": "{:,.0f}",
                            "Baseline Sale": "{:,.0f}",
                            "Incremental Sale": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_3)

            if list_kpi[3] in list_kpi_show:
                st.markdown(f"""**:red[4. Incremental TC]**""")
                dict_performance_4 = {}
                try:
                    baseline_brand_path = static_load[
                        f"baseline_tc_{brand_code}_{sbu_selected}"
                    ]
                    df_tc_baseline_reload = pd.read_csv(
                        base_path + baseline_brand_path, sep=","
                    )
                    if setting_load["restaurant_selected"] == "ALL":
                        for tuple_i in tuple_sub_campaign:
                            start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                            end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                            sub_campaign_name_i = tuple_i[0]
                            campaign_name_i = tuple_i[3]
                            campaign_name_i_mode = tuple_i[3][1:]
                            df_tc_baseline_reload_i = df_tc_baseline_reload[
                                (
                                    df_tc_baseline_reload.ShiftDate.astype(str)
                                    >= start_time_i
                                )
                                & (
                                    df_tc_baseline_reload.ShiftDate.astype(str)
                                    <= end_time_i
                                )
                            ]
                            df_transaction_reload_i = df_transactions[
                                (df_transactions.ShiftDate.astype(str) >= start_time_i)
                                & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                                & (
                                    (
                                        df_transactions.VoucherCode.astype(str)
                                        == campaign_name_i
                                    )
                                    | (
                                        df_transactions.VoucherCode.astype(str)
                                        == campaign_name_i_mode
                                    )
                                )
                            ]
                            df_tc_reload_i = df_transactions[
                                (df_transactions.ShiftDate.astype(str) >= start_time_i)
                                & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                            ]

                            df_transaction_agg_i = (
                                df_transaction_reload_i.groupby("ShiftDate")[
                                    "GuestCount"
                                ]
                                .sum()
                                .reset_index()
                            )

                            df_transaction_tc_baseline = df_tc_baseline_reload
                            df_tc_i = df_tc_baseline_reload_i[["ShiftDate", "TC"]]
                            df_tc_i["TCType"] = "TC"
                            df_tc_baseline_i = df_tc_baseline_reload_i[
                                ["ShiftDate", "TC_baseline"]
                            ].rename(columns={"TC_baseline": "TC"})
                            df_tc_baseline_i["TCType"] = "TC Baseline"
                            df_tc_baseline_i["TC"] = df_tc_baseline_i["TC"].apply(
                                lambda x: round(x)
                            )
                            df_tc_mkt_i = df_transaction_agg_i[
                                ["ShiftDate", "GuestCount"]
                            ].rename(columns={"GuestCount": "TC"})
                            df_tc_mkt_i["TCType"] = "TC MKT"
                            df_tc_mkt_i["TC"] = df_tc_mkt_i["TC"].apply(
                                lambda x: round(x)
                            )

                            if campaign_name_i not in dict_performance_4.keys():
                                total_tc_baseline = round(np.sum(df_tc_baseline_i.TC))
                                total_tc_mkt = round(np.sum(df_tc_mkt_i.TC))
                                total_tc = round(np.sum(df_tc_reload_i.GuestCount))
                                dict_performance_4[f"{campaign_name_i}"] = {
                                    "Incremental TC": total_tc - total_tc_baseline,
                                    "Baseline TC": total_tc_baseline,
                                    "TC": total_tc,
                                    "%Incremental TC": (
                                        round(
                                            total_tc_mkt
                                            / (total_tc_mkt + total_tc_baseline),
                                            4,
                                        )
                                        if total_tc_mkt + total_tc_baseline > 0
                                        else 0
                                    ),
                                }
                except:
                    st.error(
                        f"Increamental Sale/TC Support for Brand Selection Only ! Your settings: {setting_load['brand_selected']} & {setting_load['restaurant_selected']}",
                        icon="🚨",
                    )
                    for tuple_i in tuple_sub_campaign:
                        campaign_name_i = tuple_i[3]
                        if campaign_name_i not in dict_performance_4.keys():
                            dict_performance_4[f"{campaign_name_i}"] = {
                                "Incremental TC": np.nan,
                                "Baseline TC": np.nan,
                                "TC": np.nan,
                                "%Incremental TC": np.nan,
                            }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_4 = pd.DataFrame(dict_performance_4).T
                st.dataframe(
                    df_performance_plot_4.sort_values(
                        by=["Incremental TC"]
                    ).style.format(
                        {
                            "%Incremental TC": "{:.2%}",
                            "TC": "{:,.0f}",
                            "Baseline TC": "{:,.0f}",
                            "Incremental TC": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_4)

            # 7.TA MKT
            if "7.TA MKT" in list_kpi_show:
                st.markdown(f"""**:red[7.TA MKT]**""")
                dict_performance_7 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Sale MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]
                    # df_transaction_mkt_agg_i = df_transaction_mkt_i.groupby("ShiftDate")["PriceSum"].sum().reset_index()
                    df_transaction_mkt_i = df_transaction_mkt_i[
                        df_transaction_mkt_i["GuestCount"] > 0
                    ]

                    if campaign_name_i not in dict_performance_7.keys():
                        total_sale_mkt = np.sum(df_transaction_mkt_i.PriceSum)
                        total_tc_mkt = np.sum(df_transaction_mkt_i.GuestCount)
                        dict_performance_7[f"{campaign_name_i}"] = {
                            "Sale MKT": round_money(total_sale_mkt, 6),
                            "TC MKT": round(total_tc_mkt),
                            "TA MKT": (
                                round_money(round(total_sale_mkt / total_tc_mkt), 3)
                                if total_tc_mkt != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_7 = pd.DataFrame(dict_performance_7).T
                st.dataframe(
                    df_performance_plot_7.sort_values(
                        by=["TA MKT"], ascending=False
                    ).style.format(
                        {
                            "Sale MKT": "{:,.0f}",
                            "TC MKT": "{:,.0f}",
                            "TA MKT": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_7)

            # 8.TA W/o MKT
            if "8.TA W/o MKT" in list_kpi_show:
                st.markdown(f"""**:red[8.TA W/o MKT]**""")
                dict_performance_8 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Sale MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]
                    # df_transaction_mkt_agg_i = df_transaction_mkt_i.groupby("ShiftDate")["PriceSum"].sum().reset_index()
                    df_transaction_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                    ]
                    df_transaction_mkt_i = df_transaction_mkt_i[
                        df_transaction_mkt_i["GuestCount"] > 0
                    ]

                    if campaign_name_i not in dict_performance_8.keys():
                        total_sale_mkt = np.sum(df_transaction_mkt_i.PriceSum)
                        total_tc_mkt = np.sum(df_transaction_mkt_i.GuestCount)
                        total_sale = np.sum(df_transaction_i.PriceSum)
                        total_tc = np.sum(df_transaction_i.GuestCount)
                        dict_performance_8[f"{campaign_name_i}"] = {
                            "Sale MKT": round_money(total_sale_mkt, 6),
                            "TC MKT": round(total_tc_mkt),
                            "Sale": round_money(total_sale, 6),
                            "TC": round(total_tc),
                            "TA without MKT": (
                                round_money(
                                    round(
                                        (total_sale - total_sale_mkt)
                                        / (total_tc - total_tc_mkt)
                                    ),
                                    3,
                                )
                                if total_tc != total_tc_mkt
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_8 = pd.DataFrame(dict_performance_8).T
                st.dataframe(
                    df_performance_plot_8.sort_values(
                        by=["TA without MKT"], ascending=False
                    ).style.format(
                        {
                            "Sale MKT": "{:,.0f}",
                            "TC MKT": "{:,.0f}",
                            "Sale": "{:,.0f}",
                            "TC": "{:,.0f}",
                            "TA without MKT": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_8)

            # 9.Customer Size MKT
            if "9.Customer Size MKT" in list_kpi_show:
                st.markdown(f"""**:red[9.Customer Size MKT]**""")
                dict_performance_9 = {}
                for tuple_i in tuple_sub_campaign:
                    print(tuple_i)
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # GZ MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]
                    df_transaction_mkt_i = df_transaction_mkt_i[
                        df_transaction_mkt_i.GuestCount > 0
                    ]

                    if campaign_name_i not in dict_performance_9.keys():
                        total_tc_mkt = np.sum(df_transaction_mkt_i.GuestCount)
                        total_bill_mkt = df_transaction_mkt_i.shape[0]
                        dict_performance_9[f"{campaign_name_i}"] = {
                            "TC MKT": round(total_tc_mkt),
                            "Bill MKT": round(total_bill_mkt),
                            "Customer Size MKT": (
                                round(total_tc_mkt / total_bill_mkt, 4)
                                if total_bill_mkt != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_9 = pd.DataFrame(dict_performance_9).T
                st.dataframe(
                    df_performance_plot_9.sort_values(
                        by=["Customer Size MKT"], ascending=False
                    ).style.format(
                        {
                            "TC MKT": "{:,.0f}",
                            "Bill MKT": "{:,.0f}",
                            "Customer Size MKT": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_9)

            # 10.Customer Size w/o MKT
            if "10.Customer Size w/o MKT" in list_kpi_show:
                st.markdown(f"""**:red[10.Customer Size w/o MKT]**""")
                dict_performance_10 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # GZ MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]
                    df_transaction_mkt_i = df_transaction_mkt_i[
                        df_transaction_mkt_i.GuestCount > 0
                    ]

                    # GZ FULL
                    df_transaction_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                    ]
                    df_transaction_i = df_transaction_i[df_transaction_i.GuestCount > 0]

                    if campaign_name_i not in dict_performance_10.keys():
                        total_tc_mkt = np.sum(df_transaction_mkt_i.GuestCount)
                        total_bill_mkt = df_transaction_mkt_i.shape[0]
                        total_tc = np.sum(df_transaction_i.GuestCount)
                        total_bill = df_transaction_i.shape[0]
                        dict_performance_10[f"{campaign_name_i}"] = {
                            "TC MKT": round(total_tc_mkt),
                            "Bill MKT": round(total_bill_mkt),
                            "TC": round(total_tc),
                            "Bill": round(total_bill),
                            "Customer Size without MKT": (
                                round(
                                    (total_tc - total_tc_mkt)
                                    / (total_bill - total_bill_mkt),
                                    4,
                                )
                                if total_bill != total_bill_mkt
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_10 = pd.DataFrame(dict_performance_10).T
                st.dataframe(
                    df_performance_plot_10.sort_values(
                        by=["Customer Size without MKT"], ascending=False
                    ).style.format(
                        {
                            "TC MKT": "{:,.0f}",
                            "Bill MKT": "{:,.0f}",
                            "Customer Size without MKT": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_10)
            # 11.% Deduction
            if "11.% Deduction" in list_kpi_show:
                st.markdown(f"""**:red[11.% Deduction]**""")
                dict_performance_11 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Deduction MKT
                    df_transaction_mkt_i = df_transactions[
                        (df_transactions.ShiftDate.astype(str) >= start_time_i)
                        & (df_transactions.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_transactions.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_transactions.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]

                    if campaign_name_i not in dict_performance_11.keys():
                        total_sale = np.sum(df_transaction_mkt_i.PriceSum)
                        total_discount = np.sum(df_transaction_mkt_i.Discount)
                        total_mkt_deduction = np.sum(df_transaction_mkt_i.VoucherAmount)
                        total_bill = df_transaction_i.shape[0]
                        dict_performance_11[f"{campaign_name_i}"] = {
                            "Deduction (Discount)": (
                                round(total_discount / total_sale, 4)
                                if total_sale != 0
                                else 0
                            ),
                            "Deduction (MKT)": (
                                round(total_mkt_deduction / total_sale, 4)
                                if total_sale != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                df_performance_plot_11 = pd.DataFrame(dict_performance_11).T
                st.dataframe(
                    df_performance_plot_11.sort_values(
                        by=["Deduction (MKT)"], ascending=False
                    ).style.format(
                        {"Deduction (Discount)": "{:.2%}", "Deduction (MKT)": "{:.2%}"}
                    ),
                    use_container_width=True,
                )
                list_df_performance.append(df_performance_plot_11)

            # 12.% MKT Fee
            ## Load campaign free data
            df_campaign_fee = pd.read_csv(
                base_path + static_load["campaign_fee_path"], sep="\t"
            )
            df_campaign_fee["VoucherCode"] = df_campaign_fee["CampaignCode"].apply(
                lambda x: get_voucher_code_format_1(x)
            )

            if "12.%MKT Fee" in list_kpi_show:
                st.markdown(f"""**:red[12.% MKT Fee]**""")
                dict_performance_12 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Sale MKT
                    # df_transaction_mkt_i = df_transactions[(df_transactions.ShiftDate.astype(str) >= start_time_i) & (df_transactions.ShiftDate.astype(str) <= end_time_i) & ((df_transactions.VoucherCode.astype(str) == campaign_name_i) | (df_transactions.VoucherCode.astype(str) == campaign_name_i_mode))]
                    # MKT Fee
                    df_campaign_fee_i = df_campaign_fee[
                        (df_campaign_fee.ShiftDate.astype(str) >= start_time_i)
                        & (df_campaign_fee.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_campaign_fee.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_campaign_fee.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]

                    if campaign_name_i not in dict_performance_12.keys():
                        total_mkt_sale = np.sum(df_campaign_fee_i.sales)
                        total_mkt_fee = np.sum(df_campaign_fee_i.mkt_fee)
                        dict_performance_12[f"{campaign_name_i}"] = {
                            "MKT Fee": round_money(total_mkt_fee, 3),
                            "%MKT Fee": (
                                round(total_mkt_fee / total_mkt_sale, 4)
                                if total_mkt_sale != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                dict_performance_12 = pd.DataFrame(dict_performance_12).T
                st.dataframe(
                    dict_performance_12.sort_values(
                        by=["%MKT Fee"], ascending=True
                    ).style.format({"MKT Fee": "{:,.0f}", "%MKT Fee": "{:.2%}"}),
                    use_container_width=True,
                )
                list_df_performance.append(dict_performance_12)

            if "13.ME/RE" in list_kpi_show:
                st.markdown(f"""**:red[13.ME/RE]**""")
                dict_performance_13 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # Sale MKT
                    # df_transaction_mkt_i = df_transactions[(df_transactions.ShiftDate.astype(str) >= start_time_i) & (df_transactions.ShiftDate.astype(str) <= end_time_i) & ((df_transactions.VoucherCode.astype(str) == campaign_name_i) | (df_transactions.VoucherCode.astype(str) == campaign_name_i_mode))]
                    # MKT Fee
                    df_campaign_fee_i = df_campaign_fee[
                        (df_campaign_fee.ShiftDate.astype(str) >= start_time_i)
                        & (df_campaign_fee.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_campaign_fee.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_campaign_fee.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]

                    if campaign_name_i not in dict_performance_13.keys():
                        total_mkt_sale = np.sum(df_campaign_fee_i.sales)
                        deduction = np.sum(df_campaign_fee_i.deduction)
                        total_mkt_fee = np.sum(df_campaign_fee_i.mkt_fee)
                        dict_performance_13[f"{campaign_name_i}"] = {
                            "ME": round_money(total_mkt_fee + deduction, 3),
                            "ME/RE": (
                                round((total_mkt_fee + deduction) / total_mkt_sale, 4)
                                if total_mkt_sale != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                dict_performance_13 = pd.DataFrame(dict_performance_13).T
                st.dataframe(
                    dict_performance_13.sort_values(
                        by=["ME/RE"], ascending=True
                    ).style.format({"ME": "{:,.0f}", "ME/RE": "{:.2%}"}),
                    use_container_width=True,
                )
                list_df_performance.append(dict_performance_13)

            if "14.MKT Cost/TC" in list_kpi_show:
                st.markdown(f"""**:red[14.MKT Cost/TC]**""")
                dict_performance_14 = {}
                for tuple_i in tuple_sub_campaign:
                    start_time_i = datetime.strftime(tuple_i[1], "%Y-%m-%d")
                    end_time_i = datetime.strftime(tuple_i[2], "%Y-%m-%d")
                    sub_campaign_name_i = tuple_i[0]
                    campaign_name_i = tuple_i[3]
                    campaign_name_i_mode = tuple_i[3][1:]

                    # TC MKT
                    # df_transaction_mkt_i = df_transactions[(df_transactions.ShiftDate.astype(str) >= start_time_i) & (df_transactions.ShiftDate.astype(str) <= end_time_i) & ((df_transactions.VoucherCode.astype(str) == campaign_name_i) | (df_transactions.VoucherCode.astype(str) == campaign_name_i_mode))]
                    # MKT Fee
                    df_campaign_fee_i = df_campaign_fee[
                        (df_campaign_fee.ShiftDate.astype(str) >= start_time_i)
                        & (df_campaign_fee.ShiftDate.astype(str) <= end_time_i)
                        & (
                            (df_campaign_fee.VoucherCode.astype(str) == campaign_name_i)
                            | (
                                df_campaign_fee.VoucherCode.astype(str)
                                == campaign_name_i_mode
                            )
                        )
                    ]

                    if campaign_name_i not in dict_performance_14.keys():
                        total_mkt_TC = np.sum(df_campaign_fee_i.tc)
                        deduction = np.sum(df_campaign_fee_i.deduction)
                        total_mkt_fee = np.sum(df_campaign_fee_i.mkt_fee)
                        dict_performance_14[f"{campaign_name_i}"] = {
                            "ME": round_money(total_mkt_fee + deduction, 3),
                            "MKT Cost/TC": (
                                round_money(
                                    (total_mkt_fee + deduction) / total_mkt_TC, 3
                                )
                                if total_mkt_TC != 0
                                else 0
                            ),
                        }
                st.markdown(
                    f"""
                ###### :blue[Summary]
                """
                )
                dict_performance_14 = pd.DataFrame(dict_performance_14).T
                st.dataframe(
                    dict_performance_14.sort_values(
                        by=["MKT Cost/TC"], ascending=True
                    ).style.format({"ME": "{:,.0f}", "MKT Cost/TC": "{:.0f}"}),
                    use_container_width=True,
                )
                list_df_performance.append(dict_performance_14)

            df_kpi_summary = list_df_performance[0]
            df_kpi_summary.index.name = "CampaignCode"
            df_kpi_summary.reset_index()
            for df_i in list_df_performance[1:]:
                df_i.index.name = "CampaignCode"
                df_i.reset_index()
                df_kpi_summary = df_kpi_summary.merge(
                    df_i, on="CampaignCode", how="left"
                )
                # df_kpi_summary = df_kpi_summary.join(df_i)
            kpi_summary_columns = [
                "CampaignCode",
                "CampaignName",
                "%Contribute Sale",
                "%Contribute Sale (Store)",
                "%Contribute TC",
                "%Contribute TC (Store)",
                "Incremental Sale",
                "%Incremental Sale",
                "Incremental TC",
                "%Incremental TC",
                "TA MKT",
                "TA without MKT",
                "Customer Size MKT",
                "Customer Size without MKT",
                "Deduction (MKT)",
                "%MKT Fee",
                "ME/RE",
                "MKT Cost/TC",
            ]

            df_kpi_summary = df_kpi_summary
            df_kpi_campaign_name = df_MKT_full_information_agg[
                ["Mã chương trình", "Tên chương trình"]
            ]
            df_kpi_campaign_name.columns = ["CampaignCode", "CampaignName"]
            df_kpi_summary = pd.merge(
                df_kpi_summary, df_kpi_campaign_name, how="left", on="CampaignCode"
            )
            df_kpi_summary = df_kpi_summary[kpi_summary_columns].reset_index()
            df_kpi_summary.to_excel(base_path + "dataset/SameEvent_MKT_KPI.xlsx")
    st.dataframe(
        df_kpi_summary.style.format(
            {
                "%Contribute Sale": "{:.2%}",
                "%Contribute Sale (Store)": "{:.2%}",
                "%Contribute TC": "{:.2%}",
                "%Contribute TC (Store)": "{:.2%}",
                "Incremental Sale": "{:,.0f}",
                "%Incremental Sale": "{:.2%}",
                "Incremental TC": "{:,.0f}",
                "%Incremental TC": "{:.2%}",
                "TA MKT": "{:,.0f}",
                "TA without MKT": "{:,.0f}",
                "Customer Size MKT": "{:,.1f}",
                "Customer Size without MKT": "{:,.1f}",
                "Deduction (MKT)": "{:.2%}",
                "%MKT Fee": "{:,.2f}",
                "ME/RE": "{:,.2f}",
                "MKT Cost/TC": "{:,.0f}",
            }
        )
    )
    # {:,.0f}
    st.download_button(
        label="download KPI (Excel file)",
        data=to_excel(df_kpi_summary),
        file_name="SameEvent_MKT_KPI.xlsx",
        key=key_multiselect + "download_KPI",
    )
    # return (dict_insight_long_term, dict_insight_weather, df_kpi_summary)
    # with tab_insight_1:

    # with tab_insight_2:
    #     # SETTING INTERFACE MODE: BOTH OR SINGLE
    #     if insight_show_type == insight_show_options[0]:
    #         col_insight_21, col_insight_22 = st.columns([1,1])
    #         columns_mode = 'Both'
    #     elif insight_show_type == insight_show_options[1]:
    #         col_insight_21 = st.container()
    #         col_insight_22 = None
    #         columns_mode = 'Single I'
    #     else:
    #         col_insight_22 = st.container()
    #         col_insight_21 = None
    #         columns_mode = 'Single II'
    #     # INSIGHT SHORT-TERM
    #     insight_details_term(term_range_day=short_term_range_day,
    #                          compare_type=compare_type,
    #                          start_term_event_date=start_short_term_event_date,
    #                          columns_insight=[col_insight_21, col_insight_22],
    #                          columns_mode=columns_mode, key_multiselect='km_2')

    st.markdown("""> ###### :blue[1.4 Summary]:""")
    with st.expander("👉 Bảng tổng hợp về hiệu quả Campaign", expanded=True):
        st.markdown("""##### 🔑 Campaign Performance:""")
        list_columns_kpi = [
            "%Contribute Sale",
            "%Contribute TC",
            "%Incremental Sale",
            "%Incremental TC",
            "TA MKT",
            "TA without MKT",
            "Customer Size MKT",
            "Customer Size without MKT",
            "Deduction (MKT)",
            "%MKT Fee",
            "ME/RE",
            "MKT Cost/TC",
        ]
        df_kpi_reload = pd.read_excel(base_path + "dataset/SameEvent_MKT_KPI.xlsx")

        best_rank = 1e10
        worse_rank = -1
        for col_i in list_columns_kpi:
            df_kpi_reload[f"{col_i}_rank"] = df_kpi_reload[col_i].rank(ascending=False)
            best_rank = min(best_rank, min(df_kpi_reload[f"{col_i}_rank"]))
            worse_rank = max(worse_rank, max(df_kpi_reload[f"{col_i}_rank"]))
        df_kpi_reload["Deduction (MKT)_rank"] = df_kpi_reload[
            "Deduction (MKT)_rank"
        ].rank(ascending=True)
        df_kpi_reload["%MKT Fee_rank"] = df_kpi_reload["%MKT Fee_rank"].rank(
            ascending=True
        )
        df_kpi_reload["ME/RE_rank"] = df_kpi_reload["ME/RE_rank"].rank(ascending=True)
        df_kpi_reload["MKT Cost/TC_rank"] = df_kpi_reload["MKT Cost/TC_rank"].rank(
            ascending=True
        )

        def cal_kpi_summary(x):
            result = {}
            total_best_kpi = 0
            total_worse_kpi = 0
            for kpi_rank_i in [f"{col_i}_rank" for col_i in list_columns_kpi]:
                if list(x[kpi_rank_i])[0] == best_rank:
                    total_best_kpi += 1
                if list(x[kpi_rank_i])[0] == worse_rank:
                    total_worse_kpi += 1
            result["Số lượng KPI (best)"] = total_best_kpi
            result["Số lượng KPI (worse)"] = total_worse_kpi
            return pd.Series(result)

        df_kpi_agg = (
            df_kpi_reload.groupby("CampaignCode").apply(cal_kpi_summary).reset_index()
        )
        df_kpi_agg = df_kpi_agg.rename(columns={"CampaignCode": "Mã chương trình"})
        df_kpi_detail = pd.read_excel(base_path + "dataset/SameEvent_MKT_KPI_INFO.xlsx")
        df_kpi_detail.drop_duplicates(subset=["Mã chương trình"], inplace=True)
        df_kpi_agg_detail = pd.merge(
            df_kpi_agg, df_kpi_detail, how="left", on="Mã chương trình"
        )
        df_kpi_agg_detail = df_kpi_agg_detail[
            [
                "Mã chương trình",
                "Tên chương trình",
                "Loại Chương trình",
                "Số lượng KPI (best)",
                "Số lượng KPI (worse)",
                "Ngày bắt đầu",
                "Ngày kết thúc",
            ]
        ]
        st.dataframe(
            df_kpi_agg_detail,
            column_config={
                "Số lượng KPI (best)": st.column_config.ProgressColumn(
                    "Số lượng KPI (best)",
                    help="SL KPI có hiệu quả tốt nhất",
                    format=f"%f/{len(list_columns_kpi)}",
                    min_value=0,
                    max_value=len(list_columns_kpi),
                ),
                "Số lượng KPI (worse)": st.column_config.ProgressColumn(
                    "Số lượng KPI (worse)",
                    help="SL KPI có hiệu quả không tốt nhất (Chú ý: rank tính theo các chương trình so sánh với nhau)",
                    format=f"%f/{len(list_columns_kpi)}",
                    min_value=0,
                    max_value=len(list_columns_kpi),
                ),
            },
        )


if st.session_state.get("role") not in role_this_page:
    st.info("You have no access this page !", icon="🔐")
    st.markdown(
        """
    <a href="Trade_Marketing_Tool" target="_self" style="font-size: 16px">  👉 Login Page</a>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
else:
    try:
        main()
    except Exception as e:
        import traceback

        st.error(f"{error_setting_message}: {str(traceback.format_exc())}", icon="🚨")
        is_error = True
