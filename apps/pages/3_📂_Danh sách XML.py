import streamlit as st
import sys
import os
import pandas as pd
import datetime
from datetime import datetime
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sharepoint import *

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Tra cứu & Theo dõi danh sách dữ liệu XML", page_icon="🔎", layout="wide"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Setting main-function 1
container_function_1 = st.container(height = 250, border = True)
# Setting main-function 2
container_function_2_shape = 1000
container_function_2 = st.container(height = container_function_2_shape, border = True)

# Show data
with container_function_1:
  st.markdown(
      """
  ##### **:red[1. View dữ liệu XML đã lưu trữ]**
  """
  )
  st.markdown("*Người dùng có thể tìm kiếm thông tin thông qua icon 🔍*")
  # Load XML currently
  XML_file = f"{parent_dir}\dataset_XML\data.xlsx"
  df_xml_data = pd.read_excel(XML_file, engine='openpyxl')
  st.dataframe(df_xml_data)
    
with container_function_2:
  st.markdown(
      """
  ##### **:red[2. Insight dữ liệu XML đã lưu trữ]**
  """
  )
  total_xml_file = len(set(df_xml_data['Tên file']))
  total_ma_so_thue = len(set(df_xml_data['Mã số thuế người bán']))
  total_so_lan_nhap_du_lieu = len(set(df_xml_data['Thời gian cập nhập']))
  max_lan_nhap_du_lieu = max(df_xml_data['Thời gian cập nhập'])
  container_metric_shape = 150
  container_metric = st.container(height = container_metric_shape, border = True)
  container_chart = st.container(height = container_function_2_shape - container_metric_shape, border = True)

  with container_metric:
    col_metric_1, col_metric_2, col_metric_3 = st.columns([1]*3)
    with col_metric_1:
      st.metric(label=":blue[**#️⃣ SỐ LƯỢNG FILE XML**]", value=total_xml_file)
    with col_metric_2:
      st.metric(label=":blue[**#️⃣ SỐ LƯỢNG MÃ SỐ THUẾ**]", value=total_ma_so_thue)
    with col_metric_3:
      st.metric(label=":blue[**#️⃣ SỐ LẦN NHẬP DỮ LIỆU**]", value=total_so_lan_nhap_du_lieu,delta=f"Latest {max_lan_nhap_du_lieu}", delta_color="off")
  with container_chart:
    col_chart_1, col_chart_2, col_chart_3 = st.columns([1, 1, 1])
    with col_chart_1:
      df_plot_distribution_DS_Tron_Thue = df_xml_data.groupby(["DS Trốn thuế"]).agg({"Số hóa đơn": 'nunique',  "Mã số thuế người bán": 'nunique'}).reset_index()
      df_plot_distribution_DS_Tron_Thue["Trốn thuế - Validation"] = df_plot_distribution_DS_Tron_Thue["DS Trốn thuế"].apply(lambda x: "Có - Trốn thuế" if x else "Không - Trốn thuế")
      st.markdown(f"###### :blue[1.1 Biểu đồ Phân bố MST Trốn thuế]")
      fig = px.pie(
          df_plot_distribution_DS_Tron_Thue,
          values="Số hóa đơn",
          title="",
          color="Trốn thuế - Validation",
          names="Trốn thuế - Validation",
          hole=0.3,
          color_discrete_sequence=[
              "#36C2CE",
              "#C1CDCD"
          ])
      fig.update_layout(hovermode="x unified", height=300)
      fig.update_traces(textposition='inside', textinfo='label+value')
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      with st.expander("Danh sách trốn thuế", expanded=False):
        st.dataframe(df_xml_data[df_xml_data['DS Trốn thuế']])
      with col_chart_2:
        df_plot_distribution_DS_MISS_SHDon = df_xml_data.groupby(["Thiếu SHDon"]).agg({"Số hóa đơn": 'nunique',  "Mã số thuế người bán": 'nunique'}).reset_index()
        df_plot_distribution_DS_MISS_SHDon["Thiếu SHDon - Validation"] = df_plot_distribution_DS_MISS_SHDon["Thiếu SHDon"].apply(lambda x: "Có Thiếu" if x else "Không Thiếu")
        st.markdown(f"###### :blue[1.2 Phân bố MST Thiếu SHDon]")
        fig = px.pie(
            df_plot_distribution_DS_MISS_SHDon,
            values="Số hóa đơn",
            title=f"",
            color="Thiếu SHDon - Validation",
            names="Thiếu SHDon - Validation",
            hole=0.3,
            color_discrete_sequence=[
                "#36C2CE",
                "#C1CDCD"
            ])
        fig.update_layout(hovermode="x unified", height=300)
        fig.update_traces(textposition='inside', textinfo='label+value')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with st.expander("Danh sách thiếu SHDon", expanded=False):
          st.dataframe(df_xml_data[df_xml_data['Thiếu SHDon']])

      with col_chart_3:
        df_plot_distribution_DS_MISS_MHSo = df_xml_data.groupby(["Thiếu MHSo"]).agg({"Số hóa đơn": 'nunique',  "Mã số thuế người bán": 'nunique'}).reset_index()
        df_plot_distribution_DS_MISS_MHSo["Thiếu MHSo - Validation"] = df_plot_distribution_DS_MISS_MHSo["Thiếu MHSo"].apply(lambda x: "Có Thiếu" if x else "Không Thiếu")
        st.markdown(f"###### :blue[1.3 Phân bố MST Thiếu MHSo] \n")
        fig = px.pie(
            df_plot_distribution_DS_MISS_MHSo,
            values="Số hóa đơn",
            title=f"",
            color="Thiếu MHSo - Validation",
            names="Thiếu MHSo - Validation",
            hole=0.3,
            color_discrete_sequence=[
                "#36C2CE",
                "#C1CDCD"
            ])
        fig.update_layout(hovermode="x unified", height=300)
        fig.update_traces(textposition='inside', textinfo='label+value')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with st.expander("Danh sách thiếu MHSo", expanded=False):
          st.dataframe(df_xml_data[df_xml_data['Thiếu MHSo']])

    st.markdown(f"###### :blue[2.1 Biểu đồ Timeline nhập dữ liệu]")
    df_plot_distribution_timeline =  df_xml_data.groupby(["Thời gian cập nhập"]).agg({"Tên file": 'nunique',  "Mã số thuế người bán": 'nunique'}).reset_index()
    df_plot_distribution_timeline_1 = df_plot_distribution_timeline[["Thời gian cập nhập", "Tên file"]].rename(columns = {"Tên file": "Số lượng dữ liệu"})
    df_plot_distribution_timeline_1["Type"] = "Số lượng Files"
    df_plot_distribution_timeline_2 = df_plot_distribution_timeline[["Thời gian cập nhập", "Mã số thuế người bán"]].rename(columns = {"Mã số thuế người bán": "Số lượng dữ liệu"})
    df_plot_distribution_timeline_2["Type"] = "Số lượng MST"
    df_plot_distribution_timeline_plot = pd.concat([df_plot_distribution_timeline_1, df_plot_distribution_timeline_2], axis=0)
    fig = px.bar(df_plot_distribution_timeline_plot, x="Thời gian cập nhập", y="Số lượng dữ liệu", color="Type", title="",
                 barmode="group", color_discrete_sequence=["#36C2CE", "#ECFFE6"])
    fig.update_layout(hovermode="x unified", height=400)
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


