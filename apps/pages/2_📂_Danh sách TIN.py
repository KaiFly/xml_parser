import streamlit as st
import sys
import os
import math
import numpy as np
import pandas as pd
import random
from office365.sharepoint.files.file import File
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.authentication_context import AuthenticationContext
import yaml
import json
import datetime
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sharepoint import *

st.set_page_config(
    page_title="Cập nhập danh sách TIN", page_icon="🔎", layout="wide"
)

# Setting main-function 1
container_function_1 = st.container(height = 360, border = True)
# Setting main-function 2 
container_function_2 = st.container(height = 640, border = True)
with container_function_1:
  # First section: e-mail and password as input
  st.markdown(
      """
  ##### **:red[1. Phương thức lưu trữ dữ liệu]**
  """
  )
  col_sharepoint, col_local = st.columns([1, 1])
  with col_sharepoint:
    st.markdown(
        """
    ###### **:blue[NCB - Sharepoint Folder]**
    """
    )
    try:
      placeholder = st.empty()
      with placeholder.container():
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
          email_user = st.text_input("NCB E-mail")
          password_user = st.text_input("password", type="password")

          # Save the button status
          Button = st.button("📶 Kết nối")
          if st.session_state.get('button') != True:
            st.session_state['button'] = Button

      # Authentication and connection to SharePoint
      def authentication(email_user, password_user, sharepoint_url) :
        auth = AuthenticationContext(sharepoint_url) 
        auth.acquire_token_for_user(email_user, password_user)
        ctx = ClientContext(sharepoint_url, auth)
        web = ctx.web
        ctx.load(web)
        ctx.execute_query()
        return ctx

      # Second section: display results
      if st.session_state['button'] :                              
        placeholder.empty()
        if "ctx" not in st.session_state :
            sharepoint_url = sharepoint_url_path + sharepoint_folder_path
            st.session_state["ctx"] = authentication(email_user, password_user, sharepoint_url)
        st.success("Xác thực: Thành công")
        st.sucesss("Đã kết nối tới Sharepoint: **{}**".format( st.session_state["ctx"].web.properties['Title']))
    except Exception as e:
      st.error(f"Đăng nhập lỗi - Log: {e}")
  with col_local:
    st.markdown(
        """
    ###### **:blue[NCB - Local Folder]**
    """
    )
    TIN_folder = f"{parent_dir}\dataset_TIN"
    st.info(f"Thư mục lưu trữ tại máy cá nhân: {TIN_folder}")
    with st.popover("Cập nhập định dạng TIN (*mặc định .xlsx*)"):
      file_format = st.selectbox(
          "Định dạng:",
          (".xlsx", ".csv", ".xls"),
          0
      )
    with st.popover("Cập nhập danh sách TIN:"):
        st.write("**Chú ý: gồm 1 cột thông tin duy nhất: TIN**")
        uploaded_blacklist_file = st.file_uploader(label = f"⬆️Tải dữ liệu MST định dạng {file_format} ⬆️", accept_multiple_files=False, type = ['xlsx'])
        if uploaded_blacklist_file is not None:
            # load & save data
            is_save_success = False
            try:
                if file_format in ['.xlsx', '.xls']:
                  df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
                  st.info(f"Tải lên thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                  df_dn_tron_thue.to_csv(TIN_folder + "/data.csv", sep="\t")
                  st.success(f"Lưu trữ thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                  is_save_success = True
                elif file_format in ['.csv']:
                  df_dn_tron_thue = pd.read_csv(uploaded_blacklist_file, sep=",")[['TIN']]
                  st.info(f"Tải lên thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                  df_dn_tron_thue.to_csv(TIN_folder  + "/data.csv", sep="\t")
                  st.success(f"Lưu trữ thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                  is_save_success = True
                else:
                   st.error("Định dạng lựa chọn không phù hợp/chưa hỗ trợ")
            except:
              df_dn_tron_thue  = pd.DataFrame(columns=['TIN'])
              st.error("Kiểm tra lại file, chú ý file excel chỉ gồm 1 thông tin là TIN!")
            # save meta data
            if is_save_success:

              meta_data = {
                "update_time": datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
                "user": st.session_state["role"],
                "data_size": df_dn_tron_thue.shape[0]
              }
              with open(TIN_folder + "/metadata.json", "w") as outfile:
                json.dump(meta_data, outfile)
              meta_data_history = json.load(open(TIN_folder + "/metadata_his.json"))
              meta_data_history_keys = [int(i) for i in list(meta_data_history.keys())]
              meta_data_history_key = max(meta_data_history_keys)
              meta_data_history[str(meta_data_history_key + 1)] = meta_data
              with open(TIN_folder + "/metadata_his.json", "w") as outfile:
                json.dump(meta_data_history, outfile)


      # df_dn_tron_thue["TIN"] = df_dn_tron_thue["TIN"].astype(str)
  # with col_db:
  #   st.markdown(
  #       """
  #   ###### **:blue[NCB - Database (SQL-Lite)]**
  #   """
  #   )
with container_function_2:
  # First section: e-mail and password as input
  st.markdown(
      """
  ##### **:red[2. Dữ liệu TIN lưu trữ]**
  """)
  meta_data = json.load(open(TIN_folder + "/metadata.json"))
  st.success("> 📃 Thông tin về hiện trạng dữ liệu TIN (danh sách MST trốn thuế):")
  st.info(f"+ Location: {TIN_folder} \n + Số lượng TIN: {meta_data['data_size']} \n + Lần cập nhập gần nhất: {meta_data['update_time']} \n + Lịch sử cập nhập:")
  meta_data_his = json.load(open(TIN_folder + "/metadata_his.json"))
  df_history = pd.read_json(TIN_folder + "/metadata_his.json")
  _, col_history, _ = st.columns([1, 2, 1])
  with col_history:
    st.dataframe(df_history.T, width=500, height=350)