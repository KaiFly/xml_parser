import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
# from office365.sharepoint.files.file import File
# from office365.sharepoint.client_context import ClientContext
# from office365.runtime.auth.authentication_context import AuthenticationContext
# from streamlit_gsheets import GSheetsConnection
import json
import datetime
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="Cập nhập danh sách TIN", page_icon="🔎", layout="wide"
)
# Setting main-function 1
container_function_1 = st.container(height = 500, border = True)
# Setting main-function 2 
container_function_2 = st.container(height = 500, border = True)
with container_function_1:
  # First section: e-mail and password as input
  st.markdown(
      """
  ##### **:red[1. Phương thức lưu trữ dữ liệu]**
  """
  )
  col_google_sheet, col_GCP, col_sharepoint, col_local = st.tabs(['Google Sheet', 'Google Cloud Storage', 'NCB Sharepoint - NOK', 'Local Folder - NOK'])
  with col_google_sheet:
    st.markdown(
    """ ###### **:blue[Google Sheet]**"""
    )
    from ggsheet_connector import *
    #load GGSheet link for user
    try:
      GGSheet_folder = st.secrets.connection.gsheets['spreadsheet']
    except:
      GGSheet_folder = "https://docs.google.com/spreadsheets/d/1qShV33l-EQ8GBa0X84NDJ8qe9H7mxSZsblKyzsEeBYg/edit#gid=0"

    st.info(f"Thư mục lưu trữ tại Privated Google Sheet: {GGSheet_folder}")
    col_selection_2, col_selection_1, _ = st.columns([3, 2, 5])
    with col_selection_1:
      file_format = st.selectbox(
          "Định dạng:",
          (".xlsx", ".csv", ".xls"),
          0)
    with col_selection_2:
      type_param = 'xlsx' if file_format in ['.xlsx', '.xls'] else 'csv' 
      uploaded_blacklist_file = st.file_uploader(label = f"⬆️Tải dữ liệu MST định dạng {file_format} ⬆️",
                                                 accept_multiple_files=False, type = [type_param], key = 'tmp_3')
    # st.info("Chú ý: Dữ liệu mặc định xlsx, Gồm 1 cột thông tin duy nhất TIN")

    if uploaded_blacklist_file is not None:
        # load & save data
        is_save_success = False
        is_load_success = False
        try:
            if file_format in ['.xlsx', '.xls']:
              df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
              st.info(f"✅ Tải lên thành công từ .xlsx: {df_dn_tron_thue.shape[0]} bản ghi")
              is_load_success = True
            elif file_format in ['.csv']:
              with st.sidebar:
                with st.spinner(f"Đang đọc danh sách TIN tải lên từ người dùng"):
                  df_dn_tron_thue = pd.read_csv(uploaded_blacklist_file, sep=",")[['TIN']]
                  st.info(f"✅ Tải lên thành công từ .csv: {df_dn_tron_thue.shape[0]} bản ghi")
                  is_load_success = True
            else:
                st.error("Định dạng lựa chọn không phù hợp/chưa hỗ trợ")
        except:
          df_dn_tron_thue  = pd.DataFrame(columns=['TIN'])
          st.error("Kiểm tra lại file, chú ý file excel chỉ gồm 1 thông tin là TIN!")
        # if load successfully, save data & metadata to GCS
        if is_load_success:
          try:
            role = st.session_state["role"]
          except:
            role = "ncb_ketoannoibo"
          meta_data = {
            "update_time": datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
            "user": role,
            "data_size": df_dn_tron_thue.shape[0]
          }              
          df_meta_data = pd.DataFrame(columns=meta_data.keys(), data=[list(meta_data.values())])              
          try:
            with st.sidebar:
              with st.spinner(f"Đang update danh sách {df_dn_tron_thue.shape[0]} TIN lên GG Sheet"):
                write_TIN_data(df_dn_tron_thue)
                write_TIN_metadata(df_meta_data)
                is_save_success = True
          except:
            st.error("Kiểm tra lại kết nối, upload danh sách TIN không thành công!")

        if is_save_success:
          st.info(f"✅ Upload thành công lên cơ sở dữ liệu chung ! ")
  with col_sharepoint:
    st.markdown(
        """
    ###### **:blue[NCB - Sharepoint Folder]**
    """
    )
    # try:
    #   placeholder = st.empty()
    #   with placeholder.container():
    #     col1, col2, col3 = st.columns([1, 4, 1])
    #     with col2:
    #       email_user = st.text_input("NCB E-mail")
    #       password_user = st.text_input("password", type="password")

    #       # Save the button status
    #       Button = st.button("📶 Kết nối")
    #       if st.session_state.get('button') != True:
    #         st.session_state['button'] = Button

    #   # Authentication and connection to SharePoint
    #   def authentication(email_user, password_user, sharepoint_url) :
    #     auth = AuthenticationContext(sharepoint_url) 
    #     auth.acquire_token_for_user(email_user, password_user)
    #     ctx = ClientContext(sharepoint_url, auth)
    #     web = ctx.web
    #     ctx.load(web)
    #     ctx.execute_query()
    #     return ctx

    #   # Second section: display results
    #   if st.session_state['button']:                              
    #     placeholder.empty()
    #     if "ctx" not in st.session_state :
    #         sharepoint_url = sharepoint_url_path + sharepoint_folder_path
    #         st.session_state["ctx"] = authentication(email_user, password_user, sharepoint_url)
    #     st.success("Xác thực: Thành công")
    #     st.sucesss("Đã kết nối tới Sharepoint: **{}**".format( st.session_state["ctx"].web.properties['Title']))
    # except Exception as e:
    #   st.error(f"Đăng nhập lỗi - Log: {e}")
  with col_local:
    st.markdown(
        """
    ###### **:blue[NCB - Local Folder]**
    """
    )
    # TIN_folder = f"{parent_dir}\dataset_TIN"
    # st.info(f"Thư mục lưu trữ tại máy cá nhân: {TIN_folder}")
    # with st.popover("Cập nhập định dạng TIN (*mặc định .xlsx*)"):
    #   file_format = st.selectbox(
    #       "Định dạng:",
    #       (".xlsx", ".csv", ".xls"),
    #       0,
    #       key = 'tmp_2'
    #   )
    # with st.popover("Cập nhập danh sách TIN:"):
    #     st.write("**Chú ý: gồm 1 cột thông tin duy nhất: TIN**")
    #     uploaded_blacklist_file = st.file_uploader(label = f"⬆️Tải dữ liệu MST định dạng {file_format} ⬆️", accept_multiple_files=False, type = ['xlsx'])
    #     if uploaded_blacklist_file is not None:
    #         # load & save data
    #         is_save_success = False
    #         try:
    #             if file_format in ['.xlsx', '.xls']:
    #               df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
    #               st.info(f"Tải lên thành công: {df_dn_tron_thue.shape[0]} bản ghi")
    #               df_dn_tron_thue.to_csv(TIN_folder + "/data.csv", sep="\t")
    #               st.success(f"Lưu trữ thành công: {df_dn_tron_thue.shape[0]} bản ghi")
    #               is_save_success = True
    #             elif file_format in ['.csv']:
    #               df_dn_tron_thue = pd.read_csv(uploaded_blacklist_file, sep=",")[['TIN']]
    #               st.info(f"Tải lên thành công: {df_dn_tron_thue.shape[0]} bản ghi")
    #               df_dn_tron_thue.to_csv(TIN_folder  + "/data.csv", sep="\t")
    #               st.success(f"Lưu trữ thành công: {df_dn_tron_thue.shape[0]} bản ghi")
    #               is_save_success = True
    #             else:
    #                st.error("Định dạng lựa chọn không phù hợp/chưa hỗ trợ")
    #         except:
    #           df_dn_tron_thue  = pd.DataFrame(columns=['TIN'])
    #           st.error("Kiểm tra lại file, chú ý file excel chỉ gồm 1 thông tin là TIN!")
    #         # save meta data
    #         if is_save_success:

    #           meta_data = {
    #             "update_time": datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
    #             "user": st.session_state["role"],
    #             "data_size": df_dn_tron_thue.shape[0]
    #           }
    #           with open(TIN_folder + "/metadata.json", "w") as outfile:
    #             json.dump(meta_data, outfile)
    #           meta_data_history = json.load(open(TIN_folder + "/metadata_his.json"))
    #           meta_data_history_keys = [int(i) for i in list(meta_data_history.keys())]
    #           meta_data_history_key = max(meta_data_history_keys)
    #           meta_data_history[str(meta_data_history_key + 1)] = meta_data
    #           with open(TIN_folder + "/metadata_his.json", "w") as outfile:
    #             json.dump(meta_data_history, outfile)
  with col_GCP:
    st.markdown(
        """
    ###### **:blue[Google Cloud Storage]**
    """
    )

    # #load GSC folder link for user
    # try:
    #   GCP_folder = st.secrets['connection.gcs.filesystem']['gcs_folder']
    # except:
    #   GCP_folder = "https://console.cloud.google.com/storage/browser/ncb_xml_parsing_tool?project=my-project-ncb-xml"

    # st.info(f"Thư mục lưu trữ tại Cloud: {GCP_folder}")
    # with st.popover("Cập nhập định dạng TIN (*mặc định .xlsx*)"):
    #   file_format = st.selectbox(
    #       "Định dạng:",
    #       (".xlsx", ".csv", ".xls"),
    #       0
    #   )
    # with st.popover("Cập nhập danh sách TIN:"):
    #     st.write("**Chú ý: gồm 1 cột thông tin duy nhất: TIN**")
    #     uploaded_blacklist_file = st.file_uploader(label = f"⬆️Tải dữ liệu MST định dạng {file_format} ⬆️",  accept_multiple_files=False, type = ['xlsx'], key = 'tmp_3')
    #     if uploaded_blacklist_file is not None:
    #         # load & save data
    #         is_save_success = False
    #         is_load_success = False
    #         try:
    #             if file_format in ['.xlsx', '.xls']:
    #               df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
    #               st.info(f"Tải lên thành công từ .xlsx: {df_dn_tron_thue.shape[0]} bản ghi")
    #               is_load_success = True
    #             elif file_format in ['.csv']:
    #               df_dn_tron_thue = pd.read_csv(uploaded_blacklist_file, sep=",")[['TIN']]
    #               st.info(f"Tải lên thành công từ .csv: {df_dn_tron_thue.shape[0]} bản ghi")
    #               is_load_success = True
    #             else:
    #                st.error("Định dạng lựa chọn không phù hợp/chưa hỗ trợ")
    #         except:
    #           df_dn_tron_thue  = pd.DataFrame(columns=['TIN'])
    #           st.error("Kiểm tra lại file, chú ý file excel chỉ gồm 1 thông tin là TIN!")
    #         # if load successfully, save data & metadata to GCS
    #         if is_load_success:
    #           meta_data = {
    #             "update_time": datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
    #             "user": st.session_state["role"],
    #             "data_size": df_dn_tron_thue.shape[0]
    #           }

with container_function_2:
  # First section: e-mail and password as input
  st.markdown(
      """
  ##### **:red[2. Dữ liệu TIN lưu trữ]**
  """)
  #load_type = 'local'
  #load_type = 'gcs'
  load_type = 'ggs'
  if load_type =='local':
    meta_data = json.load(open(TIN_folder + "/metadata.json"))
    st.success("> 📃 Thông tin về hiện trạng dữ liệu TIN (danh sách MST trốn thuế):")
    st.info(f"+ Location: {TIN_folder} \n + Số lượng TIN: {meta_data['data_size']} \n + Lần cập nhập gần nhất: {meta_data['update_time']} \n + Lịch sử cập nhập:")
    meta_data_his = json.load(open(TIN_folder + "/metadata_his.json"))
    df_history = pd.read_json(TIN_folder + "/metadata_his.json")
    _, col_history, _ = st.columns([1, 2, 1])
    with col_history:
      st.dataframe(df_history.T, width=500, height=350)
  elif load_type == 'gcs':
    st.success("> 📃 Thông tin về hiện trạng dữ liệu TIN (danh sách MST trốn thuế) - **GCP chưa có phép Write, hiện tại đang Read-Only**:")
    from st_files_connection import FilesConnection
    conn = st.connection('gcs', type=FilesConnection)
    df_dn_tron_thue = read_TIN_data(conn)
    st.info(f"+ Location: {GCP_folder} \n + Số lượng TIN: {df_dn_tron_thue.shape[0]} \n + Lần cập nhập gần nhất: '2024-07-16 15:15:11' \n + Lịch sử cập nhập:")
    # meta_data_his = json.load(open(TIN_folder + "/metadata_his.json"))
    # df_history = pd.read_json(TIN_folder + "/metadata_his.json")
    # _, col_history, _ = st.columns([1, 2, 1])
    # with col_history:
    #   st.dataframe(df_history.T, width=500, height=350)
  else:
    #try:
    df_TIN_metadata = read_TIN_metadata()
    df_TIN_latest = df_TIN_metadata.sort_values(by=['update_time'], ascending=False).iloc[0]
    total_TIN_latest = df_TIN_latest['data_size']
    update_time_TIN_latest = df_TIN_latest['update_time']
    st.info(f"+ Location: {GGSheet_folder} \n + Số lượng TIN: {total_TIN_latest} \n + Lần cập nhập gần nhất: {update_time_TIN_latest} \n + Lịch sử cập nhập:")
    st.dataframe(df_TIN_metadata, hide_index=True)
    # except:
    #   st.error("Error loading metadata!")
