import streamlit as st
import sys
import os
import math
import numpy as np
import pandas as pd
import random
# from office365.sharepoint.files.file import File
# from office365.sharepoint.client_context import ClientContext
# from office365.runtime.auth.authentication_context import AuthenticationContext
# from streamlit_gsheets import GSheetsConnection
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
  col_sharepoint, col_local, col_GCP = st.columns([1, 1, 1])
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
          0,
          key = 'tmp_2'
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
  with col_GCP:
    st.markdown(
        """
    ###### **:blue[Google Cloud Storage]**
    """
    )

    GCP_folder = f"https://console.cloud.google.com/storage/browser/ncb_xml_parsing_tool?project=my-project-ncb-xml"
    st.info(f"Thư mục lưu trữ tại Cloud: {GCP_folder}")
    with st.popover("Cập nhập định dạng TIN (*mặc định .xlsx*)"):
      file_format = st.selectbox(
          "Định dạng:",
          (".xlsx", ".csv", ".xls"),
          0
      )
    with st.popover("Cập nhập danh sách TIN:"):
        st.write("**Chú ý: gồm 1 cột thông tin duy nhất: TIN**")
        uploaded_blacklist_file = st.file_uploader(label = f"⬆️Tải dữ liệu MST định dạng {file_format} ⬆️",
                                                    accept_multiple_files=False,
                                                      type = ['xlsx'], key = 'tmp_3')
        if uploaded_blacklist_file is not None:
            # load & save data
            is_save_success = False
            try:
                if file_format in ['.xlsx', '.xls']:
                  df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
                  st.info(f"Tải lên thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                  st.success(f"Lưu trữ thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                  is_save_success = True
                elif file_format in ['.csv']:
                  df_dn_tron_thue = pd.read_csv(uploaded_blacklist_file, sep=",")[['TIN']]
                  st.info(f"Tải lên thành công: {df_dn_tron_thue.shape[0]} bản ghi")
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
  #load_type = 'local'
  #load_type = 'gcs'
  load_type = 'simulated'
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
    df_dn_tron_thue  = pd.DataFrame(["0107816658","0107588627","0108949555","3700696130","3603316376","3603311265","6400351204","3603307526","0109695342","0109480883","0107551264","0107627386","0107427605","0314176011","0310505480","0303021961","0309494033","0312325571","0312634259","0305355712","0307761212","0312970589","0312441338","0308299946","0312233754","4200665880","5900413721","5900750614","5900288005","5900229419","5900404406","0401356839","4101396806","4100456435","4101189278","5900701254","4200430254","4400845055","4400828571","4400849356","4400824538","4400366704","4400886622","0107755116","0107767136","0107767143","0201988485","4101247272","4201529496","4000461488","4200490221","4200522515","4000492528","4200603267","4000501204","4201244204","4200445405","4000624051","4000599969","2601021777","3702896706","4400384767","2901872472","2901868927","4400666786","0400524840-010","0401347947","0400581214","5900189903-349","3800493310","6000706999","3800307437","4100966330","4200966077","4100963410","4201298961","4100640226","4201381419","4201304622","0801098188","0201118763","0201122858","2600451625-001","0201118837","0201118724","2600439184","0201124037","0201132334","0201122978","0201128779","0101048939","0101051917","0101040714","0100979526","0100963533","0101014182","0101013661","0101022200"], columns=["TIN"])
    st.info(f"+ Location: {GCP_folder} \n + Số lượng TIN: {df_dn_tron_thue.shape[0]} \n + Lần cập nhập gần nhất: '2024-07-16 15:15:11' \n + Lịch sử cập nhập:")
