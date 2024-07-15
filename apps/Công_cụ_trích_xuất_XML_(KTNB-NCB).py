import streamlit as st

from io import StringIO
import sys, os
import pandas as pd
import numpy as np
import datetime
from datetime import datetime

from utils import data_config, data_tron_thue_csv_path
from utils import get_nested, to_excel
from utils import get_leaves_and_parse
from utils import currency_format
from sharepoint import save_data_to_sharepoint

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import xmltodict
from PIL import Image

## Load setting static data: data path, data source
st.set_page_config(page_title="NCB Kế toán - Trích xuất dữ liệu hóa đơn", page_icon="👋", layout="wide")
error_setting_message = (
    "Đã có lỗi xảy ra!"
)
side_bar_progress = st.sidebar.progress(0, text="📌 Start Progress")

# Add custom CSS to hide the GitHub icon
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Define the current path 
current_dir = os.path.dirname(os.path.abspath(__file__))
TIN_file = f"{current_dir}\dataset_TIN\data.csv"
XML_file = f"{current_dir}\dataset_XML\data.xlsx"

# Define stateful functions
st.session_state.step_1 = False
st.session_state.step_2 = False
st.session_state.step_3 = False
st.session_state.step_4 = False
def step1_upload_file():
    st.session_state.step_1 = True
def step2_click_button_configuration():
    st.session_state.step_2 = True
def step3_click_button_data_save():
    st.session_state.step_3 = True
def step4_click_button_done(df_information_save):
    # finish the state after this action
    st.session_state.step_4 = True
    # Khởi tạo dataframe lưu trữ danh sách XML

    #type_saving = 'sharepoint'
    type_saving = 'local'
    with st.spinner("Đợi DL XML lưu xuống cơ sở dữ liệu:"):
        # import time
        # time.sleep(10)
        if type_saving == 'sharepoint':
            save_data_to_sharepoint()
        else:
            # Reload saved data, and dont remove duplicates
            try:
                df_information_saved = pd.read_excel(XML_file)
            except:
                df_information_saved = pd.DataFrame(columns = list(df_information_save.columns) + ['Thời gian cập nhập', 'User cập nhập'])
            df_information_append = df_information_save[~df_information_save['Tên file'].isin(df_information_saved['Tên file'])]
            df_information_append['Thời gian cập nhập'] = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            df_information_append['User cập nhập'] =  st.session_state["role"]
            total_sample_saved = df_information_append.shape[0]
            total_sample_saving = df_information_save.shape[0]
            df_information_save_final = pd.concat([df_information_saved, df_information_append], axis=0)
            try:
                df_information_save_final.to_excel(XML_file, index=False)
            except:
                st.error("Có lỗi xảy ra khi lưu dữ liệu. Kiểm tra lại có phải file đang được truy cập hay không?")
        step2_click_button_configuration()
        step3_click_button_data_save()
    st.sidebar.write("""**✅ STEP 3:** *Confirm Data Saving*""")
    side_bar_progress.progress(100, text="✅ Hoàn thành!")
    st.success(f"✅ Đã lưu thêm thành công là {total_sample_saved} dữ liệu XML (trên tổng số {total_sample_saving} XML đã tải lên).")
    st.info(f'🔚 Hoàn thành phiên làm việc này 🔚. \n Hãy đăng nhập lại hoặc reload *(Ctrl+R)* lại trang làm việc để tiếp tục trích xuất dữ liệu.')
    st.stop()
    
def main():
    # Load setting path based on account authentication

    # Setting main-function 1
    container_function_1 = st.container(height = 300, border = True)
    # Setting main-function 2 
    container_function_2 = st.container(height = 600, border = True)
    # Setting main-function 3
    container_function_3 = st.container(height = 100, border = True)

    #st.sidebar.progress(0, text="📌 Start Progress")
    with container_function_1:
        col_11, col_12, col_13 = st.columns([1, 0.2, 1])
        with col_11:
            st.markdown(
                """
            ##### **:red[1. Nhập dữ liệu - Hóa đơn điện tử]**
            """
            )
            side_bar_progress.progress(25, text="**➡️ STEP 1:** *Upload Files*")
            uploaded_files = st.file_uploader(label = "⬆️Tải nhiều file XML lên⬆️", accept_multiple_files=True, type = ['xml'])
            if len(uploaded_files) > 0:
                dict_string_io_data = {}
                list_string_io_data = []
                for uploaded_file_i in uploaded_files:
                    stringio_i = StringIO(uploaded_file_i.getvalue().decode("utf-8"))
                    string_data_i = stringio_i.read()
                    list_string_io_data.append(string_data_i)
                    #st.write(string_data_i)
                    dict_string_io_data[uploaded_file_i.name] = string_data_i
                step1_upload_file()
            if st.session_state.step_1 and not st.session_state.step_2:
                st.sidebar.write("""**✅ STEP 1:** *Upload Files*""")
                side_bar_progress.progress(50, text="**➡️ STEP 2:** *Confirm Configuration*")
                st.toast(f'Đã tải lên thành công {len(list_string_io_data)} files!', icon='🎉')
            else:
                with st.sidebar:
                    with st.spinner("Đang đợi nhập dữ liệu..."):
                        while True:
                            if st.session_state.step_1 == True:
                                break
        with col_13:
            st.markdown(
                """
            ##### **:red[2. Tùy chỉnh cấu hình]**
            """
            )
            col_configuration, col_confirm_button = st.columns([8, 2])
            with col_configuration:
                with st.expander("2.1 Danh sách các trường dữ liệu (mặc định)", expanded=False):
                    df_default_field = pd.DataFrame.from_dict(data_config).T
                    st.dataframe(df_default_field, hide_index=True)

                with st.expander("2.2 Danh sách các trường dữ liệu bổ sung", expanded=False):
                    # get first file as dictionary to get the tree structure
                    if len(list_string_io_data) == 0:
                        list_parsed = []
                    else:
                        sample_format = xmltodict.parse(list_string_io_data[0])
                        leaves, dct_parsed = get_leaves_and_parse(sample_format)
                        list_parsed = list(dct_parsed.keys())
                        list_parsed = [i for i in list_parsed if i.split(".")[-1] not in list(df_default_field["Thẻ"])]
                    list_append_field = st.multiselect(
                        label="Lựa chọn thêm các DL cần bổ sung: <br>  *Định dạng tìm kiếm: [Thẻ]/[Chỉ tiêu]* \n *Ví dụ: 'HDon/DLHDon/TTChung/KHMSHDon'*",
                        placeholder ="Chọn các trường dữ liệu cần bổ sung",
                        options=list_parsed,
                        default=[],
                    )
                    if len(list_append_field) > 0:
                        data_append = {}
                        current_index = df_default_field.shape[0]

                        for i in range(len(list_append_field)):
                            #Example format: "10": {"Thẻ": "HDon/DLHDon/NDHDon/TToan", "Chỉ tiêu": "TgTTTBSo", "Mô tả": "Tổng tiền thanh toán bằng số"}
                            data_append[str(current_index + 1)] = {
                                "Thẻ": "/".join(list_append_field[i].split(".")[:-1]),
                                "Chỉ tiêu": list_append_field[i].split(".")[-1],
                                "Mô tả": list_append_field[i].split(".")[-1]
                            }
                            current_index += 1
                        df_append_field = pd.DataFrame.from_dict(data_append).T
                        # Load append data to default data if there are default data
                        df_default_field = pd.concat([df_default_field, df_append_field], axis=0)
                        st.toast(f"Đã lấy thêm {df_append_field.shape[0]} trường dữ liệu!", icon='🎉')
                        st.dataframe(df_append_field)
                with st.expander(f"2.3 Danh sách TIN - *Sử dụng để gắn cờ trong MST trốn thuế*", expanded=True):
                    df_dn_tron_thue = pd.read_csv(TIN_file, sep=",", encoding='cp1252')
                    st.info(f"Hiện tại danh sách bao gồm: {df_dn_tron_thue.shape[0]} TIN")
                    # with st.popover("Cập nhập DS MST Blacklist:"):
                    #     st.write("**Chú ý: gồm 1 cột thông tin duy nhất: TIN**")
                    #     uploaded_blacklist_file = st.file_uploader(label = "⬆️Tải dữ liệu MST định dạng xlsx ⬆️", accept_multiple_files=False, type = ['xlsx'])
                    #     if uploaded_blacklist_file is not None:
                    #         try:
                    #             df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
                    #             st.info(f"Upload thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                    #         except:
                    #             st.error("Kiểm tra lại file, chú ý file excel chỉ gồm 1 thông tin là TIN!")
                    # df_dn_tron_thue["TIN"] = df_dn_tron_thue["TIN"].astype(str)
            with col_confirm_button:
                if st.button(f"Xác nhận \n Cấu hình", type='primary', on_click=step2_click_button_configuration):
                    st.toast(f'Đã xáC nhận cấu hình thành công!', icon='🎉')
                    step2_click_button_configuration()

    with container_function_2:
        if st.session_state.step_2:
            st.sidebar.write("""**✅ STEP 2:** *Confirm Configuration*""")       
            side_bar_progress.progress(75, text="**➡️ STEP 3:** *Confirm Data Saving*")
            st.markdown(
                """
                ##### **:red[3. Dữ liệu trích xuất]**
            """
            )
            if len(list_string_io_data) == 0:
                st.info("Chưa có dữ liệu, đợi dữ liệu được tải lên")
            else:    
                # Save string_data from files to dictionary type
                extract_information = list(zip(df_default_field['Thẻ'],  df_default_field['Chỉ tiêu'], df_default_field['Mô tả']))
                with st.expander("3.1 Logs", expanded = False):
                    file_name_index = 1
                    data_row = []
                    for file_name in list(dict_string_io_data.keys()):
                        # Save all data of file_name by row to data_row_i
                        data_row_i = [file_name]
                        st.write(f"+ File {file_name_index}: {file_name}")
                        string_io_data = dict_string_io_data[file_name]
                        dict_io_data = xmltodict.parse(string_io_data)
                        #st.write(dict_io_data)
                        is_blacklist = False
                        is_missing_SHDon = False
                        is_missing_MHSo = False
                        is_KHHDON_start_C = False
                        for information_i in extract_information:
                            roots = information_i[0]
                            leaf = information_i[1]
                            data_name = information_i[2]
                            try:
                                data_roots = get_nested(dict_io_data, roots.split("/"))
                                data_leaf = data_roots[leaf]
                            except TypeError:
                                st.error(f"*Dữ liệu {data_name} bị thiếu trong file {file_name}*")
                                data_leaf = None
                            except:
                                st.error(f"*Dữ liệu {data_name} không có trong cấu trúc {file_name}*")
                                data_leaf = None
                            st.write(f"*Dữ liệu {data_name}: {roots}/{leaf} -> {data_leaf}*")

                            # Check MST in DS Blacklist or not
                            if data_name == "MST" and not is_blacklist:
                                is_blacklist = int(data_leaf) in list(df_dn_tron_thue["TIN"].astype(int))
                            # Modified datatype of VND currency
                            if leaf in ["TgTCThue", "TgTThue", "TgTTTBSo"] and data_leaf != None:
                                data_leaf = currency_format(int(float(data_leaf.replace(",", "."))))
                            # Modified category of TCHDon
                            if leaf == 'TCHDon':
                                data_leaf = 'Hóa đơn gốc' if data_leaf == None else  ('Thay thế' if int(data_leaf) == 1 else 
                                ('Điều chỉnh' if int(data_leaf) == 2 else data_leaf))
                            if leaf == "KHHDon":
                                data_leaf = str(data_leaf)
                                is_KHHDON_start_C = True if str(data_leaf).startswith("C") else False
                            if leaf == "SHDon":
                                data_leaf = str(data_leaf)
                                is_missing_SHDon = True if str(data_leaf) in ["None", ""] else False
                            if leaf == "MHSo":
                                data_leaf = str(data_leaf)
                                is_missing_MHSo = True if str(data_leaf) in ["None", ""] else False
                            data_row_i.append(data_leaf)

                        # Add Mã số thuê - concat to -2 last column
                        try:
                            KHHDon = str(data_row_i[1]) if data_row_i[1] != None else ""
                            SHDon = str(data_row_i[2]) if data_row_i[2] != None else ""
                        except:
                            KHHDon, SHDon = "", ""
                        
                        so_hieu_hoa_don = f"{KHHDon}{SHDon}"
                        data_row_i.append(so_hieu_hoa_don)
                        # Add blacklist information to -n last column
                        data_row_i.append(is_blacklist)
                        # Add is missing SHDon information to -n+1 last column
                        data_row_i.append(is_missing_SHDon)
                        # Add is missing MHSo with KHHDON start with C information to -n+2 last column
                        data_row_i.append(is_missing_MHSo and is_KHHDON_start_C)
                        # Save data row, go to next file
                        data_row.append(data_row_i)
                        file_name_index += 1
                information_schemas = ["Tên file"] + list(df_default_field['Mô tả']) + ["Ký hiệu hóa đơn", "DS Trốn thuế", "Thiếu SHDon", "Thiếu MHSo"]
                df_information = pd.DataFrame(data_row, columns=information_schemas)


                with st.expander("3.2 Data Insight", expanded = False):


                    col_distribution_null_check, col_distribution_blacklist, col_distribution_money  = st.columns([3,2,2])
                    with col_distribution_blacklist:
                        #st.dataframe(df_information)
                        df_information_blacklist = df_information.groupby(["DS Trốn thuế"]).agg({"Số hóa đơn": 'nunique',  "Mã số thuế người bán": 'nunique'}).reset_index()
                        df_information_blacklist["Trốn thuế - Validation"] = df_information_blacklist["DS Trốn thuế"].apply(lambda x: "Có - Trốn thuế" if x else "Không - Trốn thuế")
                        fig = px.pie(
                            df_information_blacklist,
                            values="Số hóa đơn",
                            title=f"Biểu đồ Phân bố số lượng hồ sơ <br> ---- <br> Số lượng hồ sơ: {np.sum(df_information_blacklist['Số hóa đơn'])}",
                            color="Trốn thuế - Validation",
                            names="Trốn thuế - Validation",
                            hole=0.3,
                            color_discrete_sequence=[
                                "#6495ED",
                                "#C1CDCD"
                            ])
                        fig.update_layout(hovermode="x unified", height=400)
                        fig.update_traces(textposition='inside', textinfo='label+value')
                        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                    with col_distribution_money:
                        df_information_money = df_information[["Mã số thuế người bán", "Tổng tiền thanh toán bằng số"]]
                        df_information_money["Tổng tiền thanh toán bằng số"] = df_information_money["Tổng tiền thanh toán bằng số"].apply(lambda x: float(x.replace(",", "")))
                        total_money_amount = currency_format(int(sum(df_information_money['Tổng tiền thanh toán bằng số'])))
                        mean_money_amount = currency_format(int(np.mean(df_information_money['Tổng tiền thanh toán bằng số'])))
                        fig = px.histogram(
                            df_information_money,
                            x="Tổng tiền thanh toán bằng số",
                            color_discrete_sequence=["#6495ED"],
                            title=f"Biểu đồ Phân bố lượng tiền thanh toán <br> ---- <br> Tổng: {total_money_amount} VND <br> Trung bình: {mean_money_amount} VND",
                            marginal="box"
                        )
                        fig.update_layout(hovermode="x unified", height=400)
                        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                with col_distribution_null_check:
                    df_null_check = df_information[list(df_default_field['Mô tả']) + ["DS Trốn thuế"]]
                    null_counts = df_null_check.isnull().sum()
                    non_null_counts = df_null_check.notnull().sum()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=null_counts.index,
                        y=null_counts.values,
                        name='Null',
                        marker_color='red'
                    ))
                    fig.add_trace(go.Bar(
                        x=non_null_counts.index,
                        y=non_null_counts.values,
                        name='Non-Null',
                        marker_color='blue'
                    ))
                    fig.update_layout(
                        title='Biểu đồ Thống kê hiện trạng của các cột dữ liệu trích xuất',
                        xaxis_title='Columns',
                        yaxis_title='Count',
                        barmode='group'
                    )
                    fig.update_layout(hovermode="x unified", height=400)
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with st.expander("3.3 Data Export", expanded = True):
                    # Warning Definination
                    def color_warning_1(value):
                        return f"background-color: red;" if value == True else None

                    # Show dataframe
                    st.dataframe(df_information.style.applymap(color_warning_1), use_container_width=True, hide_index=True)
                    total_missing_SHDon = df_information[df_information["Thiếu SHDon"]].shape[0]
                    total_missing_MHSo = df_information[df_information["Thiếu MHSo"]].shape[0]
                    total_samples = df_information.shape[0]
                    st.warning(f"⚠️Số lượng bản ghi thiếu Số hiệu đơn (SHDon) là: {total_missing_SHDon}/{total_samples}. \n ⚠️Số lượng bản ghi thiếu Mã hiệu số (MHSo) là: {total_missing_MHSo}/{total_samples}")

                    # Download file as format
                    current_date = datetime.strftime(datetime.now(), '%Y%m%d%H%M')

                    @st.cache_data
                    def convert_df(df):
                        # Cache the conversion to prevent computation on every rerun
                        return df.to_csv().encode("utf-8-sig")
                    @st.cache_data
                    def to_excel(df):
                        """
                        Convert dataframe to excel type ready for streamlit download
                        """
                        from io import BytesIO
                        in_memory_fp = BytesIO()
                        df.to_excel(in_memory_fp)
                        in_memory_fp.seek(0, 0)
                        return in_memory_fp.read()
                    
                    df_information_excel = to_excel(df_information)
                    col_download_1, col_download_info_1 = st.columns([1, 2])
                    with col_download_1:
                        btn1 = st.download_button(
                            label="⬇️ Download (excel) ⬇️",
                            data=df_information_excel,
                            file_name=f"{current_date}_DuLieuTrichXuat.xlsx",
                        )
                    with col_download_info_1:
                        st.write(f"*File name: {current_date}_DuLieuTrichXuat.xlsx sau khi download*")

                    df_information_csv = convert_df(df_information)
                    col_download_2, col_download_info_2 = st.columns([1, 2])
                    with col_download_2:
                        btn2 = st.download_button(
                            label="⬇️ Download (csv) ⬇️",
                            data=df_information_csv,
                            file_name=f"{current_date}_DuLieuTrichXuat.csv"
                        )
                    with col_download_info_2:
                        st.write(f"*File name: {current_date}_DuLieuTrichXuat.csv sau khi download*")
        else:
            with st.sidebar:
                with st.spinner("Đang đợi xác nhận cấu hình..."):
                    while True:
                        if st.session_state.step_2 == True:
                            break
    with container_function_3:
        _, col_button_save_data = st.columns([9, 1])
        with col_button_save_data:
            df_information_save = df_information[information_schemas]
            if st.button(f"Xác nhận \n Lưu DL", type = 'primary',
                          on_click=step4_click_button_done, args=[df_information_save]):
                return 0
                # if st.session_state.step_4 == True:
            else:
                with st.sidebar:
                    with st.spinner("Đang đợi xác nhận lưu trữ dữ liệu ..."):
                        while True:
                            if st.session_state.step_3 == True:
                                break

import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader

with open("config/authentication.yaml", encoding="utf8") as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)
_, col_authen, _ = st.columns([3, 3, 4])
with col_authen:
    # st.write("🔓 NCB - Kế toán nội bộ | Trích xuất dữ liệu Hóa đơn điện tử <br> Login Page")
    name, authentication_status, username = authenticator.login(location = "main", max_concurrent_users=20)





# Hard code - 1 role
role_setting_dev = 'ncb_cds'
role_setting_user = 'ncb_ketoannoibo'
if authentication_status:
    col_notification, col_image = st.columns([8, 2])
    with col_notification:
        st.markdown("#### :blue[CÔNG CỤ TRÍCH XUẤT DỮ LIỆU HÓA ĐƠN ĐIỆN TỬ - KẾ TOÁN NỘI BỘ NCB]")
        st.info(f"**:blue[Successful login.]** \n *Chú ý: Công cụ hiển thị tốt nhất trên trình duyệt ở mức thu nhỏ 75%*")
        st.session_state["role"] = role_setting_user
        authenticator.logout("Logout", "main")
    with col_image:
        image1 = Image.open(
            "img/ncb_icon.jpg"
        )
        st.image(
            image1, caption="", use_column_width="never", width=115
        )
    main()
elif authentication_status == False:
    with col_authen:
        st.error("Username/Password is incorrect")
    st.session_state["role"] = "wrong_credential"
elif authentication_status == None:
    with col_authen:
        st.info("Please enter your Username and Password")
    st.session_state["role"] = "guest"
else:
    st.session_state["role"] = "other"