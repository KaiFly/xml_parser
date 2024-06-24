import streamlit as st

from io import StringIO
import pandas as pd
import numpy as np
import datetime
from datetime import datetime

from utils import data_config, data_tron_thue_csv_path
from utils import get_nested, to_excel
from utils import get_leaves_and_parse
from utils import currency_format

import xmltodict
from PIL import Image

## Load setting static data: data path, data source
st.set_page_config(page_title="NCB Kế toán - Trích xuất dữ liệu hóa đơn", page_icon="👋", layout="wide")
error_setting_message = (
    "ERROR!"
)
def main():
    # Load setting path based on account authentication
    # Setting header of Main

    container_function_1 = st.container(height = 400, border = True)
    container_function_2 = st.container(height = 600, border = True)


    with container_function_1:
        col_11, col_12, col_13 = st.columns([1, 0.2, 1])
        with col_11:
            st.markdown(
                """
            ##### **1. Nhập dữ liệu - Hóa đơn điện tử**
            """
            )
            tab_11, tab_12 = st.tabs(['1.1 Nhập từng hóa đơn', '1.2 Nhập nhiều hóa đơn'])
            # Save input data to dict_string_io_data, list_string_io_data
            is_uploaded = False
            with tab_11:
                dict_string_io_data = {}
                list_string_io_data = []
                uploaded_file = st.file_uploader(label = "⬆️Tải file XML⬆️", accept_multiple_files=False, type = ['xml'])
                if uploaded_file is not None:
                    bytes_data = uploaded_file.getvalue()
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    list_string_io_data.append(string_data)
                    dict_string_io_data[uploaded_file.name] = string_data
                    
            with tab_12:
                is_uploaded = False
                uploaded_files = st.file_uploader(label = "⬆️Tải nhiều file XML lên⬆️", accept_multiple_files=True, type = ['xml'])
                if uploaded_files is not None:
                    dict_string_io_data = {}
                    list_string_io_data = []
                    for uploaded_file in uploaded_files:
                        bytes_data = uploaded_file.read()
                        bytes_data = uploaded_file.getvalue()
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        string_data = stringio.read()
                        list_string_io_data.append(string_data)
                        dict_string_io_data[uploaded_file.name] = string_data
            if len(list_string_io_data) > 0:
                is_uploaded = True        
                st.toast(f'Đã tải lên thành công {len(list_string_io_data)} files!', icon='🎉')
        with col_13:
            st.markdown(
                """
            ##### **:black[2. Tùy chỉnh cấu hình]**
            """
            )
            # with st.spinner("Đang đợi nhập dữ liệu..."):
            #     while True:
            #         if is_uploaded :
            #             break
            with st.expander("2.1 Danh sách các trường dữ liệu (mặc định)", expanded=False):
                df_default_field = pd.DataFrame.from_dict(data_config).T
                st.dataframe(df_default_field, hide_index=True)

            with st.expander("2.2 Danh sách các trường dữ liệu bổ sung", expanded=False):
                # get first file as dictionary to get the tree structure
                if len(list_string_io_data) == 0:
                    st.spinner(f'Đợi tải file dữ liệu lên')
                    list_parsed = []
                else:
                    sample_format = xmltodict.parse(list_string_io_data[0])
                    leaves, dct_parsed = get_leaves_and_parse(sample_format)
                    list_parsed = list(dct_parsed.keys())
                    list_parsed = [i for i in list_parsed if i.split(".")[-1] not in list(df_default_field["Thẻ"])]
                list_append_field = st.multiselect(
                    label="Lựa chọn thêm các DL cần bổ sung: \n *Định dạng tìm kiếm: [Thẻ]/[Chỉ tiêu]* \n *Ví dụ: 'HDon/DLHDon/TTChung/KHMSHDon'*",
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
            with st.expander("2.3 Danh sách MST trong Blacklist trốn thuế", expanded=False):
                # df_dn_tron_thue = pd.read_csv(data_tron_thue_csv_path, sep=",", encoding='cp1252')
                df_dn_tron_thue  = pd.DataFrame(columns=['TIN'])
                st.info(f"Hiện tại danh sách bao gồm: {df_dn_tron_thue.shape[0]}")
                with st.popover("Cập nhập DS MST Blacklist:"):
                    st.write("**Chú ý: gồm 1 cột thông tin duy nhất: TIN**")
                    uploaded_blacklist_file = st.file_uploader(label = "⬆️Tải dữ liệu MST định dạng xlsx ⬆️", accept_multiple_files=False, type = ['xlsx'])
                    if uploaded_blacklist_file is not None:
                        try:
                            df_dn_tron_thue = pd.read_excel(uploaded_blacklist_file)[['TIN']]
                            st.info(f"Upload thành công: {df_dn_tron_thue.shape[0]} bản ghi")
                        except:
                            st.error("Kiểm tra lại file, chú ý file excel chỉ gồm 1 thông tin là TIN!")
                df_dn_tron_thue["TIN"] = df_dn_tron_thue["TIN"].astype(str)
    with container_function_2:
        st.markdown(
            """
            ##### **:black[3. Dữ liệu trích xuất]**
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
                            is_blacklist = str(data_leaf) in list(df_dn_tron_thue["TIN"])
                        data_row_i.append(data_leaf)

                    # Add blacklist information to last column
                    data_row_i.append(is_blacklist)
                    # Save data row, go to next file
                    data_row.append(data_row_i)
                    file_name_index += 1
            information_schemas = ["Tên file"] + list(df_default_field['Mô tả']) + ["DS Trốn thuế"]
            df_information = pd.DataFrame(data_row, columns=information_schemas)


            with st.expander("3.2 Data Insight", expanded = False):
                import plotly.figure_factory as ff
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots

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
                    df_information_money["Tổng tiền thanh toán bằng số"] = df_information_money["Tổng tiền thanh toán bằng số"].apply(lambda x: float(x))
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
                # Show dataframe
                st.dataframe(df_information, use_container_width=True, height = 200, hide_index=True)
                

                # Download file as format
                current_date = datetime.strftime(datetime.now(), '%Y%m%d')
                st.download_button(
                    label="⬇️ Download ⬇️",
                    data=to_excel(df_information),
                    file_name=f"{current_date}_DuLieuTrichXuat.xlsx",
                )

# main()
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
_, col_authen, _, col_image = st.columns([3, 3, 3, 1])
with col_authen:
    # st.write("🔓 NCB - Kế toán nội bộ | Trích xuất dữ liệu Hóa đơn điện tử <br> Login Page")
    name, authentication_status, username = authenticator.login(location = "main", max_concurrent_users=20)
with col_image:
        image1 = Image.open(
            "img/ncb_icon.jpg"
        )
        st.image(
            image1, caption="", use_column_width="never", width=80
        )
# Hard code - 1 role
role_setting = 'ncb_cds_team'
if authentication_status:
    st.info(f"**:blue[Successful login.]**")
    st.session_state["role"] = role_setting
    authenticator.logout("Logout", "main")
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