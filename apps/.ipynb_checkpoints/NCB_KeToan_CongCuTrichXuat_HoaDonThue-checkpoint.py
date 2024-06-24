import streamlit as st

import xmltodict
import json
import streamlit as st
from io import StringIO
import pandas as pd
import numpy as np
import openpyxl

## Load setting static data: data path, data source
st.set_page_config(page_title="NCB Accountant - Parser", page_icon="👋", layout="wide")
error_setting_message = (
    "ERROR!"
)
# def main():
## Load setting path based on account authentication
## Setting header of Main


col_1, _, col_2 = st.columns([1, 0.1, 1])
with col_1:
    with st.container(border = True):
        st.markdown(
            """
        ### **1.FILE INPUT - XML**
        """
        )
        tab_11, tab_12 = st.tabs(['Single File', 'Multiple Files'])
        with tab_11:
            is_uploaded = False
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                # st.write(bytes_data)
                # # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # st.write(stringio)
                # To read file as string:
                string_data = stringio.read()
                #st.write(string_data)
                is_uploaded = True
    
    with st.container(border = True):
        st.markdown(
            """
        ### **:black[2. CONFIGURATION]**
        """
        )
        st.write("👉🏻 *User nhập cấu hình cho dữ liệu cần trích xuất*")
        if not is_uploaded:
            st.spinner("Waiting...")
        else:
            st.write("Danh sách các trường phát hiện từ file đầu vào")
            import xmltodict
            information_1 = "HDon"
            information_2 = "DLHDon"
            information_3 = "NDHDon"
            doc = xmltodict.parse(string_data)
            data_parsed = doc[information_1][information_2][information_3]
            list_information_1 = list(data_parsed.keys())
            # df = pd.Series(" ".join(string_data.replace('\n', '').strip(' b\'').strip('\'').split('\' b\'')).split('\\n')).str.split(',', expand=True)
            # st.dataframe(df)
            information_selected = st.selectbox(
                label="🎯 Các loại thông tin phát hiện trong dữ liệu",
                options=list_information_1,
                index=0,
            )
            data_type_selected = st.selectbox(
                label="🎯 Định dạng dữ liệu tải về",
                options=['xlsx', 'csv'],
                index=0,
            )
with col_2:
    with st.container(border = True):
        st.markdown(
            """
        ### **:black[3. DATA PREVIEW]**   
        """
        )
        st.write("👉🏻 *User kiểm tra bảng dữ liệu trước khi tải về*")
        
        # st.write(data_parsed)
        data_extracted = data_parsed[information_selected]
        is_df = False
        if np.isscalar(data_extracted[list(data_extracted.keys())[0]]):
            st.write(data_extracted)
            final_data = data_extracted
        else:
            df_data_extracted = pd.DataFrame.from_dict(data_extracted)
            st.dataframe(df_data_extracted.T)
            final_data = df_data_extracted
            is_df = True
    with st.container(border = True):
        st.markdown(
            """
        ### **:black[4. FILE OUTPUT - EXCEL/CSV]**   
        """
        )
        st.write("👉🏻 *User tải về dữ liệu Excel như preview*")
        def to_excel(df: pd.DataFrame):
            from io import BytesIO
    
            in_memory_fp = BytesIO()
            df.to_excel(in_memory_fp)
            in_memory_fp.seek(0, 0)
            return in_memory_fp.read()
        if data_type_selected == 'xlsx':
            if is_df:
                st.download_button(
                    label="⬇️ Download ⬇️",
                    data=to_excel(final_data),
                    file_name=f"Output.xlsx",
                )
            else:
                st.download_button(
                    label="⬇️ Download ⬇️",
                    data=final_data,
                    file_name=f"Output.xlsx",
                )
        # st.success("Downloaded Pattern file")

