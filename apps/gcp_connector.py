import streamlit as st
from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
from io import StringIO, BytesIO

### Sharepoint - test
sharepoint_url_path = 'https://ncbbank-my.sharepoint.com/:f:/g/' 
sharepoint_folder_path = 'personal/dungda_ncb-bank_vn/EheM5Epid2VOoSa2GDopprkB2o12xweJECL-r0OvPX2tDQ?e=xWnuxr'
def save_data_to_sharepoint():
    return 0
import pandas as pd

### Google Cloud Storage - prod
project_name = 'ncb-app'
bucket_name = st.secrets.connection.gcs.filesystem['bucket_name']
TIN_filename = st.secrets.connection.gcs.filesystem['tin_data_file']
XML_filename = st.secrets.connection.gcs.filesystem['xml_data_file']

credentials_dict = {
    'type': "service_account",
    'client_id': st.secrets.connections.gcs.credential['client_id'],
    'client_email': st.secrets.connections.gcs.credential['client_email'],
    'private_key_id': st.secrets.connections.gcs.credential['private_key_id'],
    'private_key': st.secrets.connections.gcs.credential['private_key']
}
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    credentials_dict
)

client = storage.Client(credentials=credentials, project='myproject')
bucket = client.get_bucket(bucket_name)

def upload_data(data, file_name = '', method = 'overwrite', is_drop_duplicates = True):
    blob = bucket.blob(file_name)
    if method == 'overwrite':
        blob.upload_from_file(data)
        return f"Successfull - {method} - {file_name}"
    elif method == 'append':
        data = blob.download_as_bytes()
        df_load = pd.read_csv(BytesIO(data))
        df_append = pd.concat(df_load, data)
        if is_drop_duplicates:
            df_append.drop_duplicates(inplace=True)
        blob.upload_from_file(df_append)
        return f"Successfull - {method} - {file_name}"
    else:
        return "Wrong method!"
    
def download_data(file_name = ''):
    blob = bucket.blob(file_name)
    file_obj = BytesIO()
    data = blob.download_to_file(file_obj)
    df_load = pd.read_csv(data)
    #df_load = pd.read_csv(BytesIO(data))
    return df_load



# def read_XML_data(conn):
#     # conn = st.connection('gcs', type=FilesConnection)
#     df = conn.read("ncb_xml_parsing_tool/data_XML.csv", input_format="csv", ttl=600)
#     df.columns = ["Tên file","Ký hiệu mẫu số hóa đơn","KH hóa đơn","Số hóa đơn","Ngày lập","Mã hiệu số","Tính chất hóa đơn","Tên người bán","Mã số thuế người bán","Tổng tiền (chưa có thuế GTGT)","Tổng tiền thuế GTGT","Tổng tiền thanh toán bằng số","Ký hiệu hóa đơn","DS Trốn thuế","Thiếu SHDon","Thiếu MHSo","Thời gian cập nhập","User cập nhập"]
#     return df




# from st_files_connection import FilesConnection
# # Create connection object and retrieve file contents.

# def read_TIN_data(conn):
#     def reformat_TIN(x):
#         x = str(x)
#         x_split = x.split("	")
#         if len(x_split) == 1:
#             return x
#         else:
#             return x_split[1]
#     # conn = st.connection('gcs', type=FilesConnection)
#     df = conn.read("ncb_xml_parsing_tool/data_TIN.csv", input_format="csv", ttl=600)
#     df.columns = ["TIN"]
#     df["TIN"] = df["TIN"].apply(lambda x: reformat_TIN(x))
#     return df


