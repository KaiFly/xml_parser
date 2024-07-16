

sharepoint_url_path = 'https://ncbbank-my.sharepoint.com/:f:/g/' 
sharepoint_folder_path = 'personal/dungda_ncb-bank_vn/EheM5Epid2VOoSa2GDopprkB2o12xweJECL-r0OvPX2tDQ?e=xWnuxr'
def save_data_to_sharepoint():
    return 0
import pandas as pd
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


# def read_XML_data(conn):
#     # conn = st.connection('gcs', type=FilesConnection)
#     df = conn.read("ncb_xml_parsing_tool/data_XML.csv", input_format="csv", ttl=600)
#     df.columns = ["Tên file","Ký hiệu mẫu số hóa đơn","KH hóa đơn","Số hóa đơn","Ngày lập","Mã hiệu số","Tính chất hóa đơn","Tên người bán","Mã số thuế người bán","Tổng tiền (chưa có thuế GTGT)","Tổng tiền thuế GTGT","Tổng tiền thanh toán bằng số","Ký hiệu hóa đơn","DS Trốn thuế","Thiếu SHDon","Thiếu MHSo","Thời gian cập nhập","User cập nhập"]
#     return df
