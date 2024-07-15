import streamlit as st
import sys
import os

import math
import numpy as np
import pandas as pd
import random
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="Tài liệu mô tả sử dụng công cụ", page_icon="🔎", layout="wide"
)

import base64
#from sharepoint import *

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1600" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.markdown("Sử dụng Ctrl+ để phóng to file PDF trên web hoặc xem dưới dạng tải về")
displayPDF("documents/Quick Start Guide - Công cụ trích xuất dữ liệu NCB.pdf")