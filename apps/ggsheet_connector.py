import streamlit as st
from streamlit_gsheets import GSheetsConnection
# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)


def read_TIN_data(sheet_id = 0, schemas = ["TIN"], max_entries = 5):
    try:
        df = conn.read(worksheet=sheet_id, max_entries=max_entries)
        df = df[schemas]
        # df[schemas[0]] = df[schemas[0]].astype(str)
    except Exception as e:
        st.error("Error", e)
        df = pd.DataFrame(columns = ["TIN"])
    return df

def read_XML_data(sheet_id = 1, max_entries = 5):
    try:
        df = conn.read(worksheet=sheet_id, max_entries=max_entries)
    except Exception as e:
        st.error("Error", e)
    return df

def read_TIN_metadata(sheet_id=2, max_entries=5):
    #try:
    df = conn.read(worksheet=sheet_id, max_entries=max_entries)
    return df
    # except Exception as e:
    #     print("Error", e)


def write_TIN_data(data, sheet_id = 0, schemas = ["TIN"]):
    data_update = data[schemas]
    data_update[schemas[0]] = data_update[schemas[0]].astype(str)
    conn.update(data=data_update, worksheet=sheet_id)
    return 0

def write_TIN_metadata(data, sheet_id=2, schemas=["update_time", "user", "data_size"]):
    data_update = data[schemas]
    conn.update(data=data_update, worksheet=sheet_id)
    return 0
