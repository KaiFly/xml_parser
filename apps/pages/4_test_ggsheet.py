import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Function to authenticate and connect to Google Sheets
def connect_to_google_sheets(json_keyfile_name, sheet_url):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_name, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).sheet1  # Open the first sheet of the spreadsheet
    return sheet

# Function to load data from Google Sheets
def load_data(sheet):
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# Function to update data in Google Sheets
def update_google_sheet(sheet, df):
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())


st.title("Google Sheets Editor with Streamlit")

json_keyfile_name = st.text_input("Enter the path to your JSON keyfile", "path/to/your/json_keyfile.json")
sheet_url = st.text_input("Enter the URL of your Google Sheet", "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit")

if st.button("Load Data"):
    try:
        sheet = connect_to_google_sheets(json_keyfile_name, sheet_url)
        df = load_data(sheet)
        st.write("Data Loaded Successfully!")
        st.dataframe(df)

        st.write("Edit the data below:")
        edited_df = st.experimental_data_editor(df)

        if st.button("Update Google Sheet"):
            update_google_sheet(sheet, edited_df)
            st.write("Google Sheet Updated Successfully!")
    except FileNotFoundError:
        st.error("The JSON key file was not found. Please check the path and try again.")
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("The Google Sheet was not found. Please check the URL and try again.")
    except PermissionError:
        st.error("The service account does not have permission to access the Google Sheet. Please check the sharing settings.")
    except Exception as e:
        st.error(f"An error occurred: {e}")