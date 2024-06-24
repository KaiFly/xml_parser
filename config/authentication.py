import streamlit as st
import streamlit_authenticator as stauth

with open("authentication.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)
_, col_authen, _ = st.columns([2, 2, 2])
with col_authen:
    name, authentication_status, username = authenticator.login(
        "🔓 NCB - Kế toán nội bộ - Trích xuất dữ liệu Hóa đơn điện tử | Login ", "main"
    )
# Hard code - 1 role
role_setting = 'ncb_cds_team'
if authentication_status:
    st.info(f"**:green[Successful login.]**")
    st.session_state["role"] = role_setting
    main()
    authenticator.logout("Logout", "main")
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