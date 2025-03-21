import streamlit as st
from pages import login, annotation
from utils.session import initialize_session

initialize_session()

# Route between pages
if st.session_state.page == "login":
    login.show_login_page()
elif st.session_state.page == "annotation":
    annotation.show_annotation_page()