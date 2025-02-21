import streamlit as st
from utils.database import authenticate_user, init_connection

def show_login_page():
    conn = init_connection()

    if conn is None:
        st.error("Database connection failed")
        return
    
    st.title("Login to Start Annotating")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Begin Annotating")

        if submit_button:
            if username and password:
                if authenticate_user(username, password):
                    st.session_state.page = 'annotation'
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Login failed")
            else:
                st.warning("Please enter both username and password.")