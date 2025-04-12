import streamlit as st
from utils.database import authenticate_user, init_connection

def show_login_page():
    """
    Displays the login page for users to authenticate themselves before annotating data.

    Steps:
    1. Initializes a connection to the database using the `init_connection` function.
    2. Displays the login form with fields for username and password.
    3. Verifies the login credentials by calling the `authenticate_user` function.
    4. If authentication is successful, sets session state values and redirects to the annotation page.
    5. If authentication fails or fields are left empty, appropriate error or warning messages are displayed.
    """
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