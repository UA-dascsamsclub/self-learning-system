import streamlit as st
import pandas as pd
from utils.database import run_query

def initialize_session():
    if 'df' not in st.session_state:
        st.session_state.df = run_query("SELECT * FROM tbl_queryproducts LIMIT 10000;;")

    if 'index' not in st.session_state:
        st.session_state.index = 0

    if 'annotations' not in st.session_state:
        st.session_state.annotations = []

    if 'percent_confident' not in st.session_state:
        st.session_state.percent_confident = {}

    if 'model_prediction' not in st.session_state:
        st.session_state.model_prediction = {}

    if 'page' not in st.session_state:
        st.session_state.page = 'login'

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'username' not in st.session_state:
        st.session_state.username = None

    if 'previous_index' not in st.session_state:
        st.session_state.previous_index = None

    if 'annotation_history' not in st.session_state:
        st.session_state.annotation_history = []