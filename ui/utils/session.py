import streamlit as st
import pandas as pd
from utils.database import run_query, init_connection

def initialize_session():
    # loading 50 rows that are not already in the golden dataset
    # selecting from the tbl_predictions table 
    query = """
        SELECT 
            p."qpID",
            p."confidenceScore",
            p."esciID",
            p."modelID",
            qp.query,
            qp.product
        FROM 
            tbl_predictions AS p
        JOIN 
            tbl_queryproducts AS qp 
        ON 
            p."qpID" = qp."qpID"
        WHERE 
            NOT EXISTS (
                SELECT 1 
                FROM tbl_golden g 
                WHERE g."qpID" = p."qpID"
            )
        LIMIT 50;
    """  
    st.session_state.df = run_query(query)  

    if st.session_state.df is None or st.session_state.df.empty:
        st.error("Failed to load data from the database.")
        return  
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
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

def reset_session_variables():
    query = """
        SELECT 
            p."qpID",
            p."confidenceScore",
            p."esciID",
            p."modelID",
            qp.query,
            qp.product
        FROM 
            tbl_predictions AS p
        JOIN 
            tbl_queryproducts AS qp 
        ON 
            p."qpID" = qp."qpID"
        WHERE 
            NOT EXISTS (
                SELECT 1 
                FROM tbl_golden g 
                WHERE g."qpID" = p."qpID"
            )
        LIMIT 50;
    """  
    st.session_state.df = run_query(query)  

    if st.session_state.df is None or st.session_state.df.empty:
        st.error("Failed to load data from the database.")
        return  
    
    st.session_state.index = 0
    st.session_state.annotations = []
    st.session_state.previous_index = None
    st.session_state.annotation_history = []
    st.rerun()