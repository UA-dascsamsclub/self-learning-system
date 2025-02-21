import psycopg2
import streamlit as st
import pandas as pd
import bcrypt
from psycopg2 import extras
import psycopg2.extras

def init_connection():
    if 'conn' not in st.session_state:
        try:
            st.session_state.conn = psycopg2.connect(
                host=st.secrets["database"]["host"],
                database=st.secrets["database"]["database"],
                user=st.secrets["database"]["user"],
                password=st.secrets["database"]["password"]
            )
        except Exception as e:
            st.error(f"Connection error: {e}")
            st.session_state.conn = None
    return st.session_state.conn

def run_query(query, params=None):
    conn = init_connection()  
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        except Exception as e:
            st.error(f"Query error: {e}")
            return None
    return None

def execute_query(query, params):
    conn = init_connection()  
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        except Exception as e:
            st.error(f"Error executing query: {e}")
            conn.rollback()


# Verify password using bcrypt
def verify_password(stored_password, provided_password):
    conn = init_connection()
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

# Login function
def authenticate_user(username, password):
    conn = init_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT password_hash FROM tbl_analyst WHERE username = %s", (username,))
        result = cur.fetchone()

        if result and verify_password(result[0], password):
            return True
        return False

    except Exception as e:
        conn.rollback()  
        print(f"Database error: {e}")
        return False

    finally:
        cur.close()  

