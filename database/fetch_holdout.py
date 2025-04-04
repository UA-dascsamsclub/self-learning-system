import psycopg2
import pandas as pd
from db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=DB_NAME, 
        user=DB_USER, 
        password=DB_PASSWORD, 
        host=DB_HOST,
        port=DB_PORT
    )

def fetch_holdout(limit=100000):
    query = f"""
    SELECT h.query, h.product, e."esciID"
    FROM tbl_holdout h
    JOIN tbl_esci e ON (h."esciID" = e."esciID")
    LIMIT {limit}
    """

    try:
        conn = connect_to_db()
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
if __name__ == "__main__":
    df = fetch_holdout(limit=1000)
    if df is not None:
        print(df.head())
