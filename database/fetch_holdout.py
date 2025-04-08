import psycopg2
import pandas as pd
from database.fetch_data import preprocess_text
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

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
    SELECT h.query, h.product, h."esciID"
    FROM tbl_holdout h
    LIMIT {limit}
    """

    try:
        conn = connect_to_db()
        df = pd.read_sql(query, conn)
        conn.close()
        # Preprocess the DataFrame
        df['query'] = df['query'].apply(preprocess_text)
        df['product'] = df['product'].apply(preprocess_text)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
if __name__ == "__main__":
    df = fetch_holdout(limit=1000)
    if df is not None:
        print(df.head())
