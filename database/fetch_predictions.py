import psycopg2
import pandas as pd
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

def fetch_predictions(limit=1000):
    query = f"""
    SELECT qp.query, qp.product, p."esciID"
    FROM tbl_predictions p
    JOIN tbl_queryproducts qp ON (qp."qpID" = p."qpID")
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
    df = fetch_predictions()
    if df is not None:
        print(df.head())