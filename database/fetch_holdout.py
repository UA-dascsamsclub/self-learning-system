import psycopg2
import pandas as pd
from database.fetch_data import preprocess_text
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def connect_to_db():
    """Establishes a connection to the PostgreSQL database.
        
    :return: psycopg2 connection object.
    """
    return psycopg2.connect(
        dbname=DB_NAME, 
        user=DB_USER, 
        password=DB_PASSWORD, 
        host=DB_HOST,
        port=DB_PORT
    )

def fetch_holdout(limit=100000):
    """
    Fetches query-product pairs and their ESCI labels from the holdout dataset.
    Applies preprocessing to both query and product text fields.
    
    :param limit: Maximum number of rows to fetch. Defaults to 100,000.
    :return: DataFrame containing preprocessed query, product, and esciID columns, or None if an error occurs.
    """

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
    # Run data fetching when script is executed directly
    df = fetch_holdout(limit=1000)
    if df is not None:
        print(df.head())
