import psycopg2
import pandas as pd
import string
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

puncts = string.punctuation
def preprocess_text(text):
    """
    Removes punctuation from the input text and lowercases all remaining string.
    :param text: Input string.
    :return: String without punctuation and lowercased.
    """
    return ''.join([char for char in str(text).lower() if char not in puncts])

def fetch_query_product_pairs(limit=1000):
    query = f"""
    SELECT query, product
    FROM tbl_queryproducts qp
    WHERE NOT EXISTS (
    SELECT
    FROM tbl_predictions
    WHERE "qpID" = qp."qpID"
    )
    ORDER BY RANDOM()
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
    df = fetch_query_product_pairs()
    if df is not None:
        print(df.head())