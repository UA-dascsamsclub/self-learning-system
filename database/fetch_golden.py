import psycopg2
import pandas as pd
from database.fetch_data import preprocess_text
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def connect_to_db():
    """Establishes a connection to the PostgreSQL database.
    
    :return: psycopg2 connection object."""
    return psycopg2.connect(
        dbname=DB_NAME, 
        user=DB_USER, 
        password=DB_PASSWORD, 
        host=DB_HOST,
        port=DB_PORT
    )

def fetch_golden(limit=1000):
    """
    Fetches the most recent golden labeled query-product pairs along with their ESCI labels.
    Applies preprocessing to both query and product text fields.
    
    :param limit: Parameter to limit row returned. Defaults to 1000.
    :return: DataFrame containing preprocessed query, product, and esciID columns, or None if an error occurs.
    """

    query = f"""
    SELECT qp.query, qp.product, g."esciID"
    FROM tbl_golden g
    JOIN tbl_queryproducts qp ON (qp."qpID" = g."qpID")
    WHERE g."modelID" = (SELECT MAX("modelID") FROM tbl_golden)
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
    df = fetch_golden()
    if df is not None:
        print(df.head())