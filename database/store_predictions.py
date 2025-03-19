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

def fetch_qp_ids(df):
    """Fetches qpID for each (query, product) pair and merges with DataFrame."""
    with connect_to_db() as conn:
        query = """SELECT "qpID", query, product FROM tbl_queryproducts"""
        qp_df = pd.read_sql(query, conn)
        return df.merge(qp_df, on=['query', 'product'], how='left')
    
def fetch_esci_ids(df):
    """
    Fetch ESCI IDs from the database and merge them with predictions.
    """
    with connect_to_db() as conn:
        query = """SELECT "esciID", "esciLabel" FROM tbl_esci"""
        esci_df = pd.read_sql(query, conn)

    if esci_df.empty:
        print("Error: tbl_esci returned no data!")
        return df

    label_mapping = {0: "E", 1: "S", 2: "C", 3: "I"}
    df['esci_label'] = df['esci_label'].map(label_mapping).astype(str)

    esci_df['esciLabel'] = esci_df['esciLabel'].astype(str).str.strip().str.upper()

    print(f"DEBUG: Converted esci_labels in predictions: {df['esci_label'].unique()}")
    print(f"DEBUG: Unique esciLabels in tbl_esci: {esci_df['esciLabel'].unique()}")

    merged_df = df.merge(esci_df, left_on='esci_label', right_on='esciLabel', how='left').drop(columns=['esciLabel'])

    missing_esci = merged_df['esciID'].isna().sum()
    if missing_esci > 0:
        print(f"Warning: {missing_esci} missing esciID values! Check esci_label format.")

    return merged_df

def get_latest_model_id(model_type):
    """Fetches the most recent modelID for the given model type."""
    with connect_to_db() as conn:
        query = """
        SELECT "modelID"
        FROM "tbl_models"
        WHERE "modelclassID" = (SELECT "modelclassID" FROM "tbl_modelclass" WHERE "modelClass" = %s)
        ORDER BY "modelID" DESC
        LIMIT 1
        """
        df = pd.read_sql(query, conn, params=[model_type])
        
        if df.empty:
            print(f"DEBUG: No modelID found for model type {model_type}.")
            return None
        
        model_id = df.iloc[0]["modelID"]
        print(f"DEBUG: Found modelID {model_id} for model type {model_type}.")
        return model_id

def store_predictions_in_db(df, model_type):
    """Inserts or updates predictions in tbl_predictions."""
    with connect_to_db() as conn:
        model_id = get_latest_model_id(model_type)
        print(f"DEBUG: Retrieved modelID {model_id} for model type {model_type}.")
        if model_id is None:
            print(f"Error: No model found for {model_type}. Exiting prediction storage.")
            return

        df = fetch_qp_ids(df)
        print(f"DEBUG: Retrieved {df['qpID'].notnull().sum()} qpIDs from {len(df)} rows.")

        df = fetch_esci_ids(df)
        print(f"DEBUG: Retrieved {df['esciID'].notnull().sum()} esciIDs from {len(df)} rows.")

        if df[['qpID', 'esciID']].isnull().any().any():
            print("ERROR: Some qpID or esciID values are missing. Exiting storage.")
            return

        with conn.cursor() as cur:
            insert_query = """
            INSERT INTO tbl_predictions ("qpID", "esciID", "modelID", "confidenceScore")
            VALUES (%s, %s, %s, %s)
            ON CONFLICT ("qpID", "modelID") DO UPDATE
            SET "esciID" = EXCLUDED."esciID", "confidenceScore" = EXCLUDED."confidenceScore"
            """

            data_to_insert = [
                (int(row['qpID']), int(row['esciID']), int(model_id), float(row['score']))
                for _, row in df.iterrows()
            ]
            print(f"DEBUG: Preparing to insert {len(data_to_insert)} predictions into tbl_predictions.")

            if len(data_to_insert) == 0:
                print("ERROR: No data to insert into tbl_predictions.")
                return

            cur.executemany(insert_query, data_to_insert)
            conn.commit()
            print("Predictions successfully stored in tbl_predictions.")
