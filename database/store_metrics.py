import psycopg2
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from database.store_predictions import get_latest_model_id
import pandas as pd

def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def store_model_metrics(model_type, model_id, df):
    """Stores accuracy metrics from df in tbl_accuracy and links them to the latest model in tbl_model_accuracy."""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            '''
            # Fetch latest model ID for the given model type
            model_id = int(get_latest_model_id(model_type))
            if model_id is None:
                print(f"Error: No model found for {model_type}. Exiting storage.")
                return'
            '''
            
            # Extract metrics from dataframe
            precision = float(df["precision"].values[0])
            recall = float(df["recall"].values[0])
            microf1 = float(df["microf1"].values[0])
            
            # Insert accuracy metrics into tbl_accuracy
            insert_accuracy_query = """
            INSERT INTO tbl_accuracy ("precision", "recall", "microf1")
            VALUES (%s, %s, %s)
            RETURNING "accuracyID";
            """
            cur.execute(insert_accuracy_query, (precision, recall, microf1))
            accuracy_id = cur.fetchone()[0]
            
            # Insert into bridge table tbl_model_accuracy
            insert_model_accuracy_query = """
            INSERT INTO tbl_modelaccuracy ("modelID", "accuracyID")
            VALUES (%s, %s);
            """
            cur.execute(insert_model_accuracy_query, (model_id, accuracy_id))
            
            conn.commit()
            print("Model accuracy metrics successfully stored.")


if __name__ == "__main__":
    # Test the function with dummy data
    df = {
        "precision": [0.85],
        "recall": [0.90],
        "microf1": [0.87]
    }
    df = pd.DataFrame(df)

    store_model_metrics("crossencoder", df)
    store_model_metrics("biencoder", df)
