import psycopg2
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from database.store_predictions import get_latest_model_id
import pandas as pd

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

def store_model_metrics(model_id, df):
    """
    Stores accuracy metrics from the given DataFrame into tbl_accuracy,
    and creates a mapping between the model and its metrics in tbl_modelaccuracy.

    :param model_id: ID of the model to associate the metrics with.
    :param df: DataFrame containing 'precision', 'recall', and 'microf1' columns.
    :return: None
    """    
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            
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

    # Storing metrics for both crossencoder and biencoder (IDs must be valid integers, not strings)
    store_model_metrics(2, df)
    store_model_metrics(1, df)
