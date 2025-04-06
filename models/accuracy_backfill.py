import psycopg2
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from database.fetch_data import connect_to_db
from database.store_metrics import store_model_metrics
from database.fetch_holdout import fetch_holdout
from models.cross_encoder_inference import predict_labels as cross_encoder_predict
from models.bi_encoder_inference import predict_labels as bi_encoder_predict

def store_init_model_metrics(model_id, df):
    """Stores accuracy metrics in tbl_accuracy and links them to the latest model in tbl_model_accuracy."""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            # Fetch latest model ID for the given model type
            # model_id = 1
            
            # Insert accuracy metrics into tbl_accuracy
            insert_accuracy_query = """
            INSERT INTO tbl_accuracy ("precision", "recall", "microf1")
            VALUES (%s, %s, %s)
            RETURNING "accuracyID";
            """
            # Extract metrics from dataframe
            precision = float(df["precision"].values[0])
            recall = float(df["recall"].values[0])
            microf1 = float(df["microf1"].values[0])

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

    connect_to_db()
    # Load the model
    model_path = "models/model_ce_finetuned/"
    model_id = 4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Fetch holdout data
    df = fetch_holdout(limit=100000)

    if df is not None:
        # Make predictions using the holdout dataset and store them in a new column
        df["esci_label_predicted"] = cross_encoder_predict(df, model=model_path)["esci_label"]

        df = df.dropna(subset=["esci_label_predicted"])

        df["esciID"] = df["esciID"].astype(int)
        df["esci_label_predicted"] = df["esci_label_predicted"].astype(int)

        # Proceed with metrics
        y_true = df["esciID"]
        y_pred = df["esci_label_predicted"]
            
        # Compute metrics
        precision = precision_score(y_true, y_pred, average="micro")
        recall = recall_score(y_true, y_pred, average="micro")
        f1 = f1_score(y_true, y_pred, average="micro")
        
        df_metrics = pd.DataFrame([{
            "precision": precision,
            "recall": recall,
            "microf1": f1
        }])

        # Store metrics in the database
        print("Evaluation Metrics:")
        print(f"Precision     : {precision:.4f}")
        print(f"Recall (micro): {recall:.4f}")
        print(f"F1 Score (micro): {f1:.4f}")

        confirm = input(f"\nStore metrics for model_id {model_id}? (y/n): ")
        if confirm.lower() == "y":
            store_init_model_metrics(model_id, df_metrics)
            print("Metrics stored in database.")
        else:
            print("❌ Metrics not stored.")
    else:
        print("❌ No holdout data fetched.")