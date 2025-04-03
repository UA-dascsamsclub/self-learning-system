import psycopg2
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score
from models.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from database.fetch_data import connect_to_db
from database.store_metrics import store_model_metrics
from database.fetch_holdout import fetch_holdout
from database.store_predictions import store_predictions
from models.cross_encoder_inference import predict_labels as cross_encoder_predict

def store_init_model_metrics(model_type, df):
    """Stores accuracy metrics in tbl_accuracy and links them to the latest model in tbl_model_accuracy."""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            # Fetch latest model ID for the given model type
            model_id = 1
            
            # Insert accuracy metrics into tbl_accuracy
            insert_accuracy_query = """
            INSERT INTO tbl_accuracy ("precision", "recall", "microf1")
            VALUES (%s, %s, %s)
            RETURNING "accuracyID";
            """
            # Extract metrics from dataframe
            precision = df["precision"].values[0]
            recall = df["recall"].values[0]
            microf1 = df["microf1"].values[0]

            cur.execute(insert_accuracy_query, (precision, recall, microf1))
            accuracy_id = cur.fetchone()[0]
            
            # Insert into bridge table tbl_model_accuracy
            insert_model_accuracy_query = """
            INSERT INTO tbl_modelaccuracy ("modelID", "accuracyID")
            VALUES (%s, %s);
            """
            cur.execute((insert_model_accuracy_query), (model_id, accuracy_id))
            
            conn.commit()
            print("Model accuracy metrics successfully stored.")

if __name__ == "__main__":

    connect_to_db()
    # Load the model
    model_path = "models/model_ce_trained/"
    model = CrossEncoder(model_path, num_labels=4, automodel_args={'ignore_mismatched_sizes': True})
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.model.to(device)
    
    # Fetch holdout data
    df = fetch_holdout(limit=1000)

    if df is not None:
        # Make predictions using the holdout dataset and store them in a new column
        df["esci_label_predicted"] = cross_encoder_predict(df, model=model)["esci_label"]
                    
            # Calculate metrics
        y_true = df['esciID']
        y_pred = df['esci_label_predicted']
            
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        
        # Store metrics in the database
        #store_model_metrics("crossencoder", accuracy, recall_micro, f1_micro)
        print("Evaluation Metrics:")
        print(f"Accuracy     : {accuracy:.4f}")
        print(f"Recall (micro): {recall_micro:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")