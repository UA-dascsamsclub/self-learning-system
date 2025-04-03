import os
import sys
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score
from cross_encoder_inference import predict_labels as cross_encoder_predict
sys.path.append('../self-learning-system')
from database.fetch_holdout import fetch_holdout

def calculate_ce_metrics(df, model):
    """
    Fetches the holdout labels and model predictions from holdout data from the database,
    and computes accuracy, recall (micro), and F1 score (micro).
    
    Args:
        limit (int): The number of records to fetch from the database.
        
    Returns:
        dict: Dictionary containing accuracy, recall_micro, and f1_micro scores.
    """



    # Make predictions using the holdout dataset and store them in a new column
    df["esci_label_predicted"] = cross_encoder_predict(df, model=model)["esci_label"]
    
    if df["esci_label_predicted"].isnull().all():
        print("No predictions generated.")
        return None

    # Extract labels
    y_true = df['esciID']
    y_pred = df['esci_label_predicted']

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    metrics = {
        'accuracy': accuracy,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }

    print(df['esci_label_predicted'].value_counts())
    print(df['esciID'].value_counts())    

    print("Evaluation Metrics:")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")

    return metrics

if __name__ == "__main__":
    # Define model path 
    model_path = "models/model_ce_trained/"

    
    # Initialize the cross-encoder model
    model = CrossEncoder(
        model_path,
        num_labels=4,
        automodel_args={'ignore_mismatched_sizes': True}
    )
        # Fetch golden data (ground truth)
    df = fetch_holdout(limit=1000)
    if df is None or df.empty:
        print("No holdout data fetched.")
        
    
    ce_scores = calculate_ce_metrics(df, model)
    print(ce_scores)
