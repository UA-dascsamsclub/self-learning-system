import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Add the parent folder to sys.path
sys.path.append('../self-learning-system')

# Import fetch functions from the database folder
from database.fetch_golden import fetch_golden
from database.fetch_predictions import fetch_predictions

def calculate_metrics(limit=1000):
    """
    Fetches the golden labels and model predictions from the database,
    and computes accuracy, recall (micro), and F1 score (micro).
    
    Args:
        limit (int): The number of records to fetch from the database.
        
    Returns:
        dict: Dictionary containing accuracy, recall_micro, and f1_micro scores.
    """

    # Fetch golden data (ground truth)
    golden_df = fetch_golden(limit=limit)
    if golden_df is None or golden_df.empty:
        print("No golden data fetched.")
        return None

    # Fetch predicted data
    predictions_df = fetch_predictions(limit=limit)
    if predictions_df is None or predictions_df.empty:
        print("No predictions data fetched.")
        return None

    # Merge on query and product to align both datasets
    merged_df = pd.merge(
        golden_df,
        predictions_df,
        on=['query', 'product'],
        suffixes=('_golden', '_predicted')
    )

    if merged_df.empty:
        print("No matching query-product pairs found between golden and predictions.")
        return None

    # Extract labels
    y_true = merged_df['esciID_golden']
    y_pred = merged_df['esciID_predicted']

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    metrics = {
        'accuracy': accuracy,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }

    print(merged_df)    

    print("Evaluation Metrics:")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")

    return metrics

if __name__ == "__main__":
    calculate_metrics(limit=1000)
