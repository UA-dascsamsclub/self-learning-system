import os
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from models.bi_encoder_inference import predict_labels as bi_encoder_predict
from models.bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
import torch
from transformers import AutoTokenizer
sys.path.append('../self-learning-system')
from database.fetch_holdout import fetch_holdout

def calculate_be_metrics(df, model):
    """
    Computes precision, recall, and micro-F1 using a BiEncoder model path and labeled holdout data.

    Args:
        df (pd.DataFrame): Holdout data including ground-truth 'esciID' labels.
        model (str): Path to a fine-tuned BiEncoder model directory.

    Returns:
        pd.DataFrame: A one-row DataFrame with precision, recall, and micro-F1.
    """

    # Predict labels
    prediction_df = bi_encoder_predict(df, model)

    if prediction_df.empty:
        print("No predictions generated.")
        return pd.DataFrame(columns=["precision", "recall", "microf1"])

    # Assign predicted labels back to df (assumes original df is same order)
    df = df.reset_index(drop=True)
    prediction_df = prediction_df.reset_index(drop=True)
    df["esci_label_predicted"] = prediction_df["esci_label"]

    df = df.dropna(subset=["esci_label_predicted"])

    y_true = df['esciID']
    y_pred = df['esci_label_predicted']

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print("Evaluation Metrics:")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro)   : {recall:.4f}")
    print(f"F1 Score (micro) : {f1:.4f}")

    return pd.DataFrame([{
        "precision": precision,
        "recall": recall,
        "microf1": f1
    }])

if __name__ == "__main__":
    from database.fetch_holdout import fetch_holdout

    # Load holdout data and model
    df = fetch_holdout(limit=1000)
    model_path = "models/model_be_finetuned/"

    if df is None or df.empty:
        print("No holdout data fetched.")
    else:
        be_scores = calculate_be_metrics(df, model=model_path)
        print(be_scores)
