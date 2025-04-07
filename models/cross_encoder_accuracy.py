import os
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import CrossEncoder
from models.cross_encoder_inference import predict_labels as cross_encoder_predict
sys.path.append('../self-learning-system')
from database.fetch_holdout import fetch_holdout

def calculate_ce_metrics(df, model):
    """
    Computes precision, recall, and micro-F1 using a CrossEncoder model path and labeled holdout data.

    Args:
        df (pd.DataFrame): Holdout data including ground-truth 'esciID' labels.
        model (str): Path to a fine-tuned CrossEncoder model directory.

    Returns:
        pd.DataFrame: A one-row DataFrame with precision, recall, and micro-F1.
    """

    if not isinstance(model, str) or not os.path.exists(model):
        raise ValueError(f"Invalid model path provided: {model}")

    # Predict labels using model directory path
    prediction_df = cross_encoder_predict(df, model=model)

    if prediction_df.empty:
        print("No predictions generated.")
        return pd.DataFrame(columns=["precision", "recall", "microf1"])

    # Merge predictions back into the input DataFrame (ensure order aligns)
    df = df.reset_index(drop=True)
    prediction_df = prediction_df.reset_index(drop=True)
    df["esci_label_predicted"] = prediction_df["esci_label"]
    
    df = df.dropna(subset=["esci_label_predicted"])

    df["esciID"] = df["esciID"].astype(int)
    df["esci_label_predicted"] = df["esci_label_predicted"].astype(int)

    # Extract labels
    y_true = df['esciID']
    y_pred = df['esci_label_predicted']

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")

    print(df['esci_label_predicted'].value_counts())
    print(df['esciID'].value_counts())    

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

    model_path = "models/model_ce_trained/"
    df = fetch_holdout(limit=1000)

    if df is None or df.empty:
        print("No holdout data fetched.")
    else:
        ce_scores = calculate_ce_metrics(df, model=model_path)
        print(ce_scores.head())
