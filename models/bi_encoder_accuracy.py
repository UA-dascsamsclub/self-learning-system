import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from bi_encoder_inference import predict_labels as bi_encoder_predict
from bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
import torch
from transformers import AutoTokenizer
sys.path.append('../self-learning-system')
from database.fetch_holdout import fetch_holdout

def calculate_be_metrics(df, model, tokenizer):
    """
    Fetches the holdout labels and model predictions from holdout data from the database,
    and computes accuracy, recall (micro), and F1 score (micro).
    
    Args:
        limit (int): The number of records to fetch from the database.
        
    Returns:
        dict: Dictionary containing accuracy, recall_micro, and f1_micro scores.
    """

    # Make predictions using the holdout dataset and store them in a new column
    df["esci_label_predicted"] = bi_encoder_predict(df, model, tokenizer)['esci_label']
    print(f"Number of Nulls:  {df['esci_label_predicted'].isnull().sum()}")

    if df["esci_label_predicted"].isnull().all():
        print("No predictions generated.")
        return None

    print(df['esci_label_predicted'].value_counts())
    print(df['esciID'].value_counts()) 

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

    print("Evaluation Metrics:")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")

    return metrics

if __name__ == "__main__":
    # Fetch holdout data 
    df = fetch_holdout(limit=1000)
    if df is None or df.empty:
        print("No holdout data fetched.")
    
    # Define model path 
    model_dir = "models/model_be/"
    model_checkpoint = "models/model_be_finetuned/"

    # Get the model weights path
    model_weights_path = os.path.join(model_checkpoint, "bi_encoder_model.pth")

    # Load tokenizer from the same directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Initialize config from model directory (if saved), else hardcode or pass manually
    # You can adjust this if you have a config.json in the directory
    config = BiEncoderConfig.from_pretrained(model_dir) if hasattr(BiEncoderConfig, 'from_pretrained') else BiEncoderConfig(
        encoder_name="sentence-transformers/all-distilroberta-v1",
        num_classes=4
    )

    # Initialize and load the model weights
    model = BiEncoderWithClassifier(config)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device("cpu")))
    model.eval()

    be_scores = calculate_be_metrics(df, model, tokenizer)
    print(be_scores)
