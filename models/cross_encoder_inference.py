import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import sys
sys.path.append('../self-learning-system')
from database.fetch_data import fetch_query_product_pairs
import torch.mps
from database.store_predictions import store_predictions_in_db
torch.mps.empty_cache()
import datetime
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

def predict_labels(df, model):
    """
    Runs inference on query-product pairs using a sentence-transformers CrossEncoder.
    Returns a DataFrame with query, product, score, and esci_label columns.

    Args:
        df (pd.DataFrame): DataFrame with columns 'query' and 'product'.
        model (str): Path to the CrossEncoder model directory.
    """

    if not isinstance(model, str) or not os.path.exists(model):
        raise ValueError(f"Invalid model directory: {model}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    cross_encoder = CrossEncoder(
        model,
        num_labels=4,
        automodel_args={'ignore_mismatched_sizes': True}
    )
    cross_encoder.model.to(device)

    if df is None or df.empty:
        print("No query-product pairs provided. Exiting inference.")
        return pd.DataFrame(columns=["query", "product", "score", "esci_label"])

    data = list(zip(df['query'], df['product']))

    batch_size = 16
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Predicting"):
        batch = data[i:i + batch_size]

        # Get logits or probabilities
        probs = cross_encoder.predict(batch, convert_to_tensor=True, apply_softmax=True)
        probs = probs.cpu()

        max_scores, predicted_classes = torch.max(probs, dim=1)

        for (query, product), label, score in zip(batch, predicted_classes.tolist(), max_scores.tolist()):
            results.append({
                "query": query,
                "product": product,
                "score": score,
                "esci_label": label
            })

    result_df = pd.DataFrame(results, columns=["query", "product", "score", "esci_label"])
    num_nans = result_df["esci_label"].isna().sum()
    if num_nans > 0:
        print(f"Warning: Dropping {num_nans} rows with NaN predictions.")
    result_df = result_df.dropna(subset=["esci_label"])
    return result_df

if __name__ == "__main__":
    model_path = "models/model_ce_trained/"
    df = fetch_query_product_pairs(limit=1000)

    predictions_df = predict_labels(df, model=model_path)
    print("Predictions DataFrame:")
    print(predictions_df.head())

    store_predictions_in_db(df=predictions_df, model_type='crossencoder')


