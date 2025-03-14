import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import sys
sys.path.append('../self-learning-system')
from database.fetch_data import fetch_query_product_pairs
import torch.mps
torch.mps.empty_cache()
import datetime
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

# Define model path once and reuse it
model_dir = "models/model_ce/"

# Initialize the cross-encoder model
model = CrossEncoder(
    model_dir,
    num_labels=4,
    automodel_args={'ignore_mismatched_sizes': True}
)

def predict_labels():
    """
    Runs inference on query-product pairs pulled from the database using a sentence-transformers CrossEncoder.
    Returns a DataFrame with query, product, score, and esci_label columns.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.model.to(device)

    df = fetch_query_product_pairs(limit=1000)

    if df is None or df.empty:
        print("No query-product pairs fetched. Exiting inference.")
        return pd.DataFrame(columns=["query", "product", "score", "esci_label"])

    data = list(zip(df['query'], df['product']))

    batch_size = 8
    results = []

    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Predicting"):
        batch = data[i:i + batch_size]

        # model.predict returns logits or probabilities depending on config
        # If multi-class, the highest score is usually the predicted class
        probs = model.predict(batch, convert_to_tensor=True)

        # If probs is a 2D tensor, it's multi-class scores
        probs = probs.cpu()

        # Get max score and predicted label index
        max_scores, predicted_classes = torch.max(probs, dim=1)

        for (query, product), label, score in zip(batch, predicted_classes.tolist(), max_scores.tolist()):
            results.append({
                "query": query,
                "product": product,
                "score": score,
                "esci_label": label
            })

    result_df = pd.DataFrame(results, columns=["query", "product", "score", "esci_label"])

    return result_df


if __name__ == "__main__":
    predictions_df = predict_labels()
    print("Predictions DataFrame:")
    print(predictions_df.head())

    time = datetime.datetime.now(datetime.timezone.utc) 
    predictions_df.to_excel(f'/Users/sarahlawlis/Desktop/preds_{time}.xlsx')
