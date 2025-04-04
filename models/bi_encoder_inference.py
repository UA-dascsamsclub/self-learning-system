import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
import sys
sys.path.append('../self-learning-system')
from database.fetch_data import fetch_query_product_pairs

def predict_labels(df, model):
    """
    Runs inference on query-product pairs using the specified model directory.
    Returns a DataFrame with query, product, score, and esci_label columns.

    Args:
        df (pd.DataFrame): DataFrame with columns 'query' and 'product'.
        model (str): Path to the model directory containing tokenizer and weights.
    """

    if not isinstance(model, str) or not os.path.exists(model):
        raise ValueError(f"Invalid model directory: {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model)

    config = BiEncoderConfig.from_pretrained(model) if hasattr(BiEncoderConfig, 'from_pretrained') else BiEncoderConfig(
        encoder_name="sentence-transformers/all-distilroberta-v1",
        num_classes=4
    )

    # Load model and weights
    model_weights_path = os.path.join(model, "bi_encoder_model.pth")
    model_obj = BiEncoderWithClassifier(config)
    model_obj.load_state_dict(torch.load(model_weights_path, map_location=device))
    model_obj.to(device)
    model_obj.eval()

    if df is None or df.empty:
        print("No query-product pairs provided. Exiting inference.")
        return pd.DataFrame(columns=["query", "product", "score", "esci_label"])

    data = list(zip(df['query'], df['product']))
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch_queries = [item[0] for item in batch]
            batch_products = [item[1] for item in batch]

            inputs = tokenizer(
                list(batch_queries), list(batch_products),
                truncation=True, padding=True, max_length=256,
                return_tensors="pt"
            ).to(device)

            with autocast():
                outputs = model_obj(**inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_scores, predicted_classes = torch.max(probs, dim=1)

                predicted_classes = predicted_classes.cpu().tolist()
                max_scores = max_scores.cpu().tolist()

                for query, product, label, score in zip(batch_queries, batch_products, predicted_classes, max_scores):
                    results.append({
                        "query": query,
                        "product": product,
                        "score": score,
                        "esci_label": label
                    })

            torch.cuda.empty_cache()

    result_df = pd.DataFrame(results, columns=["query", "product", "score", "esci_label"])
    return result_df

if __name__ == "__main__":
    from database.fetch_data import fetch_query_product_pairs

    df = fetch_query_product_pairs(limit=1000)
    model_dir = "models/model_be/"

    predictions_df = predict_labels(df, model=model_dir)
    print("Predictions DataFrame:")
    print(predictions_df.head())
    
