import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from bi_encoder import BiEncoderWithClassifier, BiEncoderConfig  # Ensure this matches your actual model import
import sys
sys.path.append('../self-learning-system')  # Add the parent folder to the path
from database.fetch_data import fetch_query_product_pairs  # Function to pull query-product pairs

# Load trained model and tokenizer
model_path = "models/model_be/bi_encoder_model.pth"  
tokenizer_path = "models/model_be/" 

config = BiEncoderConfig(encoder_name="sentence-transformers/all-distilroberta-v1", num_classes=4)
model = BiEncoderWithClassifier(config)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # Load tokenizer from the trained model

def predict_labels():
    """
    Runs inference on query-product pairs pulled from the database.
    Returns a list of predicted ESCI labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Fetch query-product pairs from the database
    df = fetch_query_product_pairs(limit=100)  

    if df is None or df.empty:  # Check if data is valid
        print("No query-product pairs fetched. Exiting inference.")
        return []
    
    # Convert DataFrame to list of tuples
    data = list(zip(df['query'], df['product']))  
    
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

    predicted_classes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):

            # Unpack the batch manually
            batch_queries = [item[0] for item in batch]  # All queries from the batch
            batch_products = [item[1] for item in batch]  # All products from the batch

            # Tokenize inputs
            inputs = tokenizer(
                list(batch_queries), list(batch_products),
                truncation=True, padding=True, max_length=256,
                return_tensors="pt"
            ).to(device)

            with autocast():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs, dim=1).tolist()
                predicted_classes.extend(predictions)
            
            torch.cuda.empty_cache()

    return predicted_classes

if __name__ == "__main__":
    predictions = predict_labels()
    print("Predicted ESCI Labels:", predictions)
