import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, util
from bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
from transformers import AutoTokenizer

# Load tokenizer for the encoder model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")

# Dataset class to handle query-product pairs and labels
class QueryProductDataset(Dataset):
    def __init__(self, queries, products, labels):
        """
        Initializes the dataset with query-product pairs and corresponding labels.
        :param queries: List of query texts.
        :param products: List of product texts.
        :param labels: Tensor of label indices (ESCI classification).
        """
        self.queries = queries
        self.products = products
        self.labels = labels
    
    def __len__(self):
        """Returns the total number of query-product pairs."""
        return len(self.queries)
    
    def __getitem__(self, idx):
        """Retrieves the query, product, and label for a given index."""
        return self.queries[idx], self.products[idx], self.labels[idx]

def preprocess(queries, products, max_length=128):
    """
    Tokenizes and prepares input tensors for the bi-encoder model.
    """
    inputs = tokenizer(queries, products, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids")

def train_biencoder(model, dataloader, num_epochs=5, learning_rate=1e-4):
    """
    Trains the Bi-Encoder model for multi-class classification.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model_be")
    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for queries, products, labels in dataloader:
            optimizer.zero_grad()

            # 🔹 Tokenize the input before passing to the model
            input_ids, attention_mask, token_type_ids = preprocess(queries, products)
            
            # Move inputs and labels to the same device as model
            input_ids, attention_mask, labels = input_ids.to(model.device), attention_mask.to(model.device), labels.to(model.device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(model_dir, "bi_encoder_model.pth"))
    print("Model saved in model_be/ directory!")

    # Save the tokenizer
    tokenizer.save_pretrained(model_dir)
    print("Tokenizer saved in model_be/ directory!")

if __name__ == "__main__":
    import pandas as pd
    
    # Load data from GitHub repo directly
    csv_url = "https://raw.githubusercontent.com/sarahlawlis/esci-shopping-queries/main/data/df_golden.csv"    

    try:
        df = pd.read_csv(csv_url)
        print(f"Loaded dataset with {len(df)} records.")
    except Exception as e:
        print(f"Failed to load CSV from {csv_url}: {e}")
        exit(1)

    # Extract queries, products, and labels
    queries = df["query"].tolist()
    products = df["product_title"].tolist()
    labels = torch.tensor(df["encoded_labels"].tolist(), dtype=torch.long)

    # Initialize dataset and dataloader
    dataset = QueryProductDataset(queries, products, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create a config instance
    config = BiEncoderConfig()

    # Initialize the model with the config
    model = BiEncoderWithClassifier(config)

    # Move model to GPU if available
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Train the model
    train_biencoder(model, dataloader)