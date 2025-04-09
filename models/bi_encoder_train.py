import os
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, util
from models.cross_encoder_train import preprocess_text
from models.bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.mps
torch.mps.empty_cache()
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

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
    Ensures queries and products are lists of strings.
    """
    queries = [str(q) for q in queries]
    products = [str(p) for p in products]

    inputs = tokenizer(
        queries, products, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids")

def train_biencoder(model, dataloader, num_epochs=3, learning_rate=1e-5, save_path="models/model_be/", fine_tune=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    torch.mps.empty_cache()

    if fine_tune:
        print("🔹 Fine-tuning mode: Freezing early layers...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for queries, products, labels in progress_bar:
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids = preprocess(queries, products)
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            torch.mps.synchronize()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    torch.save(model.state_dict(), os.path.join(save_path, "bi_encoder_model.pth"))
    tokenizer.save_pretrained(save_path)
    print(f"Bi-Encoder model saved in {save_path} (Fine-tuning: {fine_tune})")

if __name__ == "__main__":
    import pandas as pd
    
    # Load data from GitHub repo directly
    csv = '/Users/sarahlawlis/Documents/repos/self-learning-system/df_golden.csv'
    
    # For local testing
    #csv = '/Users/thomasburns/Documents/Repos/esci-shopping-queries/data/df_golden_test.csv'
    
    try:
        df = pd.read_csv(csv)
        print(df.columns)
        print(f"Loaded dataset with {len(df)} records.")
    except Exception as e:
        print(f"Failed to load CSV from {csv}: {e}")
        exit(1)

    # Extract queries, products, and labels
    queries = df["query"].tolist()
    queries = [preprocess_text(q) for q in queries]
    products = df["product_title"].tolist()
    products = [preprocess_text(p) for p in products]

    # Encode labels
    esci_mapping = {"E": 0, "S": 1, "C": 2, "I": 3}
    df["encoded_labels"] = df["esci_label"].map(esci_mapping)
    labels = torch.tensor(df["encoded_labels"].tolist(), dtype=torch.long)

    # Print sample data to verify preprocessing
    #print(f"Queries: {queries[:5]}")
    #print(f"Products: {products[:5]}")

    sample_input = torch.rand(1, 128).to("mps")
    print(sample_input.device)

    # Initialize dataset and dataloader
    dataset = QueryProductDataset(queries, products, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create a config instance
    config = BiEncoderConfig()

    # Initialize the model with the config
    model = BiEncoderWithClassifier(config)

    # Move model to GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    train_biencoder(model, dataloader, num_epochs=3, learning_rate=1e-4, save_path="models/model_be/", fine_tune=False)