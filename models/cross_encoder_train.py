import os
import torch
import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch.mps
import string
torch.mps.empty_cache()
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class QueryProductDataset(Dataset):
    def __init__(self, samples):
        """
        Initializes the dataset with query-product pairs and corresponding labels.
        :param samples: List of InputExample objects.
        """
        self.samples = samples
    
    def __len__(self):
        """Returns the total number of query-product pairs."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Return as a tuple with 'texts' key for CrossEncoder compatibility
        return sample.texts, sample.label

puncts = string.punctuation
def preprocess_text(text):
    """
    Removes punctuation from the input text and lowercases all remaining string.
    :param text: Input string.
    :return: String without punctuation and lowercased.
    """
    return ''.join([char for char in str(text).lower() if char not in puncts])

def prepare_data(dataset):
    samples = []
    for _, row in dataset.iterrows():
        query = preprocess_text(row["query"])
        product = preprocess_text(row["product_title"])
        label = int(row["encoded_labels"])
        samples.append(InputExample(texts=[query, product], label=label))
    return samples

def collate_fn(batch):
    # Convert batch to a list of InputExample objects
    input_examples = []
    for item in batch:
        query, product, label = item[0][0], item[0][1], item[1]  # Unpacking the tuple correctly
        input_example = InputExample(texts=[query, product], label=label)
        input_examples.append(input_example)

    return input_examples

def train_crossencoder(model, dataset, num_epochs=3, learning_rate=1e-5, batch_size=16, save_path="models/model_ce_trained/", fine_tune=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)

    if fine_tune:
        print("🔹 Fine-tuning mode: Freezing early layers...")
        for param in model.model.base_model.parameters():
            param.requires_grad = False

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            sentences = batch
            labels = torch.tensor([sample.label for sample in batch], dtype=torch.long).to(device)
            
            query = [sample.texts[0] for sample in sentences]
            product = [sample.texts[1] for sample in sentences]

            inputs = model.tokenizer(
                query,
                product,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            outputs = model.model(**inputs)
            logits = outputs.logits
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            torch.mps.synchronize()
            optimizer.step()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    model.model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)
    print(f"Model saved in {save_path} (Fine-tuning: {fine_tune})")

if __name__ == "__main__":
    # Load data from GitHub repo directly
    csv = '/Users/sarahlawlis/Documents/repos/self-learning-system/df_golden.csv'
    
    # For local testing, uncomment the line below
    #csv = '/Users/thomasburns/Documents/Repos/esci-shopping-queries/data/df_golden_test.csv'
    
    try:
        df = pd.read_csv(csv)
        print(f"Loaded dataset with {len(df)} records.")

    except Exception as e:
        print(f"Failed to load CSV from {csv}: {e}")
        exit(1)

    esci_mapping = {"E": 0, "S": 1, "C": 2, "I": 3}

    df["encoded_labels"] = df["esci_label"].map(esci_mapping)

    if df["encoded_labels"].isna().sum() > 0:
        print("Warning: Some 'esci_label' values could not be mapped. Check for unexpected labels.")
        df = df.dropna(subset=["encoded_labels"])
    # Prepare data
    samples = prepare_data(df)

    # Initialize dataset
    dataset = QueryProductDataset(samples)

    # # Preview the first few preprocessed samples
    # for sample in samples[:5]:
    #     print(f"Query: {sample.texts[0]}")
    #     print(f"Product: {sample.texts[1]}")
    #     print(f"Label: {sample.label}")
    #     print("---")

    # Initialize CrossEncoder model (for multi-class classification)
    model = CrossEncoder(
        "models/model_ce", 
        num_labels=4,
        automodel_args={'ignore_mismatched_sizes': True}
    )

    model.model.to(device)

    # Train the model
    train_crossencoder(model, dataset, num_epochs=3, learning_rate=1e-5, batch_size=16, save_path="models/model_ce_trained/", fine_tune=False)
