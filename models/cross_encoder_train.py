import os
import torch
import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch.mps
torch.mps.empty_cache()
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

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

def prepare_data(dataset):
    samples = []
    for _, row in dataset.iterrows():
        query = row["query"]
        product = row["product_title"]
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

def train_crossencoder(model, dataset, num_epochs=3, learning_rate=1e-5, batch_size=16):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model_ce_trained")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Wrap dataset in a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)  # Accessing model's parameters

    num_training_steps = len(dataloader) * num_epochs

    # Training loop
    for epoch in range(num_epochs):
        model.model.train()  # Accessing the underlying model and setting it to train mode
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            sentences = batch
            labels = torch.tensor([sample.label for sample in batch], dtype=torch.long).to(device)

            # Extract query and product from InputExample objects
            query = [sample.texts[0] for sample in sentences]  # First element of texts is the query
            product = [sample.texts[1] for sample in sentences]  # Second element of texts is the product
            
            # Prepare inputs for the model
            inputs = model.tokenizer(
                query,
                product,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            # Forward pass to get logits
            outputs = model.model(**inputs)  # Accessing underlying model
            logits = outputs.logits

            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Backward pass + optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Save the trained model
    model.save(model_dir)
    print("CrossEncoder model saved in model_ce_trained/ directory!")

if __name__ == "__main__":
    # Load data from GitHub repo directly
    csv = '/Users/sarahlawlis/Documents/repos/self-learning-system/df_golden.csv'
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

    # Initialize CrossEncoder model (for multi-class classification)
    model = CrossEncoder(
        "models/model_ce", 
        num_labels=4,
        automodel_args={'ignore_mismatched_sizes': True}
    )

    # Move model to GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    train_crossencoder(model, dataset)
