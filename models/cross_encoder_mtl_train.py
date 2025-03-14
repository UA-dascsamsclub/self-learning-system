import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import InputExample
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class QueryProductDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
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
    input_examples = []
    for item in batch:
        query, product, label = item[0][0], item[0][1], item[1]
        input_example = InputExample(texts=[query, product], label=label)
        input_examples.append(input_example)
    return input_examples

class CrossEncoderESCI(nn.Module):
    def __init__(self, model_name, num_classes=4):
        """
        Cross-Encoder for ESCI classification with confidence scores.
        :param model_name: Transformer model name (e.g., 'roberta-base')
        :param num_classes: Number of ESCI label classes
        """
        super(CrossEncoderESCI, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.esci_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, query, product):
        """
        Forward pass through transformer encoder and classification head.
        :param query: List of query texts
        :param product: List of product texts
        :return: Logits for ESCI classification
        """
        inputs = self.tokenizer(
            query,
            product,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        encoder_outputs = self.encoder(**inputs)
        cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]

        esci_logits = self.esci_head(cls_embedding)
        return esci_logits

    def predict(self, query, product):
        self.eval()
        with torch.no_grad():
            esci_logits = self.forward(query, product)

            esci_probs = F.softmax(esci_logits, dim=1)

            esci_preds = torch.argmax(esci_probs, dim=1)
            esci_confidence = torch.max(esci_probs, dim=1).values

        return esci_preds.cpu().numpy(), esci_confidence.cpu().numpy()

def train_crossencoder(model, dataset, num_epochs=3, learning_rate=1e-5, batch_size=16):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model_ce_trained")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    esci_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            queries = [sample.texts[0] for sample in batch]
            products = [sample.texts[1] for sample in batch]
            labels = torch.tensor([sample.label for sample in batch], dtype=torch.long).to(device)

            esci_logits = model(queries, products)
            esci_loss = esci_loss_fn(esci_logits, labels)

            total_loss += esci_loss.item()

            optimizer.zero_grad()
            esci_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, "crossencoder_esci.pth"))
    print("CrossEncoder model saved in model_ce_trained/ directory!")

if __name__ == "__main__":
    csv = '/Users/sarahlawlis/Documents/repos/self-learning-system/df_golden.csv'
    try:
        df = pd.read_csv(csv)
        print(df.columns)
        print(f"Loaded dataset with {len(df)} records.")
    except Exception as e:
        print(f"Failed to load CSV from {csv}: {e}")
        exit(1)

    esci_mapping = {"E": 0, "S": 1, "C": 2, "I": 3}
    df["encoded_labels"] = df["esci_label"].map(esci_mapping)

    samples = prepare_data(df)
    dataset = QueryProductDataset(samples)

    model_name = "sentence-transformers/all-distilroberta-v1"
    model = CrossEncoderESCI(model_name, num_classes=4)
    model.to(device)

    train_crossencoder(model, dataset)
