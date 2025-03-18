import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from models.bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
from models.bi_encoder_train import train_biencoder, QueryProductDataset
from database.fetch_golden import fetch_golden

# Configuration
pretrained_model_path = "models/model_be/bi_encoder_model.pth"
finetuned_model_path = "models/model_be_finetuned/bi_encoder_model.pth"
batch_size = 16
epochs = 3
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories for saving
os.makedirs("models/model_be_finetuned", exist_ok=True)

def load_or_initialize_be_model():
    """
    Load the finetuned model if it exists, otherwise load the pretrained model.
    """
    if os.path.exists(finetuned_model_path):
        print(f"Loading finetuned model from {finetuned_model_path}")
        model = BiEncoderWithClassifier(BiEncoderConfig()).to(device)
        model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    else:
        print(f"Finetuned model not found. Loading pretrained model from {pretrained_model_path}")
        model = BiEncoderWithClassifier(BiEncoderConfig()).to(device)
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    return model

def finetune_biencoder():
    # Fetch labeled training data from fetch_golden
    df = fetch_golden(limit=1000)

    if df is None or df.empty:
        print("No data fetched. Exiting finetuning.")
        return

    # Extract queries, products, labels from DataFrame
    queries = df['query'].tolist()
    products = df['product'].tolist()
    labels = df['esciID'].tolist()  # Assuming "esciID" is the label column

    # Initialize the label encoder
    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(labels), dtype=torch.long)

    # Create Dataset and DataLoader
    dataset = QueryProductDataset(queries, products, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load or initialize the model
    model = load_or_initialize_be_model()

    # Call the train_biencoder function from bi_encoder_train.py
    train_biencoder(model, dataloader, num_epochs=epochs, learning_rate=learning_rate)

    # Save finetuned model
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"Finetuned model saved to {finetuned_model_path}")

if __name__ == "__main__":
    finetune_biencoder()
