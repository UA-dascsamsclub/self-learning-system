import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from models.bi_encoder import BiEncoderWithClassifier, BiEncoderConfig
from models.bi_encoder_train import train_biencoder, QueryProductDataset
from database.fetch_golden import fetch_golden
from database.store_model import insert_model

# Configuration
pretrained_model_path = "models/model_be/bi_encoder_model.pth"
finetuned_model_path = "models/model_be_finetuned/bi_encoder_model.pth"
batch_size = 16
epochs = 3
learning_rate = 1e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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

def finetune_biencoder(df_golden):
    if df_golden is None or df_golden.empty:
        print("No data provided for fine-tuning. Exiting.")
        return False

    queries = df_golden['query'].tolist()
    products = df_golden['product'].tolist()
    labels = df_golden['esciID'].tolist()

    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(labels), dtype=torch.long)

    dataset = QueryProductDataset(queries, products, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = load_or_initialize_be_model()

    train_biencoder(model, dataloader, num_epochs=epochs, learning_rate=learning_rate, save_path="models/model_be_finetuned/", fine_tune=True)

    model_id = insert_model("biencoder")
    print(f"Stored fine-tuned Bi-Encoder model with modelID {model_id}")

    print("Fine-tuning complete.")
    return True

if __name__ == "__main__":
    df_golden = fetch_golden(limit=1000)
    finetune_biencoder(df_golden)
