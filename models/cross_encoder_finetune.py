import os
import torch
from models.cross_encoder_train import train_crossencoder, QueryProductDataset
from database.fetch_golden import fetch_golden
from database.store_model import insert_model
from sentence_transformers import CrossEncoder, InputExample
from safetensors.torch import load_file

# Configuration
pretrained_model_path = "models/model_ce_trained/"
finetuned_model_path = "models/model_ce_finetuned/"
batch_size = 16
epochs = 3
learning_rate = 1e-5
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Directories for saving
os.makedirs("models/model_ce_finetuned", exist_ok=True)

def load_or_initialize_ce_model():
    """
    Load the finetuned model if it exists, otherwise load the pretrained model.
    """
    print(f"Pretrained model path: {pretrained_model_path}")
    print(f"Finetuned model path: {finetuned_model_path}")

    if os.path.exists(finetuned_model_path) and len(os.listdir(finetuned_model_path)) > 0:
        print(f"Loading finetuned model from {finetuned_model_path}")
        model = CrossEncoder(finetuned_model_path, num_labels=4, automodel_args={'ignore_mismatched_sizes': True})
        
        # Load the finetuned weights
        finetuned_model_file = os.path.join(finetuned_model_path, 'model.safetensors')
        if os.path.exists(finetuned_model_file):
            model.model.load_state_dict(load_file(finetuned_model_file))
    
    else:
        print(f"Finetuned model not found. Loading pretrained model from {pretrained_model_path}")
        assert os.path.exists(pretrained_model_path), f"Path does not exist: {pretrained_model_path}"
        model = CrossEncoder(pretrained_model_path, num_labels=4, automodel_args={'ignore_mismatched_sizes': True})  
        
        # Load the pretrained model weights from the specific file
        pretrained_model_file = os.path.join(pretrained_model_path, 'model.safetensors')
        if os.path.exists(pretrained_model_file):
            model.model.load_state_dict(load_file(pretrained_model_file))

    model.model.to(device) 
    return model

def finetune_crossencoder(df_golden):
    if df_golden is None or df_golden.empty:
        print("No data provided for fine-tuning. Exiting.")
        return False

    queries = df_golden['query'].tolist()
    products = df_golden['product'].tolist()
    labels = df_golden['esciID'].tolist()

    samples = [InputExample(texts=[query, product], label=label) for query, product, label in zip(queries, products, labels)]
    dataset = QueryProductDataset(samples)

    model = load_or_initialize_ce_model()

    train_crossencoder(model, dataset, num_epochs=3, learning_rate=1e-6, batch_size=16, save_path="models/model_ce_finetuned/", fine_tune=True)

    model_id = insert_model("crossencoder")
    print(f"Stored fine-tuned Cross-Encoder model with modelID {model_id}")

    print("Fine-tuning complete.")
    return True

if __name__ == "__main__":
    df_golden = fetch_golden()
    finetune_crossencoder(df_golden)
