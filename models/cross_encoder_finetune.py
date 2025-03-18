import os
import torch
from models.cross_encoder_train import train_crossencoder, QueryProductDataset
from database.fetch_golden import fetch_golden
from sentence_transformers import CrossEncoder, InputExample
from safetensors.torch import load_file

# Configuration
pretrained_model_path = "models/model_ce_trained/"
finetuned_model_path = "models/model_ce_finetuned/"
batch_size = 16
epochs = 3
learning_rate = 1e-5
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

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
        model = CrossEncoder(finetuned_model_path, 
                             num_labels=4, 
                             automodel_args={'ignore_mismatched_sizes': True}).to(device)
        
        # Load the finetuned weights
        finetuned_model_file = os.path.join(finetuned_model_path, 'model.safetensors')  # Replace with the correct file
        if os.path.exists(finetuned_model_file):
            model.load_state_dict(load_file(finetuned_model_file))  # Use safetensors to load model
    
    else:
        print(f"Finetuned model not found. Loading pretrained model from {pretrained_model_path}")
        assert os.path.exists(pretrained_model_path), f"Path does not exist: {pretrained_model_path}"
        model = CrossEncoder(pretrained_model_path, 
                             num_labels=4, 
                             automodel_args={'ignore_mismatched_sizes': True}).to(device)  
        
        # Load the pretrained model weights from the specific file
        pretrained_model_file = os.path.join(pretrained_model_path, 'model.safetensors')  # Replace with the correct file
        if os.path.exists(pretrained_model_file):
            model.load_state_dict(load_file(pretrained_model_file))
    
    return model

def finetune_crossencoder():
    # Fetch labeled training data from fetch_golden
    df = fetch_golden(limit=1000)

    if df is None or df.empty:
        print("No data fetched. Exiting finetuning.")
        return

    # Extract queries, products, labels from DataFrame
    queries = df['query'].tolist()
    products = df['product'].tolist()
    labels = df['esciID'].tolist()  # Assuming "esciID" is the label column

    # Prepare samples for the CrossEncoder
    samples = []
    for query, product, label in zip(queries, products, labels):
        samples.append(InputExample(texts=[query, product], label=label))

    # Initialize Dataset and DataLoader
    dataset = QueryProductDataset(samples)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load or initialize the model
    model = load_or_initialize_ce_model()

    # Call the train_crossencoder function from cross_encoder_train.py
    train_crossencoder(model, dataset, num_epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    # Save finetuned model
    #torch.save(model.state_dict(), finetuned_model_path)
    model.save(finetuned_model_path)
    print(f"Finetuned model saved to {finetuned_model_path}")

if __name__ == "__main__":
    finetune_crossencoder()
