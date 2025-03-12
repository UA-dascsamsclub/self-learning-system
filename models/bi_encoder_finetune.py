import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from tqdm import tqdm
from models.bi_encoder import BiEncoderWithClassifier
from models.bi_encoder_train import train_biencoder
from database.fetch_data import fetch_labeled_data  # (Update) Function to pull labeled training data

# Configuration
model_path = "model_be/bi_encoder_model.pth"  # Load previously saved model
batch_size = 16
epochs = 3
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = BiEncoderWithClassifier(model_name=None).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
tokenizer = AutoTokenizer.from_pretrained("model_be/bi_encoder_model.pth" )

def finetune_biencoder():
    """
    Finetunes the bi-encoder model using labeled training data.
    """
    # Fetch labeled training data
    queries, products, labels = fetch_labeled_data()
    
    # Create DataLoader
    dataset = list(zip(queries, products, labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # Training loop
    model.train_biencoder()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch_queries, batch_products, batch_labels = zip(*batch)
            
            # Tokenize inputs
            inputs = tokenizer(
                list(batch_queries), list(batch_products),
                truncation=True, padding=True, max_length=256,
                return_tensors="pt"
            ).to(device)
            
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            loop.set_postfix(loss=loss.item())
    
    # Save finetuned model
    model_path = "bi_encoder_finetuned.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Finetuned model saved to {model_path}")

if __name__ == "__main__":
    finetune_biencoder()
