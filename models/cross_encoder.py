from sentence_transformers import CrossEncoder
from database.fetch_data import fetch_query_product_pairs

model_path = "models/model_ce"
model = CrossEncoder(model_path, num_labels=4, automodel_args={'ignore_mismatched_sizes': True})

scores = model.predict([('carrots', 'Fresh-Cut Vegetable Tray and Ranch Dressing, priced per pound'), 
                        ('carrots', 'Sweet Potato & Carrot Recipe, Grain Free Dry Dog Food , 23.5 lbs.'), 
                        ('carrots', 'Carrot Bar Cake, 38 oz.'),
                        ('carrots', 'Whole Carrots, 5 lbs.'),
                        ('carrots', 'Blue Buffalo Nudges Homestyle Chicken, Peas, and Carrots Natural Dog Treats, 40 oz.')])
print(scores)

def assign_esci_label(score):
    """Assigns ESCI labels based on the similarity score."""
    if score > 0.9:
        return 'E'
    elif score > 0.7:
        return 'S'
    elif score > 0.5:
        return 'C'
    else:
        return 'I'

def generate_esci_labels(model_path, limit=1000):
    """Generates ESCI labels for query-product pairs using the cross-encoder model."""
    df = fetch_query_product_pairs(limit)
    if df is None or df.empty:
        print("No data fetched.")
        return None
    
    model = CrossEncoder(model_path)
    pairs = list(zip(df['query'], df['product']))
    scores = model.predict(pairs)
    
    df['score'] = scores
    df['esci_label'] = df['score'].apply(assign_esci_label)
    
    return df

if __name__ == "__main__":
    df_labeled = generate_esci_labels(model_path)
    
    if df_labeled is not None:
        print(df_labeled.head())