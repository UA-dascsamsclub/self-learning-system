import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import logging
from database.fetch_data import fetch_query_product_pairs
from models.cross_encoder import generate_esci_labels
from database.store_predictions import store_predictions_in_db

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/pipeline.yaml")

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def setup_logging(log_config):
    log_file = log_config['log_file']
    log_dir = os.path.dirname(log_file)

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=log_config['level'],
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized.")

def main():
    config = load_config()
    setup_logging(config['pipeline']['logging'])

    logging.info("Pipeline triggered.")

    # PIPELINE STEPS
    # 1. data ingestion from tbl_queryproducts
    logging.info("Fetching query-product pairs from database.")
    query_product_pairs = fetch_query_product_pairs()
    logging.info(f"Fetched {len(query_product_pairs)} query-product pairs.")

    # 2. model inference
    model_config = config['pipeline']['model']
    model_type = model_config['type']
    model_path = model_config['path']

    logging.info(f"Running inference using model {model_type}.")
    df_labeled = generate_esci_labels(model_path)
    logging.info("Inference complete.")

    # 3. store predictions to tbl_predictions
    # logging.info("Storing predictions in database.")
    # store_predictions_in_db(df_labeled, model_type)

    # printing results for now until further steps completed
    if df_labeled is not None:
        print(df_labeled.head())
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()