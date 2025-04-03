import sys
import os
import yaml
import logging
import pandas as pd
from database.fetch_data import fetch_query_product_pairs
from database.fetch_golden import fetch_golden
from models.bi_encoder_finetune import finetune_biencoder
from models.cross_encoder_finetune import finetune_crossencoder
from models.bi_encoder_inference import predict_labels as bi_encoder_predict
from models.cross_encoder_inference import predict_labels as cross_encoder_predict
from database.store_predictions import store_predictions_in_db
from database.store_model import insert_model
from database.fetch_holdout import fetch_holdout
from database.store_metrics import store_model_metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score
from models.bi_encoder_accuracy import calculate_be_metrics
from models.cross_encoder_accuracy import calculate_ce_metrics

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/pipeline.yaml")

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def setup_logging(log_config, base_dir):
    log_file = os.path.join(base_dir, log_config['log_file'])
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
    base_dir = os.path.abspath(config['pipeline']['base_dir'])  # Get absolute path

    setup_logging(config['pipeline']['logging'], base_dir)

    logging.info("Pipeline triggered.")

    # 1. Fetch 1000 new annotated query-product pairs
    logging.info("Fetching labeled query-product pairs from tbl_golden.")
    df_golden = fetch_golden(limit=1000)

    if df_golden is None or df_golden.empty:
        logging.error("No data fetched from tbl_golden. Exiting pipeline.")
        return

    logging.info(f"Fetched {len(df_golden)} rows for fine-tuning.")

    # 2. Fine-tune Bi-Encoder and Cross-Encoder
    logging.info("Starting fine-tuning for Bi-Encoder and Cross-Encoder.")

    be_result = finetune_biencoder(df_golden)
    ce_result = finetune_crossencoder(df_golden)

    if not (be_result and ce_result):
        logging.error("Fine-tuning failed. Stopping pipeline.")
        return
    
    # Capture model type and path
    be_model_type, be_model_path = be_result
    ce_model_type, ce_model_path = ce_result

    # Evaluate model accuracy using holdout data
    holdout_df = fetch_holdout(limit=100000)
    if holdout_df is None or holdout_df.empty:
        logging.error("No holdout data fetched. Exiting pipeline.")
        return
    
    # Make predictions using the holdout dataset and store them in a new df
    be_accuracy_df = bi_encoder_predict(holdout_df, model=be_result)
    ce_accuracy_df = cross_encoder_predict(holdout_df, model=ce_result)

    # Calculate metrics
    be_scores = calculate_be_metrics(model=be_result, df=be_accuracy_df)
    ce_scores = calculate_ce_metrics(model=ce_result, df=ce_accuracy_df)

    # Insert models into DB
    be_model_id = insert_model(be_model_type)
    ce_model_id = insert_model(ce_model_type)

    logging.info(f"Stored Bi-Encoder with modelID {be_model_id}")
    logging.info(f"Stored Cross-Encoder with modelID {ce_model_id}")

    # Push metrics to DB
    store_model_metrics(model_type=be_model_type, model_id=be_model_id, df=be_scores)
    store_model_metrics(model_type=ce_model_type,model_id=ce_model_id, df=ce_scores)

    # 3. Fetch query-product pairs for inference
    logging.info("Fetching query-product pairs from database.")
    query_product_pairs = fetch_query_product_pairs()

    if query_product_pairs is None or query_product_pairs.empty:
        logging.error("No query-product pairs fetched. Exiting pipeline.")
        return

    logging.info(f"Fetched {len(query_product_pairs)} query-product pairs.")

    # 4. Model inference using fine-tuned model
    model_config = config['pipeline']['model']
    model_type = model_config['type']
    
    logging.info(f"Running inference using model type: {model_type}.")

    if model_type == "crossencoder":
        df_labeled = cross_encoder_predict(query_product_pairs, model=ce_result)
    elif model_type == "biencoder":
        df_labeled = bi_encoder_predict(query_product_pairs, model=be_result)
    else:
        logging.error("Invalid model type in config. Exiting pipeline.")
        return

    if df_labeled is None or df_labeled.empty:
        logging.error("Model inference returned no results. Exiting pipeline.")
        return

    logging.info("Inference complete.")

    # 5. Store predictions in tbl_predictions
    logging.info("Storing predictions in database.")
    store_predictions_in_db(df_labeled, model_type)

    # 6. Printing results for now
    print(df_labeled.head())
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
