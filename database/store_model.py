import psycopg2
import datetime
from database.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def insert_model_version():
    """Inserts a new modelVersion and returns the corresponding modelversionID."""
    model_version = datetime.datetime.now(datetime.timezone.utc) 

    with connect_to_db() as conn:
        with conn.cursor() as cur:
            insert_query = """
            INSERT INTO tbl_versionmodel (modelVersion)
            VALUES (%s)
            RETURNING modelversionID
            """
            cur.execute(insert_query, (model_version,))
            modelversion_id = cur.fetchone()[0]
            conn.commit()
            return modelversion_id

def get_modelclass_id(model_type):
    """Retrieves the modelclassID from tbl_modelclass based on model type."""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            query = """
            SELECT modelclassID FROM tbl_modelclass WHERE modelClass = %s
            """
            cur.execute(query, (model_type,))
            result = cur.fetchone()
            return result[0] if result else None

def insert_model(model_type):
    """Handles inserting a new model into tbl_models after fine-tuning is completed."""
    modelversion_id = insert_model_version()
    if modelversion_id is None:
        print("Error: Failed to insert modelVersion.")
        return None

    modelclass_id = get_modelclass_id(model_type)
    if modelclass_id is None:
        print(f"Error: Model class '{model_type}' not found in tbl_modelclass.")
        return None

    with connect_to_db() as conn:
        with conn.cursor() as cur:
            insert_query = """
            INSERT INTO tbl_models (modelversionID, modelclassID)
            VALUES (%s, %s)
            RETURNING modelID
            """
            cur.execute(insert_query, (modelversion_id, modelclass_id))
            model_id = cur.fetchone()[0]
            conn.commit()
            print(f"New model inserted with modelID = {model_id}, modelclassID = {modelclass_id}, modelversionID = {modelversion_id}")
            return model_id
