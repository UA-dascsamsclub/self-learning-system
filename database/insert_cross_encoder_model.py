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

def insert_base_crossencoder_model():
    """Inserts a single base crossencoder model into tbl_models if it does not exist."""
    print("Connecting to database...")  # Debugging print
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            print("Checking if base crossencoder model exists...")  # Debugging print
            cur.execute("""
                SELECT "modelID" FROM tbl_models
                WHERE "modelclassID" = (SELECT "modelclassID" FROM tbl_modelclass WHERE "modelClass" = 'crossencoder')
                LIMIT 1
            """)
            if cur.fetchone():
                print("Base crossencoder model already exists.")
                return

            print("Inserting modelVersion into tbl_versionmodel...")  # Debugging print
            model_version = datetime.datetime.now(datetime.timezone.utc)
            cur.execute("""
                INSERT INTO tbl_versionmodel ("modelVersion")
                VALUES (%s)
                RETURNING "modelversionID"
            """, (model_version,))
            modelversion_id = cur.fetchone()[0]

            print("Inserting base crossencoder model into tbl_models...")  # Debugging print
            cur.execute("""
                INSERT INTO tbl_models ("modelversionID", "modelclassID")
                VALUES (
                    %s,
                    (SELECT "modelclassID" FROM tbl_modelclass WHERE "modelClass" = 'crossencoder')
                )
            """, (modelversion_id,))

            conn.commit()
            print(f"Inserted base crossencoder model with modelVersion {model_version}.")
if __name__ == "__main__":
    insert_base_crossencoder_model()