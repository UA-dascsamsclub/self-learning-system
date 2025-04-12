import bcrypt
import psycopg2
import psycopg2.extras

def add_users_to_database():
    """
    Adds multiple users to the database by hashing their passwords and 
    inserting their username and hashed password into the `tbl_analyst` table.

    Steps:
    1. Connect to the database using psycopg2.
    2. Iterate through a list of users (username, password) tuples.
    3. For each user:
       - Hash the password using bcrypt.
       - Insert the username and hashed password into the database.
    4. Commit the changes if all insertions are successful.
    5. If an error occurs, rollback the transaction to avoid partial inserts.
    6. Close the cursor and database connection at the end.

    """
    # Connect to your database
    conn = psycopg2.connect(
        host="insert host here",
        database="insert database here",
        user="insert personal username here",
        password="insert personal password here"
    )
    cur = conn.cursor()

    users = [
        ("insert-username", "insert-password"), 
        ("insert-username", "insert-password"), 
        ("insert-username", "insert-password"), 
        ("insert-username", "insert-password")
    ]
    try:
        for username, password in users:
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cur.execute(
                "INSERT INTO tbl_analyst (username, password_hash) VALUES (%s, %s)",
                (username, hashed.decode('utf-8'))
            )
        conn.commit()
        print(f"Successfully added {len(users)} users to the database")
        
    except Exception as e:
        conn.rollback()
        print(f"Error adding users: {e}")
        
    finally:
        cur.close()
        conn.close()

# Run the function
add_users_to_database()

