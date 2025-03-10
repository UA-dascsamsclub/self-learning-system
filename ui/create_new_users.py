import bcrypt
import psycopg2
import psycopg2.extras

def add_users_to_database():
    # Connect to your database
    conn = psycopg2.connect(
        host = "xxx", 
        database = "xxx",
        user = "xxx",
        password = "xxxx"
    )
    cur = conn.cursor()

    users = [
        ("username", "pass")
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

