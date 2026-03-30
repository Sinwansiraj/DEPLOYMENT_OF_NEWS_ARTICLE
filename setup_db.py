import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname="postgres",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE DATABASE news_categorizer;")
print("Database created.")
cur.close()
conn.close()
