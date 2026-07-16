import sqlite3
import orjson
from tqdm import tqdm

DATASET_PATH = "datasets/raw/openfoodfacts-products.jsonl"
DATABASE_PATH = "database/packs.db"

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS products(

id INTEGER PRIMARY KEY AUTOINCREMENT,

barcode TEXT,

product_name TEXT,

brand TEXT,

category TEXT,

country TEXT,

quantity TEXT,

image_url TEXT,

ingredients_text TEXT,

calories REAL,

protein REAL,

fat REAL,

carbohydrates REAL,

sugars REAL,

salt REAL,

sodium REAL

)
""")

connection.commit()

print("Database Created.")