import sqlite3
import orjson
from tqdm import tqdm
import time

# -----------------------------
# CONFIGURATION
# -----------------------------

DATASET_PATH = "datasets/raw/openfoodfacts-products.jsonl"
DATABASE_PATH = "database/packs.db"

# Change this to None later if you want to import everything
LIMIT = 100000

# -----------------------------
# CONNECT DATABASE
# -----------------------------

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

# -----------------------------
# START TIMER
# -----------------------------

start_time = time.time()

imported = 0
skipped = 0

print("Starting Import...\n")

# -----------------------------
# READ DATASET
# -----------------------------

with open(DATASET_PATH, "rb") as file:

    for line in tqdm(file):

        if LIMIT and imported >= LIMIT:
            break

        try:

            product = orjson.loads(line)

            nutrition = (
                product
                .get("nutrition", {})
                .get("aggregated_set", {})
                .get("nutrients", {})
            )

            cursor.execute("""
            INSERT INTO products(

            barcode,
            product_name,
            brand,
            category,
            country,
            quantity,
            image_url,
            ingredients_text,
            calories,
            protein,
            fat,
            carbohydrates,
            sugars,
            salt,
            sodium

            )

            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)

            """,

            (

            product.get("code"),

            product.get("product_name"),

            product.get("brands"),

            product.get("categories"),

            product.get("countries"),

            product.get("quantity"),

            product.get("image_url"),

            product.get("ingredients_text"),

            nutrition.get("energy-kcal",{}).get("value"),

            nutrition.get("proteins",{}).get("value"),

            nutrition.get("fat",{}).get("value"),

            nutrition.get("carbohydrates",{}).get("value"),

            nutrition.get("sugars",{}).get("value"),

            nutrition.get("salt",{}).get("value"),

            nutrition.get("sodium",{}).get("value")

            )

            )

            imported += 1

            # Commit every 5000 rows
            if imported % 5000 == 0:
                connection.commit()

        except Exception:
            skipped += 1

# -----------------------------
# FINAL COMMIT
# -----------------------------

connection.commit()

connection.close()

# -----------------------------
# REPORT
# -----------------------------

end_time = time.time()

print("\n---------------------------")
print("IMPORT COMPLETED")
print("---------------------------")

print(f"Imported : {imported}")
print(f"Skipped  : {skipped}")
print(f"Time     : {round(end_time-start_time,2)} sec")