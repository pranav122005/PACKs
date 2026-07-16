import json
import sqlite3

JSONL_FILE = "datasets/raw/openfoodfacts-products.jsonl"
DB_FILE = "database/packs.db"

LIMIT = 500000

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

count = 0

with open(JSONL_FILE, "r", encoding="utf-8") as f:

    for line in f:

        try:
            product = json.loads(line)
        except:
            continue

        barcode = product.get("code")

        if not barcode:
            continue

        cursor.execute("""

        INSERT OR REPLACE INTO products(

            barcode,

            product_name,

            brand,

            ingredients_text,

            calories,

            protein,

            fat,

            carbohydrates,

            sugars,

            salt,

            sodium,

            image_url,

            category,

            country,

            quantity

        )

        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)

        """,

        (

            barcode,

            product.get("product_name"),

            product.get("brands"),

            product.get("ingredients_text"),

            product.get("nutriments",{}).get("energy-kcal_100g"),

            product.get("nutriments",{}).get("proteins_100g"),

            product.get("nutriments",{}).get("fat_100g"),

            product.get("nutriments",{}).get("carbohydrates_100g"),

            product.get("nutriments",{}).get("sugars_100g"),

            product.get("nutriments",{}).get("salt_100g"),

            product.get("nutriments",{}).get("sodium_100g"),

            product.get("image_url"),

            product.get("categories"),

            product.get("countries"),

            product.get("quantity")

        ))

        count += 1

        if count % 5000 == 0:

            conn.commit()

            print(f"Imported {count}")

        if count >= LIMIT:
            break

conn.commit()

conn.close()

print("DONE")