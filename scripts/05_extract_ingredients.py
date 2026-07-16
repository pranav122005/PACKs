import sqlite3
from collections import Counter
import csv
from tqdm import tqdm

DATABASE_PATH = "database/packs.db"
OUTPUT_FILE = "datasets/processed/ingredients_raw.csv"

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

cursor.execute("""
SELECT ingredients_text
FROM products
WHERE ingredients_text IS NOT NULL
""")

rows = cursor.fetchall()

ingredient_counter = Counter()

for row in tqdm(rows):

    ingredients = row[0]

    if not ingredients:
        continue

    ingredients = ingredients.replace(";", ",")

    parts = ingredients.split(",")

    for ingredient in parts:

        ingredient = ingredient.strip()

        if len(ingredient) < 2:
            continue

        ingredient_counter[ingredient] += 1

connection.close()

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Ingredient",
        "Count"
    ])

    for ingredient, count in ingredient_counter.most_common():

        writer.writerow([
            ingredient,
            count
        ])

print()

print("Unique Ingredients :", len(ingredient_counter))

print("Saved to :", OUTPUT_FILE)