import pandas as pd
import sqlite3

df = pd.read_csv("datasets/knowledge/ingredient_knowledge.csv")

conn = sqlite3.connect("backend/database/ingredient_master.db")

df.to_sql(
    "ingredients",
    conn,
    if_exists="append",
    index=False
)

conn.close()

print("Ingredient Knowledge Imported!")