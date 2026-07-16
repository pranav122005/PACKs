import sqlite3

conn = sqlite3.connect("backend/database/ingredient_master.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS ingredients(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT UNIQUE,

    aliases TEXT,

    category TEXT,

    purpose TEXT,

    health_score INTEGER,

    risk_level TEXT,

    description TEXT,

    daily_limit TEXT,

    pregnancy TEXT,

    children TEXT,

    diabetes TEXT,

    kidney TEXT,

    heart TEXT,

    approved_fssai TEXT,

    approved_fda TEXT,

    approved_efsa TEXT
)
""")

conn.commit()
conn.close()

print("Ingredient Master Database Created Successfully!")