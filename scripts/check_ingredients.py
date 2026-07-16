import sqlite3

conn = sqlite3.connect("backend/database/ingredient_master.db")

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM ingredients")

print(cursor.fetchone())

cursor.execute("""
SELECT name
FROM ingredients
LIMIT 20
""")

for row in cursor.fetchall():
    print(row)

conn.close()