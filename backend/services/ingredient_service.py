import sqlite3

DB = "backend/database/ingredient_master.db"


def get_ingredient(name):

    conn = sqlite3.connect(DB)

    cursor = conn.cursor()

    cursor.execute("""

    SELECT *

    FROM ingredients

    WHERE LOWER(name)=LOWER(?)

    """,(name,))

    result = cursor.fetchone()

    conn.close()

    return result