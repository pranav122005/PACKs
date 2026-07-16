from backend.database.connection import get_connection


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Products
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        barcode TEXT UNIQUE,
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

    # Ingredients
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingredients (
        ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ingredient_name TEXT UNIQUE,
        category TEXT,
        description TEXT
    )
    """)

    # Ingredient Aliases
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingredient_aliases (
        alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ingredient_id INTEGER,
        alias TEXT UNIQUE
    )
    """)

    # Ingredient Knowledge
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingredient_knowledge (
        knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ingredient_id INTEGER,
        purpose TEXT,
        health_effect TEXT,
        daily_limit TEXT,
        pregnancy TEXT,
        children TEXT,
        diabetes TEXT,
        kidney TEXT,
        heart TEXT,
        reference_links TEXT
    )
    """)

    # Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        weight REAL,
        height REAL,
        goal TEXT,
        disease TEXT,
        allergies TEXT
    )
    """)

    # Scan History
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        product_id INTEGER,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    print("Database Ready")