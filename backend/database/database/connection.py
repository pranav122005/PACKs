import sqlite3


DATABASE = "database/packs.db"


def get_connection():
    return sqlite3.connect(DATABASE)