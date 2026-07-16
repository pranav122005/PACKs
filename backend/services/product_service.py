from backend.database.connection import get_connection

def get_product(barcode):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(

        "SELECT * FROM products WHERE barcode=?",

        (barcode,)

    )

    product = cursor.fetchone()

    conn.close()

    return product