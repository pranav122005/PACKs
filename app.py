
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import os

# -----------------------------------
# Configuration
# -----------------------------------

load_dotenv()

app = Flask(__name__)

DATABASE = "database.db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# -----------------------------------
# Database
# -----------------------------------

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            product_name TEXT,

            ocr_text TEXT NOT NULL,

            ai_analysis TEXT,

            created_at TEXT NOT NULL

        )
    """)

    conn.commit()

    conn.close()


# -----------------------------------
# Home
# -----------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# -----------------------------------
# Analyze Product
# -----------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No data received"
            }), 400


        text = data.get(
            "text",
            ""
        ).strip()


        instruction = data.get(
            "instruction",
            ""
        )


        if not text:

            return jsonify({
                "error": "OCR text is empty"
            }), 400


        # -----------------------------------
        # Groq
        # -----------------------------------

        if client is None:
            return jsonify({"error": "GROQ_API_KEY is not configured. Add it to your local .env file."}), 503

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",

                    "content": """
You are NutriScan, a food product
nutrition knowledge assistant.

Analyze packaged food product labels.

Be accurate and conservative.

Never invent nutrition values.

Only report numerical nutrition
values that are actually present
in the OCR text.

Clearly distinguish label facts
from general nutrition education.

Do not diagnose medical conditions.

Return a useful, easy-to-understand
product analysis.
"""
                },

                {
                    "role": "user",

                    "content": f"""
{instruction}

OCR TEXT:

{text}
"""
                }

            ],

            temperature=0.2,

            max_tokens=2048

        )


        analysis = (
            response
            .choices[0]
            .message
            .content
        )


        # -----------------------------------
        # Try to determine product name
        # -----------------------------------

        product_name = data.get(
            "product_name",
            "Unknown Product"
        )


        # -----------------------------------
        # Save to SQLite
        # -----------------------------------

        conn = get_db()

        conn.execute(
            """
            INSERT INTO scans
            (
                product_name,
                ocr_text,
                ai_analysis,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,

            (
                product_name,
                text,
                analysis,
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )
        )

        conn.commit()

        scan_id = conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]

        conn.close()


        return jsonify({

            "success": True,

            "id": scan_id,

            "product_name":
                product_name,

            "result":
                analysis

        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------------
# Scan History
# -----------------------------------

@app.route("/api/history")
def history():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT
            id,
            product_name,
            created_at
        FROM scans
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()


    history_data = []

    for row in rows:

        history_data.append({

            "id": row["id"],

            "product_name":
                row["product_name"],

            "created_at":
                row["created_at"]

        })


    return jsonify(history_data)


# -----------------------------------
# Get Single Scan
# -----------------------------------

@app.route("/api/history/<int:scan_id>")
def get_scan(scan_id):

    conn = get_db()

    row = conn.execute(
        """
        SELECT *
        FROM scans
        WHERE id = ?
        """,
        (scan_id,)
    ).fetchone()

    conn.close()


    if not row:

        return jsonify({
            "error": "Scan not found"
        }), 404


    return jsonify({

        "id": row["id"],

        "product_name":
            row["product_name"],

        "ocr_text":
            row["ocr_text"],

        "ai_analysis":
            row["ai_analysis"],

        "created_at":
            row["created_at"]

    })


# -----------------------------------
# Delete Scan
# -----------------------------------

@app.route(
    "/api/history/<int:scan_id>",
    methods=["DELETE"]
)
def delete_scan(scan_id):

    conn = get_db()

    cursor = conn.execute(
        """
        DELETE FROM scans
        WHERE id = ?
        """,
        (scan_id,)
    )

    conn.commit()

    conn.close()


    if cursor.rowcount == 0:

        return jsonify({
            "error": "Scan not found"
        }), 404


    return jsonify({
        "success": True
    })


# -----------------------------------
# Health Check
# -----------------------------------

@app.route("/api/health")
def health():

    return jsonify({

        "status": "ok",

        "database": "SQLite",

        "ocr": "Tesseract.js",

        "ai": "Groq"

    })


# -----------------------------------
# Start
# -----------------------------------

if __name__ == "__main__":

    init_db()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
