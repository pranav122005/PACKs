import os
import re
import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import (
    Groq,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    GroqError
)

# -----------------------------------
# Configuration & Setup
# -----------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger("packs")

app = Flask(__name__)

# Request limit: 2MB max payload
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

DATABASE = os.getenv("DATABASE_PATH", "database.db")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30.0"))

MAX_OCR_TEXT_LENGTH = 20000
MAX_PRODUCT_NAME_LENGTH = 120
MAX_INSTRUCTION_LENGTH = 4000


@app.after_request
def apply_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "replace_with_your_groq_api_key":
        return None
    return Groq(api_key=api_key.strip(), timeout=GROQ_TIMEOUT_SECONDS)


def sanitize_input(val, max_len=1000):
    if not isinstance(val, str):
        return ""
    # Strip null bytes and truncate
    cleaned = val.replace("\x00", "").strip()
    return cleaned[:max_len]


# -----------------------------------
# Database Management
# -----------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    try:
        with get_db() as conn:
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
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")


# -----------------------------------
# Routes: Web Interface
# -----------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------
# Routes: AI Product Analysis
# -----------------------------------

SYSTEM_NUTRITION_PROMPT = """You are NutriScan, an expert food nutrition knowledge assistant.
Analyze packaged food product labels factually and conservatively.

STRICT ACCURACY RULES:
1. Never invent or extrapolate numerical nutrition values or ingredients.
2. Only report nutrition values and ingredients explicitly supported by the OCR text.
3. Distinguish missing label values from zero (report missing values as 'Not listed').
4. Clearly distinguish facts from general nutrition education.
5. Do not diagnose medical conditions or prescribe health treatments.
6. Return an easy-to-understand, structured product analysis."""


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        text = sanitize_input(data.get("text", ""), max_len=MAX_OCR_TEXT_LENGTH)
        instruction = sanitize_input(data.get("instruction", ""), max_len=MAX_INSTRUCTION_LENGTH)
        product_name = sanitize_input(data.get("product_name", "Scanned Product"), max_len=MAX_PRODUCT_NAME_LENGTH)

        if not text:
            return jsonify({"error": "OCR text is empty"}), 400

        if len(text) < 3:
            return jsonify({"error": "OCR text is too short to analyze"}), 400

        client = get_groq_client()
        if client is None:
            return jsonify({
                "error": "GROQ_API_KEY is not configured. Add your API key to your local .env file."
            }), 503

        user_content = f"{instruction}\n\nOCR TEXT:\n{text}" if instruction else f"OCR TEXT:\n{text}"

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_NUTRITION_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=2048
            )
        except AuthenticationError:
            logger.warning("Groq API authentication failed. Check GROQ_API_KEY.")
            return jsonify({"error": "Invalid Groq API key. Please check your GROQ_API_KEY in .env."}), 502
        except RateLimitError:
            logger.warning("Groq API rate limit exceeded.")
            return jsonify({"error": "Groq API rate limit reached. Please wait a moment and try again."}), 429
        except (APITimeoutError, APIConnectionError) as net_err:
            logger.error(f"Groq API connection error: {net_err}")
            return jsonify({"error": "Groq API connection timed out. Please check your network and retry."}), 504
        except APIStatusError as status_err:
            logger.error(f"Groq API returned HTTP error {status_err.status_code}")
            return jsonify({"error": f"Groq AI service error ({status_err.status_code}). Please retry."}), 502
        except GroqError as groq_err:
            logger.error(f"Groq general error: {groq_err}")
            return jsonify({"error": "Groq AI processing failed. Please retry."}), 500

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            return jsonify({"error": "Groq returned an empty response. Please retry."}), 502

        analysis = response.choices[0].message.content.strip()

        # Save record to SQLite
        try:
            with get_db() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO scans (product_name, ocr_text, ai_analysis, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        product_name or "Scanned Product",
                        text,
                        analysis,
                        datetime.now().isoformat(timespec="seconds")
                    )
                )
                conn.commit()
                scan_id = cursor.lastrowid
        except sqlite3.Error as db_err:
            logger.error(f"Failed to persist scan to database: {db_err}")
            # Still return analysis result even if SQLite save encountered an issue
            return jsonify({
                "success": True,
                "id": None,
                "product_name": product_name,
                "result": analysis,
                "warning": "Analysis succeeded but history persistence encountered an error."
            })

        return jsonify({
            "success": True,
            "id": scan_id,
            "product_name": product_name,
            "result": analysis
        })

    except Exception as e:
        logger.exception("Unhandled error in /api/analyze")
        return jsonify({"error": "An unexpected error occurred during analysis."}), 500


# -----------------------------------
# Routes: Scan History
# -----------------------------------

@app.route("/api/history")
def history():
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT id, product_name, created_at
                FROM scans
                ORDER BY id DESC
                """
            ).fetchall()

        history_data = [
            {
                "id": row["id"],
                "product_name": row["product_name"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
        return jsonify(history_data)
    except sqlite3.Error as e:
        logger.error(f"Database error in /api/history: {e}")
        return jsonify({"error": "Failed to load scan history"}), 500


@app.route("/api/history/<int:scan_id>")
def get_scan(scan_id):
    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT id, product_name, ocr_text, ai_analysis, created_at
                FROM scans
                WHERE id = ?
                """,
                (scan_id,)
            ).fetchone()

        if not row:
            return jsonify({"error": "Scan not found"}), 404

        return jsonify({
            "id": row["id"],
            "product_name": row["product_name"],
            "ocr_text": row["ocr_text"],
            "ai_analysis": row["ai_analysis"],
            "created_at": row["created_at"]
        })
    except sqlite3.Error as e:
        logger.error(f"Database error in /api/history/{scan_id}: {e}")
        return jsonify({"error": "Failed to retrieve scan"}), 500


@app.route("/api/history/<int:scan_id>", methods=["DELETE"])
def delete_scan(scan_id):
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM scans WHERE id = ?",
                (scan_id,)
            )
            conn.commit()

        if cursor.rowcount == 0:
            return jsonify({"error": "Scan not found"}), 404

        return jsonify({"success": True})
    except sqlite3.Error as e:
        logger.error(f"Database error deleting scan {scan_id}: {e}")
        return jsonify({"error": "Failed to delete scan"}), 500


# -----------------------------------
# Routes: Health Check
# -----------------------------------

@app.route("/api/health")
def health():
    db_ok = True
    try:
        with get_db() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_ok = False

    groq_ready = bool(
        os.getenv("GROQ_API_KEY")
        and os.getenv("GROQ_API_KEY").strip() != ""
        and os.getenv("GROQ_API_KEY") != "replace_with_your_groq_api_key"
    )

    return jsonify({
        "status": "ok" if db_ok else "degraded",
        "database": "SQLite",
        "database_status": "connected" if db_ok else "error",
        "ocr": "Tesseract.js",
        "ai": "Groq",
        "groq_configured": groq_ready
    })


# -----------------------------------
# Startup
# -----------------------------------

init_db()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
