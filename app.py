import os
import re
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, auth
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

# Debug mode configuration: disabled by default in production, configurable for local dev
debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
app.config["DEBUG"] = debug_mode

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
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com https://apis.google.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob: https://*.googleusercontent.com; "
        "connect-src 'self' https://cdn.jsdelivr.net https://tessdata.projectnaptha.com https://*.googleapis.com https://identitytoolkit.googleapis.com https://securetoken.googleapis.com https://accounts.google.com; "
        "frame-src 'self' https://*.firebaseapp.com https://accounts.google.com;"
    )
    return response


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "Request payload exceeds 2MB limit"}), 413


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
# Firebase Admin & Firestore Management
# -----------------------------------

_firestore_db = None
_firestore_initialized = False
_firestore_init_error = None


def get_firestore_client():
    global _firestore_db, _firestore_initialized, _firestore_init_error
    if _firestore_initialized:
        return _firestore_db

    raw_creds = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not raw_creds or raw_creds.strip() in ("", "replace_with_your_firebase_service_account_json"):
        _firestore_init_error = "FIREBASE_SERVICE_ACCOUNT_JSON is not configured."
        return None

    try:
        raw_stripped = raw_creds.strip()
        if raw_stripped.startswith("{") and raw_stripped.endswith("}"):
            cred_dict = json.loads(raw_stripped)
            cred = credentials.Certificate(cred_dict)
        elif os.path.isfile(raw_stripped):
            cred = credentials.Certificate(raw_stripped)
        else:
            cred_dict = json.loads(raw_stripped)
            cred = credentials.Certificate(cred_dict)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        _firestore_db = firestore.client()
        _firestore_initialized = True
        _firestore_init_error = None
        return _firestore_db
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin / Firestore: {e}")
        _firestore_init_error = str(e)
        return None


def get_authenticated_user():
    """
    Extracts and validates Firebase ID token from Authorization header.
    Returns decoded token dict containing 'uid' and optional 'email', or None.
    """
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header.startswith("Bearer "):
        return None
    id_token = auth_header.split("Bearer ", 1)[1].strip()
    if not id_token:
        return None

    # Ensure Firebase Admin SDK is initialized before verifying
    get_firestore_client()

    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        logger.warning(f"ID token verification failed: {e}")
        return None


# -----------------------------------
# Routes: Web Interface & Public Config
# -----------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/firebase-config")
def firebase_config():
    """
    Returns public Firebase Web Client configuration.
    Safe for browser consumption; does NOT contain private service account keys.
    """
    return jsonify({
        "apiKey": os.getenv("FIREBASE_CLIENT_API_KEY", ""),
        "authDomain": os.getenv("FIREBASE_CLIENT_AUTH_DOMAIN", ""),
        "projectId": os.getenv("FIREBASE_CLIENT_PROJECT_ID", ""),
        "appId": os.getenv("FIREBASE_CLIENT_APP_ID", "")
    })


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

        groq_client = get_groq_client()
        if not groq_client:
            return jsonify({"error": "GROQ_API_KEY is not configured. Add your API key to your local .env file."}), 503

        # Assemble prompt grounded strictly in the OCR text
        user_message_content = f"PRODUCT NAME: {product_name}\n\nEXTRACTED LABEL TEXT:\n{text}"
        if instruction:
            user_message_content += f"\n\nADDITIONAL INSTRUCTION:\n{instruction}"

        user_message_content += (
            "\n\nPlease provide a clear, factual breakdown structured with these exact sections:\n"
            "PRODUCT: [Name]\n"
            "NUTRITION:\n- Calories: ...\n- Protein: ...\n- Carbohydrates: ...\n- Total sugar: ...\n"
            "- Fat: ...\n- Saturated fat: ...\n- Fiber: ...\n- Sodium/salt: ...\n"
            "INGREDIENTS:\n[Key ingredients bullet points]\n"
            "ADDITIVES:\n[Preservatives, sweeteners, emulsifiers bullet points or 'None detected']\n"
            "GOOD POINTS:\n[Positive nutritional highlights bullet points]\n"
            "THINGS TO WATCH:\n[Nutritional watchouts bullet points]\n"
            "QUICK VERDICT:\n[Concise summary verdict]"
        )

        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_NUTRITION_PROMPT},
                    {"role": "user", "content": user_message_content}
                ],
                temperature=0.1,
                timeout=GROQ_TIMEOUT_SECONDS
            )
        except AuthenticationError:
            logger.error("Groq authentication failed: invalid API key.")
            return jsonify({"error": "Groq API authentication failed. Verify GROQ_API_KEY."}), 502
        except RateLimitError:
            logger.error("Groq rate limit exceeded.")
            return jsonify({"error": "Groq AI rate limit reached. Please wait a moment and retry."}), 429
        except (APITimeoutError, APIConnectionError) as conn_err:
            logger.error(f"Groq connection or timeout error: {conn_err}")
            return jsonify({"error": "Groq AI service timed out. Please try again."}), 503
        except APIStatusError as status_err:
            logger.error(f"Groq API returned HTTP error {status_err.status_code}")
            return jsonify({"error": f"Groq AI service error ({status_err.status_code}). Please retry."}), 502
        except GroqError as groq_err:
            logger.error(f"Groq general error: {groq_err}")
            return jsonify({"error": "Groq AI processing failed. Please retry."}), 500

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            return jsonify({"error": "Groq returned an empty response. Please retry."}), 502

        analysis = response.choices[0].message.content.strip()

        # Check user authentication for scan persistence
        user = get_authenticated_user()
        db = get_firestore_client()
        scan_id = None
        created_at = datetime.now().isoformat(timespec="seconds")
        doc_data = {
            "product_name": product_name or "Scanned Product",
            "ocr_text": text,
            "ai_analysis": analysis,
            "created_at": created_at
        }

        if user and db:
            uid = user.get("uid")
            try:
                doc_ref = db.collection("users").document(uid).collection("scans").document()
                doc_ref.set(doc_data)
                scan_id = doc_ref.id
            except Exception as db_err:
                logger.error(f"Failed to persist scan to user Firestore collection: {db_err}")
                return jsonify({
                    "success": True,
                    "id": None,
                    "product_name": product_name,
                    "result": analysis,
                    "warning": "Analysis succeeded but history persistence encountered an error."
                })
        elif not user:
            # Unauthenticated users can analyse labels, but their scans do not enter persistent shared history
            return jsonify({
                "success": True,
                "id": None,
                "product_name": product_name,
                "result": analysis,
                "note": "Analysis completed. Sign in or continue as guest to save to history."
            })
        else:
            return jsonify({
                "success": True,
                "id": None,
                "product_name": product_name,
                "result": analysis,
                "warning": "Analysis succeeded but Firestore is not configured."
            })

        return jsonify({
            "success": True,
            "id": scan_id,
            "product_name": product_name,
            "result": analysis
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /api/analyze")
        return jsonify({"error": "An unexpected error occurred during analysis."}), 500


# -----------------------------------
# Routes: User-Isolated Scan History
# -----------------------------------

@app.route("/api/history")
def history():
    user = get_authenticated_user()
    if not user:
        return jsonify({"error": "Authentication required to view scan history."}), 401

    db = get_firestore_client()
    if not db:
        logger.warning("Firestore not configured in /api/history")
        return jsonify([])

    uid = user.get("uid")
    try:
        docs = db.collection("users").document(uid).collection("scans").stream()
        history_data = []
        for doc in docs:
            d = doc.to_dict() or {}
            history_data.append({
                "id": doc.id,
                "product_name": d.get("product_name", "Scanned Product"),
                "created_at": d.get("created_at", "")
            })
        history_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jsonify(history_data)
    except Exception as e:
        logger.error(f"Firestore error in /api/history for user {uid}: {e}")
        return jsonify({"error": "Failed to load scan history"}), 500


@app.route("/api/history/<scan_id>")
def get_scan(scan_id):
    user = get_authenticated_user()
    if not user:
        return jsonify({"error": "Authentication required to view scan."}), 401

    db = get_firestore_client()
    if not db:
        return jsonify({"error": "Firestore is not configured"}), 503

    uid = user.get("uid")
    try:
        doc = db.collection("users").document(uid).collection("scans").document(scan_id).get()
        if not doc.exists:
            return jsonify({"error": "Scan not found"}), 404
        d = doc.to_dict() or {}
        return jsonify({
            "id": doc.id,
            "product_name": d.get("product_name", "Scanned Product"),
            "ocr_text": d.get("ocr_text", ""),
            "ai_analysis": d.get("ai_analysis", ""),
            "created_at": d.get("created_at", "")
        })
    except Exception as e:
        logger.error(f"Firestore error in /api/history/{scan_id} for user {uid}: {e}")
        return jsonify({"error": "Failed to retrieve scan"}), 500


@app.route("/api/history/<scan_id>", methods=["DELETE"])
def delete_scan(scan_id):
    user = get_authenticated_user()
    if not user:
        return jsonify({"error": "Authentication required to delete scan."}), 401

    db = get_firestore_client()
    if not db:
        return jsonify({"error": "Firestore is not configured"}), 503

    uid = user.get("uid")
    try:
        doc_ref = db.collection("users").document(uid).collection("scans").document(scan_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Scan not found"}), 404
        doc_ref.delete()
        return jsonify({"success": True, "message": "Scan record deleted successfully."})
    except Exception as e:
        logger.error(f"Firestore error deleting scan {scan_id} for user {uid}: {e}")
        return jsonify({"error": "Failed to delete scan"}), 500


# -----------------------------------
# Routes: Health Check
# -----------------------------------

@app.route("/api/health")
def health():
    raw_creds = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    is_configured = bool(
        raw_creds
        and raw_creds.strip() != ""
        and raw_creds.strip() != "replace_with_your_firebase_service_account_json"
    )
    db = get_firestore_client()
    firestore_ok = False

    if db:
        try:
            list(db.collection("users").limit(1).stream())
            firestore_ok = True
        except Exception as e:
            logger.error(f"Firestore connectivity check failed: {e}")
            firestore_ok = False

    groq_ready = bool(
        os.getenv("GROQ_API_KEY")
        and os.getenv("GROQ_API_KEY").strip() != ""
        and os.getenv("GROQ_API_KEY") != "replace_with_your_groq_api_key"
    )

    if firestore_ok:
        db_status = "connected"
    elif is_configured:
        db_status = "error"
    else:
        db_status = "unconfigured"

    status_overall = "ok" if (firestore_ok or not is_configured) else "degraded"
    http_status = 503 if (is_configured and not firestore_ok) else 200

    return jsonify({
        "status": status_overall,
        "database": "Firestore",
        "database_status": db_status,
        "ocr": "Tesseract.js",
        "ai": "Groq",
        "groq_configured": groq_ready,
        "firestore_configured": is_configured
    }), http_status


# -----------------------------------
# Startup
# -----------------------------------

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    port = int(os.getenv("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )
