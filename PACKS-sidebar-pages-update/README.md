# PACKS — Food Label Intelligence

Minimal monochrome food-label scanner using browser OCR (Tesseract.js), Flask, Groq, and SQLite scan history.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your own Groq API key. Never commit `.env`.
4. Start the app: `python app.py`
5. Open `http://127.0.0.1:5000`.

The SQLite database is created locally on startup. OCR runs in the browser; the extracted text is sent to the Flask API for AI analysis.
