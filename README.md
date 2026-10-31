<div align="center">

# PACKS – AI-Powered Packaged Food Analyzer

**Instant nutritional transparency from food labels using client-side OCR and fast LLM intelligence.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Groq Cloud](https://img.shields.io/badge/AI-Groq%20Cloud-F55036?style=flat-square&logo=fastapi&logoColor=white)](https://groq.com/)
[![Tesseract.js](https://img.shields.io/badge/OCR-Tesseract.js%20v5-5C4EE5?style=flat-square)](https://tesseract.projectnaptha.com/)
[![Database](https://img.shields.io/badge/Database-Firebase%20Firestore-FFA611?style=flat-square&logo=firebase&logoColor=white)](https://firebase.google.com/docs/firestore)
[![Auth](https://img.shields.io/badge/Auth-Firebase%20Auth-FFCA28?style=flat-square&logo=firebase&logoColor=black)](https://firebase.google.com/docs/auth)
[![WSGI](https://img.shields.io/badge/Server-Gunicorn-499848?style=flat-square&logo=gunicorn&logoColor=white)](https://gunicorn.org/)
[![Deploy](https://img.shields.io/badge/Deploy-Render%20Free-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey?style=flat-square)](#license)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Data Flow](#architecture--data-flow)
- [Application Workflow](#application-workflow)
- [Technology Stack](#technology-stack)
- [Project Directory Structure](#project-directory-structure)
- [Firebase Setup (Auth & Firestore)](#firebase-setup-auth--firestore)
- [Prerequisites & Installation](#prerequisites--installation)
  - [Windows Installation](#windows-installation)
  - [macOS & Linux Installation](#macos--linux-installation)
- [Environment Configuration](#environment-configuration)
- [Running Locally](#running-locally)
- [API Endpoint Documentation](#api-endpoint-documentation)
- [Database Schema & Data Persistence](#database-schema--data-persistence)
- [Production Deployment (Render Free + Firestore)](#production-deployment-render-free--firestore)
  - [Deploying to Render via Blueprint](#deploying-to-render-via-blueprint)
  - [Manual Web Service Setup](#manual-web-service-setup)
  - [Free-Tier Architecture & Quotas](#free-tier-architecture--quotas)
- [Interface & Screenshots](#interface--screenshots)
- [Security & Privacy Considerations](#security--privacy-considerations)
- [Current Limitations & Roadmap](#current-limitations--roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Modern packaged food labels are often deliberately confusing: marketing claims obscure high sugars, sodium quantities are obfuscated across arbitrary serving sizes, and long lists of industrial additives (preservatives, stabilizers, synthetic colors, artificial sweeteners) remain impenetrable to consumers.

**PACKS** solves this with a **100% free-tier cloud architecture**:
1. **Privacy-Preserving OCR**: Food label images are processed directly in the user's browser using [Tesseract.js v5](https://tesseract.projectnaptha.com/) Web Workers. Raw images never leave the device.
2. **Structured Nutrition Breakdown**: The extracted label text is sent to a secured [Flask](https://flask.palletsprojects.com/) API powered by [Groq Cloud](https://groq.com/) LLM inference (`openai/gpt-oss-20b`). The analysis is parsed into a rigorous 7-section breakdown.
3. **Daily Nutrition Tracking**: Users can log analyzed items or manual food entries to monitor daily calories and macronutrient intake (Protein, Carbs, Fat) against a customizable calorie target via browser `localStorage`.
4. **Multi-Tenant Cloud History (Firebase Auth & Firestore)**: Scans are isolated per user under `users/{uid}/scans` in Google Cloud Firestore. Automatic Anonymous authentication gives users an instant private workspace without registration, while optional Email/Password authentication enables permanent account linking and cross-device synchronization. Unauthenticated scans run in guest privacy mode without entering cloud storage.

---

## Key Features

<details open>
<summary><strong>1. In-Browser OCR Extraction (Tesseract.js v5)</strong></summary>

- **Client-Side Text Recognition**: OCR runs on client hardware using WebAssembly and Web Workers.
- **Zero Raw Image Uploads**: No image files are transferred or stored on the backend, minimizing network overhead and guaranteeing image privacy.
- **Real-Time Progress & Inspection**: Visual progress bar tracks OCR stages (loading core, initializing engine, recognizing text).
- **Editable Transcript**: Extracted text can be corrected or pruned before triggering AI analysis.
</details>

<details open>
<summary><strong>2. Grounded 7-Section Nutrition Intelligence</strong></summary>

PACKS strictly enforces structured AI evaluation to eliminate hallucinations:
- **Product Name**: Extracted brand and product name.
- **Nutrition Facts**: Key macros (Calories, Protein, Carbohydrates, Total Sugar, Fat, Saturated Fat, Fiber, Sodium/Salt).
- **Ingredients Breakdown**: Plain-language explanations of primary ingredients.
- **Additives & Preservatives**: Detection of artificial sweeteners, colors, and emulsifiers.
- **Good Points**: Positive nutritional highlights (e.g. low saturated fat, high fiber).
- **Things to Watch**: Watchouts (e.g. excessive sodium, high added sugars).
- **Quick Verdict**: Objective, conservative summary verdict.
</details>

<details>
<summary><strong>3. Daily Dietary Log & Macro Tracker</strong></summary>

- **Direct Log Integration**: Add foods directly from an analysis report into the daily log with one click.
- **Manual Entry Support**: Log external meals with customizable portions, calories, and macronutrient values.
- **Date Navigation**: Filter logs by specific dates with per-day macro aggregation.
- **CRUD Controls**: Edit quantities or delete individual food records with immediate recalculation.
- **Client Storage**: Log entries persist locally on the user's browser via `localStorage`.
</details>

<details>
<summary><strong>4. Dynamic Nutrition Dashboard</strong></summary>

- **Target Tracking**: Real-time progress bar tracking consumed calories against a customizable daily calorie goal (default: 2000 kcal).
- **Macronutrient Aggregation**: Live calculation of total daily protein (g), carbohydrates (g), and fat (g).
- **Recent Activity Feed**: Chronological list of foods consumed today with timestamps and energy values.
</details>

<details>
<summary><strong>5. Context-Aware AI Nutrition Assistant</strong></summary>

- **Grounded Chat**: Ask questions about ingredients, dietary suitability, or allergen warnings.
- **Active Label Context**: Automatically incorporates active OCR text and product metadata into queries.
- **Defensive UI**: Disables input during query processing to prevent duplicate submissions.
</details>

<details>
<summary><strong>6. User-Isolated Cloud History (Firebase Auth & Firestore)</strong></summary>

- **Firebase Anonymous Authentication**: New visitors instantly receive a private cryptographic session and isolated Firestore workspace without signup barriers.
- **Optional Email/Password & Account Linking**: Link anonymous guest sessions to permanent email credentials to preserve history across devices and browsers.
- **Subcollection Data Isolation**: Scans are stored strictly in `users/{uid}/scans/{scan_id}`. Flask verifies Firebase ID tokens (`auth.verify_id_token`) on every history request, preventing cross-tenant access.
- **Unauthenticated Privacy Mode**: Unauthenticated scans execute AI analysis smoothly and return `id: null` without writing data to the cloud database.
- **Server-Side Firebase Admin**: Service account credentials never reach the browser; the client only receives safe public client configuration via `/api/firebase-config`.
- **Persistent Across Restarts**: Overcomes Render Free container restarts; scan history remains permanently stored in Firestore on the free Spark plan.
</details>

<details>
<summary><strong>7. Premium Monochrome Interface</strong></summary>

- **Design Philosophy**: Deep black background (`#000000`), graphite panels (`#111111`, `#1c1c1c`), high-contrast white typography, and subtle borders (`#282828`).
- **Responsive Navigation**: Full desktop sidebar with quick history list and collapsible mobile bottom navigation bar.
- **Zero Heavy Frameworks**: Built entirely with vanilla HTML5, CSS3, and modern JavaScript for maximum speed and zero build-step overhead.
</details>

---

## Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Client["Client Browser (Vanilla JS / SPA)"]
        Upload["File Dropzone / Camera"]
        OCR["Tesseract.js v5 Worker"]
        Editor["Editable Text Field"]
        AuthClient["Firebase Auth Client (Anon / Email)"]
        UI_Report["7-Section Report View"]
        UI_Dash["Nutrition Dashboard"]
        UI_Log["Daily Log (localStorage)"]
        UI_Chat["AI Assistant Chat"]
    end

    subgraph Backend["Flask 3.0 on Render Free (app.py)"]
        SecHeaders["Security Headers (CSP, FrameGuard, NoSniff)"]
        TokenVerifier["auth.verify_id_token()"]
        ConfigRoute["GET /api/firebase-config"]
        AnalyzeRoute["POST /api/analyze"]
        HistoryRoute["GET/DELETE /api/history"]
        HealthRoute["GET /api/health"]
    end

    subgraph AI["Groq Cloud"]
        GroqLLM["openai/gpt-oss-20b Inference"]
    end

    subgraph FirebaseCloud["Google Cloud / Firebase Spark Plan"]
        FirebaseAuth["Firebase Authentication"]
        Firestore["users/{uid}/scans/{scan_id}"]
    end

    AuthClient <-->|Token Exchange & State| FirebaseAuth
    ConfigRoute -->|Public Client Config| AuthClient
    Upload -->|Local File Buffer| OCR
    OCR -->|Extracted Text| Editor
    Editor -->|Payload + Bearer Token| SecHeaders
    SecHeaders --> TokenVerifier
    TokenVerifier --> AnalyzeRoute
    TokenVerifier --> HistoryRoute
    AnalyzeRoute -->|Prompt & Label Context| GroqLLM
    GroqLLM -->|Structured 7-Section Text| AnalyzeRoute
    AnalyzeRoute -->|Save to user subcollection (if authed)| Firestore
    AnalyzeRoute -->|JSON Result + Doc ID| UI_Report
    UI_Report -->|Add to Daily Log| UI_Log
    UI_Log -->|Aggregate Metrics| UI_Dash
    UI_Chat -->|Contextual Question| AnalyzeRoute
    HistoryRoute <-->|Isolated Query / Delete| Firestore
    HealthRoute -->|Probe Limit 1 Query| Firestore
```

---

## Application Workflow

1. **Upload & Extract**: Select or drag-and-drop a packaged food label image (`PNG`, `JPG`, `WEBP`). Tesseract.js recognizes label text directly in the browser.
2. **Review OCR**: The extracted text appears in an editable workspace. Correct any scanning artifacts or unreadable words.
3. **Analyze**: Click **Analyse**. The Flask API routes the text to Groq Cloud with a conservative, hallucination-resistant nutrition prompt.
4. **Structured Report**: PACKS parses the AI output into the 7 standard report sections: Product, Nutrition, Ingredients, Additives, Good Points, Things to Watch, and Quick Verdict.
5. **Track Diet**: Click **+ Add to Daily Log** to import the product's macros into your daily diary and monitor progress toward your calorie target.
6. **Chat with AI**: Switch to the **AI Assistant** to ask follow-up questions (e.g. *"Is this gluten-free?"* or *"What is additive E330?"*).
7. **History**: Re-open or delete historical analyses at any time from the sidebar.

---

## Technology Stack

| Layer | Component | Official Documentation | Purpose |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | Flask 3.0+ | [flask.palletsprojects.com](https://flask.palletsprojects.com/) | RESTful API, static file delivery, and routing |
| **WSGI Server** | Gunicorn | [gunicorn.org](https://gunicorn.org/) | Multi-worker production HTTP server |
| **AI Inference** | Groq Cloud SDK | [console.groq.com/docs](https://console.groq.com/docs) | Fast LLM inference with structured nutrition prompts |
| **Client-Side OCR** | Tesseract.js v5 | [tesseract.projectnaptha.com](https://tesseract.projectnaptha.com/) | In-browser optical character recognition via WebAssembly |
| **Authentication** | Firebase Authentication | [firebase.google.com/docs/auth](https://firebase.google.com/docs/auth) | Anonymous guest login, email/password linking, ID token generation |
| **Cloud Database** | Firebase Cloud Firestore | [firebase.google.com/docs/firestore](https://firebase.google.com/docs/firestore) | Serverless NoSQL user-isolated scan history (`users/{uid}/scans`) |
| **Firebase Admin** | firebase-admin Python SDK | [firebase.google.com/docs/admin/setup](https://firebase.google.com/docs/admin/setup) | Server-side ID token verification and Firestore execution |
| **Configuration** | python-dotenv | [pypi.org/project/python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |
| **Frontend Runtime** | Vanilla HTML5 / CSS3 / ES6 | [developer.mozilla.org](https://developer.mozilla.org/) | Zero-framework responsive monochrome interface |
| **Cloud Deployment** | Render Free | [render.com/docs](https://render.com/docs) | Free-tier Web Service hosting with automated GitHub builds |

---

## Project Directory Structure

```text
PACKS-sidebar-pages-update/
│
├── .env.example              # Template for local environment variables
├── .gitignore                # Excludes .env, *.db, uploads, and runtime caches
├── app.py                    # Flask application, REST routes, Groq, and Firestore
├── render.yaml               # Render Blueprint free deployment configuration
├── requirements.txt          # Production Python dependencies (includes firebase-admin)
├── README.md                 # Project documentation
│
├── templates/
│   └── index.html            # Complete monochrome SPA (HTML, CSS, JS)
│
└── [Runtime Files - Excluded from Git]
    └── .env                  # Local secret configuration (ignored)
```

---

## Firebase Setup (Auth & Firestore)

PACKS uses **Firebase Authentication** and **Google Cloud Firestore** on the free **Firebase Spark Plan** (no credit card required). Follow these steps to configure your project:

1. **Create a Firebase Project**:
   - Go to the [Firebase Console](https://console.firebase.google.com/).
   - Click **Add Project** and name it (e.g. `packs-food-analyzer`).
   - Disable Google Analytics (optional) and create the project.

2. **Create Cloud Firestore Database**:
   - In the left sidebar, navigate to **Build** > **Firestore Database**.
   - Click **Create Database**.
   - Choose a location (e.g. `nam5 (us-central)`).
   - Select **Production Mode** (Flask's server-side Firebase Admin SDK has direct access while clients cannot read or tamper with other users' data directly).

3. **Enable Firebase Authentication Providers**:
   - In the left sidebar, navigate to **Build** > **Authentication**.
   - Click **Get Started** if initializing authentication for the first time.
   - Under the **Sign-in method** tab, enable the required providers:
     - **Google**: Click **Google**, toggle **Enable**, choose your Project support email, and click **Save**.
     - **Email/Password**: Click **Email/Password**, toggle **Enable**, and click **Save**. (Powers account creation, password login, and password reset recovery).
     - **Anonymous**: Click **Anonymous**, toggle **Enable**, and click **Save**. (Powers instant private guest sessions with isolated UIDs).
   - Under the **Settings** tab > **Authorized domains**, ensure your domains are listed:
     - `localhost` and `127.0.0.1` (automatically authorized for local testing).
     - Your Render production domain (e.g., `packs-food-analyzer.onrender.com`).

4. **Register Web App (Frontend Configuration)**:
   - In **Project Settings** (gear icon) > **General**, scroll to the **Your apps** section.
   - Click the Web icon (`</>`) to add a web app (e.g., `packs-web`).
   - Copy the configuration values into your `.env`:
     ```env
     FIREBASE_CLIENT_API_KEY=AIzaSy...
     FIREBASE_CLIENT_AUTH_DOMAIN=packs-food-analyzer.firebaseapp.com
     FIREBASE_CLIENT_PROJECT_ID=packs-food-analyzer
     FIREBASE_CLIENT_APP_ID=1:123456789:web:abcdef
     ```

5. **Generate Admin Service Account Key (Backend Configuration)**:
   - In **Project Settings**, switch to the **Service accounts** tab.
   - Click **Generate new private key** and confirm.
   - Open the downloaded JSON file in a text editor and copy the contents.
   - In your `.env` (or Render Environment Variables), set:
     ```env
     FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"...","private_key":"..."}
     ```
   - Alternatively, provide the local file path on your development machine.

---

## Prerequisites & Installation

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **A Groq API Key**: Obtain a free key from the [Groq Console](https://console.groq.com/).
- **A Firebase Service Account Key**: Follow [Firebase Firestore Setup](#firebase-firestore-setup) above.

---

### Windows Installation

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/pranav122005/PACKs.git
   cd PACKs
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

3. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```powershell
   Copy-Item .env.example .env
   ```
   Open `.env` in your text editor and supply your `GROQ_API_KEY` and `FIREBASE_SERVICE_ACCOUNT_JSON`.

---

### macOS & Linux Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranav122005/PACKs.git
   cd PACKs
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to include your valid `GROQ_API_KEY` and `FIREBASE_SERVICE_ACCOUNT_JSON`.

---

## Environment Configuration

The application reads configuration settings via `python-dotenv` from `.env`. A complete template is provided in [`.env.example`](file:///.env.example):

```env
# Groq AI Cloud Credentials
GROQ_API_KEY=replace_with_your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TIMEOUT_SECONDS=30.0

# Firebase Cloud Firestore (Admin SDK credentials as single-line JSON or local path)
FIREBASE_SERVICE_ACCOUNT_JSON=replace_with_your_firebase_service_account_json

# Firebase Web Client Configuration (Public configuration for Firebase Authentication & User Isolation)
FIREBASE_CLIENT_API_KEY=replace_with_your_firebase_client_api_key
FIREBASE_CLIENT_AUTH_DOMAIN=replace_with_your_project_id.firebaseapp.com
FIREBASE_CLIENT_PROJECT_ID=replace_with_your_firebase_project_id
FIREBASE_CLIENT_APP_ID=replace_with_your_firebase_client_app_id

# Server Configuration
FLASK_DEBUG=0
PORT=5000
```

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `GROQ_API_KEY` | **Yes** | *None* | Authentication key for Groq Cloud LLM inference. |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | **Yes** | *None* | Service Account JSON string or file path for server-side Firebase Admin. |
| `FIREBASE_CLIENT_API_KEY` | No | *None* | Firebase Web API key exposed safely via `/api/firebase-config`. |
| `FIREBASE_CLIENT_AUTH_DOMAIN` | No | *None* | Firebase Auth Domain (e.g., `<project>.firebaseapp.com`). |
| `FIREBASE_CLIENT_PROJECT_ID` | No | *None* | Firebase project identifier. |
| `FIREBASE_CLIENT_APP_ID` | No | *None* | Firebase Web Application ID. |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Target Groq model identifier. |
| `GROQ_TIMEOUT_SECONDS` | No | `30.0` | Upstream API timeout before failing gracefully. |
| `FLASK_DEBUG` | No | `0` | Enable Flask debugger (`1` for dev, `0` for production). |
| `PORT` | No | `5000` | Port for local server or PaaS binding. |

> [!WARNING]
> Never commit real API keys, Firebase service account keys, or `.env` files to Git. The `.gitignore` file excludes `.env` and all sensitive files automatically.

---

## Running Locally

Once configured, start the server:

```bash
python app.py
```

The application will bind to `http://127.0.0.1:5000`. Open this URL in any modern web browser.

---

## API Endpoint Documentation

All backend routes are documented below according to actual implementation in [`app.py`](file:///app.py).

### 1. Web Interface
```http
GET /
```
- **Description**: Delivers the single-page application interface (`templates/index.html`).
- **Response**: `200 OK` (HTML)

---

### 2. Public Firebase Client Configuration
```http
GET /api/firebase-config
```
- **Description**: Returns safe, public client configuration keys required by the browser to initialize Firebase Authentication. Does not expose private service account keys.
- **Success Response (`200 OK`)**:
  ```json
  {
    "apiKey": "AIzaSy...",
    "authDomain": "packs-app.firebaseapp.com",
    "projectId": "packs-app",
    "appId": "1:123456789:web:abcdef"
  }
  ```

---

### 3. AI Product Analysis
```http
POST /api/analyze
Content-Type: application/json
Authorization: Bearer <FIREBASE_ID_TOKEN>   [Optional]
```
- **Description**: Takes label OCR text and optional instructions, queries Groq Cloud, and saves the result to the authenticated user's private Firestore subcollection (`users/{uid}/scans`).
- **Authentication**:
  - **Authenticated (Anonymous or Email)**: Scan is evaluated and persisted under `users/{uid}/scans/{scan_id}`. Returns generated string scan `id`.
  - **Unauthenticated**: Scan is evaluated with full AI intelligence for privacy, returning `id: null` without entering database history.
- **Request Body**:
  ```json
  {
    "text": "NUTRITION FACTS: Calories 150, Total Fat 3g...",
    "product_name": "Almond Crackers",
    "instruction": "Optional custom prompt extension"
  }
  ```
- **Validation Constraints**:
  - Request body maximum size: 2MB (exceeding returns HTTP 413).
  - `text` must contain at least 3 non-whitespace characters (otherwise returns HTTP 400).
  - Maximum OCR text length: 20,000 characters.
- **Success Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "id": "7xK9u8vW2y1Z",
    "product_name": "Almond Crackers",
    "result": "PRODUCT: Almond Crackers\nNUTRITION:\n- Calories: 150 kcal..."
  }
  ```
- **Error Responses**:
  - `400 Bad Request`: Missing or invalid JSON payload.
  - `413 Payload Too Large`: Payload exceeds 2MB limit.
  - `503 Service Unavailable`: `GROQ_API_KEY` is not configured or upstream API is unreachable.
  - `429 Too Many Requests`: Groq Cloud rate limit exceeded.

---

### 4. Scan History List
```http
GET /api/history
Authorization: Bearer <FIREBASE_ID_TOKEN>
```
- **Description**: Returns all saved scans for the verified user (`users/{uid}/scans`) ordered by `created_at` descending.
- **Success Response (`200 OK`)**:
  ```json
  [
    {
      "id": "7xK9u8vW2y1Z",
      "product_name": "Almond Crackers",
      "created_at": "2026-10-30T20:15:00"
    }
  ]
  ```
- **Error Response**:
  - `401 Unauthorized`: Missing, expired, or invalid Firebase ID token.

---

### 5. Scan History Detail
```http
GET /api/history/<scan_id>
Authorization: Bearer <FIREBASE_ID_TOKEN>
```
- **Description**: Retrieves a single historical scan including full OCR transcript and AI analysis strictly from the verified user's subcollection (`users/{uid}/scans/<scan_id>`). Cross-tenant access is prohibited.
- **Success Response (`200 OK`)**:
  ```json
  {
    "id": "7xK9u8vW2y1Z",
    "product_name": "Almond Crackers",
    "ocr_text": "RAW OCR TEXT...",
    "ai_analysis": "STRUCTURED 7-SECTION ANALYSIS...",
    "created_at": "2026-10-30T20:15:00"
  }
  ```
- **Error Responses**:
  - `401 Unauthorized`: Missing or invalid Firebase ID token.
  - `404 Not Found`: Scan does not exist or belongs to another user.
  - `503 Service Unavailable`: Firestore is not configured.

---

### 6. Delete Scan Record
```http
DELETE /api/history/<scan_id>
Authorization: Bearer <FIREBASE_ID_TOKEN>
```
- **Description**: Permanently removes a document strictly from the authenticated user's subcollection (`users/{uid}/scans/<scan_id>`).
- **Success Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "message": "Scan record deleted successfully."
  }
  ```
- **Error Responses**:
  - `401 Unauthorized`: Missing or invalid Firebase ID token.
  - `404 Not Found`: Scan does not exist or belongs to another user.
  - `503 Service Unavailable`: Firestore is not configured.

---

### 7. Health & Status Check
```http
GET /api/health
```
- **Description**: Monitors Firestore database connectivity and Groq configuration readiness without exposing credentials.
- **Success Response (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "database": "Firestore",
    "database_status": "connected",
    "ocr": "Tesseract.js",
    "ai": "Groq",
    "groq_configured": true,
    "firestore_configured": true
  }
  ```
- **Degraded Response (`503 Service Unavailable`)**: Returned if `FIREBASE_SERVICE_ACCOUNT_JSON` is configured but Firestore connection fails.

---

## Database Schema & Data Persistence

### Server-Side Persistence: User-Isolated Cloud Firestore (`users/{uid}/scans`)

Scans are strictly segmented by verified Firebase User ID (`uid`):

```text
Collection: users/
  └── Document: {uid}
        └── Subcollection: scans/
              └── Document: {auto-generated-scan-id}
                    ├── product_name: string (e.g. "Organic Almond Milk")
                    ├── ocr_text: string (raw extracted label text)
                    ├── ai_analysis: string (structured 7-section breakdown)
                    └── created_at: string (ISO 8601 timestamp)
```

### Client-Side Persistence: Browser `localStorage`

User-specific dietary tracking data is kept on the client device:
- `packs_calorie_goal`: Numeric string storing the target daily calories (e.g. `"2000"`).
- `packs_daily_logs`: Structured JSON object keyed by date (`YYYY-MM-DD`):
  ```json
  {
    "2026-10-30": [
      {
        "id": "1730312400000",
        "name": "Organic Almond Milk",
        "qty": "1 cup",
        "cal": 30,
        "prot": 1,
        "carb": 1,
        "fat": 2.5,
        "time": "08:30 AM"
      }
    ]
  }
  ```

---

## Production Deployment (Render Free + Firestore)

PACKS is pre-configured for a **100% free-tier production stack** using [Render Free Web Services](https://render.com/) and [Firebase Cloud Firestore](https://firebase.google.com/):

### Deploying to Render via Blueprint

1. Push your repository to GitHub.
2. Log in to the [Render Dashboard](https://dashboard.render.com/).
3. Click **New** > **Blueprint** and select your repository.
4. Render will parse [`render.yaml`](file:///render.yaml) automatically:
   - Sets runtime to **Python**.
   - Executes build: `pip install -r requirements.txt`.
   - Starts production WSGI: `gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`.
   - Binds health check to `/api/health`.
5. Under Environment Variables, supply:
   - `GROQ_API_KEY`: *(Your Groq API Key)*
   - `FIREBASE_SERVICE_ACCOUNT_JSON`: *(Your single-line Firebase Service Account JSON)*
   - `FIREBASE_CLIENT_API_KEY`: *(Your public Web API key)*
   - `FIREBASE_CLIENT_AUTH_DOMAIN`: *(e.g. `your-project.firebaseapp.com`)*
   - `FIREBASE_CLIENT_PROJECT_ID`: *(Your Firebase project ID)*
   - `FIREBASE_CLIENT_APP_ID`: *(Your Firebase web app ID)*
6. Click **Apply**.

---

### Manual Web Service Setup

If deploying manually without Blueprints:
- **Environment**: `Python`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`
- **Health Check Path**: `/api/health`
- **Environment Variables**:
   - `GROQ_API_KEY`: *(Your secret API Key)*
   - `FIREBASE_SERVICE_ACCOUNT_JSON`: *(Your secret service account JSON)*
   - `FIREBASE_CLIENT_API_KEY`: *(Your public Web API key)*
   - `FIREBASE_CLIENT_AUTH_DOMAIN`: *(e.g. `your-project.firebaseapp.com`)*
   - `FIREBASE_CLIENT_PROJECT_ID`: *(Your Firebase project ID)*
   - `FIREBASE_CLIENT_APP_ID`: *(Your Firebase web app ID)*
   - `GROQ_MODEL`: `openai/gpt-oss-20b`
   - `FLASK_DEBUG`: `0`

---

### Free-Tier Architecture & Quotas

| Service | Free Tier Allowance | How PACKS Utilizes It |
| :--- | :--- | :--- |
| **Render Web Service** | 750 free instance hours / month | Hosts the Gunicorn/Flask web server. Spins down after 15 min of inactivity (spins back up in ~50s upon new requests). |
| **Firebase Cloud Firestore** | 1 GiB storage, 50k reads/day, 20k writes/day, 20k deletes/day | Permanently preserves scan history across Render container restarts with zero persistent disk costs. |
| **Groq Cloud API** | Generous free tier requests / tokens | Performs near-instant LLM nutritional analysis using `openai/gpt-oss-20b`. |
| **Client Tesseract.js** | Unlimited (runs on client hardware) | Performs fast optical character recognition inside the user's browser via WebAssembly Web Workers. |

---

## Interface & Screenshots

> *Screenshots can be added here. Replace placeholders with actual images.*

| Workspace | Description | Preview |
| :--- | :--- | :---: |
| **New Scan & OCR** | Drag-and-drop zone with Tesseract.js real-time OCR progress | `[ Screenshot: New Scan Workspace ]` |
| **7-Section Report** | Parsed macros, additives breakdown, positives, and watchouts | `[ Screenshot: Analysis Report ]` |
| **Nutrition Dashboard** | Calorie target ring, macro breakdown, and consumption feed | `[ Screenshot: Dashboard Workspace ]` |
| **Daily Log** | Date picker, macro totals, and editable food consumption table | `[ Screenshot: Daily Log Workspace ]` |
| **AI Assistant** | Context-aware chat grounded in food label data | `[ Screenshot: AI Assistant Chat ]` |

---

## Security & Privacy Considerations

1. **Client-Side Image Privacy**: Raw food label images are parsed inside browser WebAssembly memory via Tesseract.js. No images are sent to or stored on backend servers.
2. **Server-Side Firebase Security**: Service account credentials (`FIREBASE_SERVICE_ACCOUNT_JSON`) remain strictly server-side inside Flask environment variables and are never transmitted to the browser.
3. **Content-Security-Policy (CSP)**: `app.py` enforces a strict CSP restricting executable scripts to self and verified CDNs (`cdn.jsdelivr.net`, `tessdata.projectnaptha.com`).
4. **MIME & Frame Guarding**: `X-Content-Type-Options: nosniff` and `X-Frame-Options: SAMEORIGIN` prevent MIME confusion and clickjacking.
5. **Input Sanitization & Limits**:
   - 2MB request body ceiling prevents memory exhaustion attacks.
   - Text inputs are sanitized to strip null bytes and enforce character limits.
6. **Zero Credential Exposure**: The health check endpoint (`/api/health`) exposes only boolean configuration status, never revealing keys or service account credentials.

---

## Current Limitations & Roadmap

### Current Limitations
- **OCR Accuracy**: OCR performance depends on image sharpness, lighting, and packaging curvature.
- **Language Support**: Default client-side OCR is currently configured for English (`eng`).
- **Render Cold Starts**: On Render Free tier, instances sleep after 15 minutes of inactivity; the first request after idling takes ~50 seconds to spin up.

### Planned Roadmap
- [ ] Multi-language OCR support (French, Spanish, German, Hindi).
- [ ] Barcode scanning integration via browser camera.
- [ ] Exportable dietary reports (PDF / CSV format).
- [ ] Allergen profile alerts (customizable warnings for gluten, nuts, dairy, soy).

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**.
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your modifications**:
   ```bash
   git commit -m "feat(module): add concise feature description"
   ```
4. **Verify tests and syntax**:
   ```bash
   python -m py_compile app.py
   ```
5. **Push to your fork and submit a Pull Request**.

---

## License

Licensing has not yet been specified for this project. All rights are reserved by the repository owner.
