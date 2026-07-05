from fastapi import FastAPI
from backend.api.upload import router

app = FastAPI(title="PACKS API")

app.include_router(router)

@app.get("/")
def home():
    return {
        "message": "PACKS Running"
    }