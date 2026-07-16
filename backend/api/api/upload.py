from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pathlib import Path
import re
import uuid
import shutil
import logging
from typing import Optional

logger = logging.getLogger("packs.api.upload")

router = APIRouter(tags=["scan"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_scans_db: dict = {}


def _fallback_parse_ingredients(report: dict) -> list:
    raw = report.get("message", "") or ""
    if not raw or "extract" in raw.lower() or "no" in raw.lower():
        return []
    if "ingredient" in raw.lower() or ":" in raw:
        parts = re.split(r"[;,]", raw)
        return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
    return []


def _parse_disease_warnings(disease: dict) -> list:
    results = []
    seen = set()
    for w in disease.get("warnings", []):
        kw = w.lower()
        condition = None
        if "diabet" in kw:
            condition = "Diabetes"
        elif "hypertension" in kw or "sodium" in kw or "salt" in kw:
            condition = "Hypertension"
        elif "trans fat" in kw or "cardiovascular" in kw or "heart" in kw:
            condition = "Cardiovascular Health"
        elif "obes" in kw or "calorie" in kw:
            condition = "Obesity"
        elif "allergen" in kw or "allerg" in kw:
            condition = "Allergies"
        if condition and condition not in seen:
            seen.add(condition)
            results.append({
                "condition": condition,
                "compatibility": "avoid" if "high" in kw or "risk" in kw else "caution",
                "explanation": w,
            })
    return results


def _map_report_to_scan_result(scan_id: str, report: dict, image_url: str = "") -> dict:
    health_score = report.get("health_score", {}) or {}
    nutrition = report.get("nutrition", {}) or {}
    nova = report.get("nova", {}) or {}
    disease = report.get("disease_analysis", {}) or {}
    disease_warnings = disease.get("warnings", []) if isinstance(disease, dict) else []
    warnings = report.get("warnings", []) or []
    positives = report.get("positives", []) or []
    recommendations = report.get("recommendations", []) or []

    score = health_score.get("overall_score", 0) or 0
    grade = health_score.get("grade", "C") or "C"

    ingredients_raw = report.get("ingredients", []) or []
    ingredients = [
        {"name": i, "isAdditive": False, "isAllergen": False, "riskLevel": "safe"}
        if isinstance(i, str)
        else {"name": i.get("name", str(i)), "isAdditive": i.get("is_additive", False), "isAllergen": i.get("is_allergen", False), "riskLevel": i.get("risk_level", "safe")}
        for i in ingredients_raw
    ]

    disease_analysis = [
        {"condition": k.replace("_", " ").title(), "compatibility": v.get("compatibility", "caution"), "explanation": v.get("explanation", "")}
        for k, v in (disease if isinstance(disease, dict) else {}).items()
        if isinstance(v, dict)
    ]
    if not disease_analysis and disease_warnings:
        disease_analysis = _parse_disease_warnings(disease)

    return {
        "id": scan_id,
        "productName": report.get("product_name", ""),
        "brand": report.get("brand", ""),
        "imageUrl": image_url,
        "barcode": report.get("barcode"),
        "scannedAt": report.get("scanned_at", ""),
        "overallScore": score,
        "healthGrade": grade,
        "novaGroup": nova.get("nova_group", 1),
        "nutrition": {
            "servingSize": nutrition.get("serving_size", ""),
            "calories": nutrition.get("calories"),
            "protein": nutrition.get("protein_g"),
            "carbohydrates": nutrition.get("carbohydrates_g"),
            "sugar": nutrition.get("sugar_g"),
            "addedSugar": nutrition.get("added_sugar_g"),
            "fat": nutrition.get("fat_g"),
            "saturatedFat": nutrition.get("saturated_fat_g"),
            "transFat": nutrition.get("trans_fat_g"),
            "fiber": nutrition.get("fiber_g"),
            "sodium": nutrition.get("sodium_mg"),
            "cholesterol": nutrition.get("cholesterol"),
            "calcium": nutrition.get("calcium"),
            "iron": nutrition.get("iron"),
        },
        "ingredients": ingredients,
        "additives": nova.get("additives", report.get("additives", [])),
        "warnings": list(dict.fromkeys(warnings + disease_warnings))[:3],
        "positives": positives[:3] if positives else [],
        "recommendations": recommendations[:3] if recommendations else [],
        "diseaseAnalysis": disease_analysis,
        "aiSummary": report.get("ai_summary", ""),
    }


@router.post("/scan")
async def scan_product(
    image: Optional[UploadFile] = File(None),
    barcode: Optional[str] = Form(None),
):
    if not image and not barcode:
        raise HTTPException(status_code=400, detail="Provide an image or barcode")

    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    image_url = ""
    filepath = None

    if image:
        ext = Path(image.filename).suffix if image.filename else ".jpg"
        filename = f"{scan_id}{ext}"
        filepath = UPLOAD_DIR / filename
        with filepath.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_url = f"/uploads/{filename}"

    report = {}
    try:
        if filepath:
            from backend.services.final_analysis_pipeline import run_final_analysis_pipeline
            report = run_final_analysis_pipeline(str(filepath))
    except Exception as e:
        logger.warning("Pipeline failed, returning partial result: %s", e)

    result = _map_report_to_scan_result(scan_id, report, image_url)
    result["scannedAt"] = now
    if barcode:
        result["barcode"] = barcode
    if not result["productName"]:
        result["productName"] = image.filename.rsplit(".", 1)[0] if image and image.filename else "Product Scan"
    if not result["ingredients"]:
        fallback = _fallback_parse_ingredients(report)
        if fallback:
            result["ingredients"] = [{"name": i, "isAdditive": False, "isAllergen": False, "riskLevel": "safe"} for i in fallback]
    if not result["aiSummary"]:
        result["aiSummary"] = (
            f"This product received an overall health grade of {result['healthGrade']} "
            f"({result['overallScore']}/100). "
            f"{len(result['ingredients'])} ingredient(s) detected."
        )

    _scans_db[scan_id] = result
    return result


@router.get("/scan/{scan_id}")
async def get_scan(scan_id: str):
    result = _scans_db.get(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    return result



