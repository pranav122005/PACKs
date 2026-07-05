from fastapi import APIRouter, UploadFile, File
import shutil
import os

from backend.scanner.barcode_scanner import scan_barcode
from backend.services.product_lookup import find_product_by_barcode
from backend.services.analyse_service import analyse_product

router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/scan")
async def scan(file: UploadFile = File(...)):

    path = os.path.join(

        UPLOAD_DIR,

        file.filename

    )

    with open(path, "wb") as buffer:

        shutil.copyfileobj(

            file.file,

            buffer

        )

    barcodes = scan_barcode(path)

    if not barcodes:

        return {

            "status": "barcode_not_found"

        }

    barcode = barcodes[0]["data"]

    product = find_product_by_barcode(barcode)

    if product is None:

        return {

            "status": "product_not_found",

            "barcode": barcode

        }

    report = analyse_product(product)

    return report