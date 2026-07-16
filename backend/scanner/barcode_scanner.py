import cv2
from pyzbar.pyzbar import decode


def scan_barcode(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return []

    barcodes = decode(image)

    results = []

    for barcode in barcodes:
        results.append({
            "type": barcode.type,
            "data": barcode.data.decode("utf-8")
        })

    return results