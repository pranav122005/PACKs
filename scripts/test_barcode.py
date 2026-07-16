from pathlib import Path
from backend.scanner.barcode_scanner import scan_barcode

image = Path("uploads/test.jpg")

print("Exists:", image.exists())
print("Absolute:", image.resolve())

result = scan_barcode(str(image))

print(result)