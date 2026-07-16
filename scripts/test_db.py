from backend.services.product_lookup import find_product_by_barcode

barcode = "8901491101837"

product = find_product_by_barcode(barcode)

print(product)