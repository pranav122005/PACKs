from backend.services.product_parser import parse_ingredients

print(
    parse_ingredients(
        "Sugar, Palm Oil, Salt, INS621"
    )
)