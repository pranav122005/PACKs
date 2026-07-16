import orjson

DATASET_PATH = "datasets/raw/openfoodfacts-products.jsonl"

with open(DATASET_PATH, "rb") as file:

    first_line = file.readline()
    product = orjson.loads(first_line)

    print("=" * 50)
    print("PRODUCT INFORMATION")
    print("=" * 50)

    print("Barcode :", product.get("code"))
    print("Product :", product.get("product_name"))
    print("Brand   :", product.get("brands"))
    print("Country :", product.get("countries"))
    print("Category:", product.get("categories"))
    print("Quantity:", product.get("quantity"))

    print("\nIngredients:")
    print(product.get("ingredients_text"))

    print("\nNutrition:")

    nutrition = product.get("nutrition", {})
    aggregated = nutrition.get("aggregated_set", {})
    nutrients = aggregated.get("nutrients", {})

    for nutrient in [
        "energy-kcal",
        "proteins",
        "fat",
        "carbohydrates",
        "sugars",
        "fiber",
        "salt",
        "sodium"
    ]:

        if nutrient in nutrients:
            print(
                nutrient,
                ":",
                nutrients[nutrient].get("value"),
                nutrients[nutrient].get("unit")
            )
            