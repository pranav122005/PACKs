from backend.services.nutrition_engine import nutrition_score

product = {

    "calories":520,
    "sugars":28,
    "salt":1.8,
    "protein":8,
    "fiber":2

}

print(nutrition_score(product))