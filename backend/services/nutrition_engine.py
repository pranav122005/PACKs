def nutrition_score(product):

    score = 100

    calories = product.get("calories") or 0
    sugar = product.get("sugars") or 0
    salt = product.get("salt") or 0
    protein = product.get("protein") or 0
    fiber = product.get("fiber") or 0

    warnings = []

    # Sugar
    if sugar > 22:
        score -= 25
        warnings.append("Very High Sugar")

    elif sugar > 10:
        score -= 15
        warnings.append("High Sugar")

    # Salt
    if salt > 1.5:
        score -= 20
        warnings.append("High Salt")

    elif salt > 0.6:
        score -= 10
        warnings.append("Moderate Salt")

    # Calories
    if calories > 500:
        score -= 15

    # Protein Bonus
    if protein > 15:
        score += 5

    # Fiber Bonus
    if fiber > 5:
        score += 5

    score = max(0, min(score, 100))

    return score, warnings