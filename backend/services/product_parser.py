import re


def parse_ingredients(text):
    """
    Convert ingredient text into a clean list.
    Example:
    'Sugar, Palm Oil, Salt, INS621'
    ->
    ['Sugar', 'Palm Oil', 'Salt', 'INS621']
    """

    if text is None:
        return []

    text = str(text)

    ingredients = re.split(r",|;", text)

    cleaned = []

    for ingredient in ingredients:

        ingredient = ingredient.strip()

        if ingredient != "":
            cleaned.append(ingredient)

    return cleaned