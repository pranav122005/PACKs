from backend.services.alias_service import normalize
from backend.services.ingredient_service import get_ingredient


def analyse(ingredients):

    report=[]

    for ingredient in ingredients:

        ingredient=normalize(ingredient)

        data=get_ingredient(ingredient)

        if data:
            report.append(data)

    return report 