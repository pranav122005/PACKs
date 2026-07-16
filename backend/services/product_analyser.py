from backend.services.health_service import analyse
from backend.services.score_service import calculate


def analyse_product(ingredients):

    report=analyse(ingredients)

    score=calculate(report)

    return {

        "score":score,

        "ingredients":report

    }