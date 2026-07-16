from backend.services.health_service import analyse
from backend.services.score_service import calculate

ingredients=[

"Sugar",

"MSG",

"Palmolein"

]

report=analyse(ingredients)

print(report)

print(calculate(report))