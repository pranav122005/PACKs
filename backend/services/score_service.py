def calculate(report):

    score=100

    for item in report:

        health=item[5]

        score-=100-health

    if score<0:
        score=0

    return score