ALIASES = {

    "E621":"Monosodium Glutamate",
    "INS621":"Monosodium Glutamate",
    "MSG":"Monosodium Glutamate",

    "HFCS":"High Fructose Corn Syrup",
    "Corn Syrup":"High Fructose Corn Syrup",
    "Glucose Syrup":"High Fructose Corn Syrup",

    "E951":"Aspartame",

    "Palmolein":"Palm Oil"

}


def normalize(name):

    if name in ALIASES:
        return ALIASES[name]

    return name