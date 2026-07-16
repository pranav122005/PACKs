import orjson

# Path to your dataset
DATASET_PATH = "datasets/raw/openfoodfacts-products.jsonl"

with open(DATASET_PATH, "rb") as file:

    # Read only the first line
    first_line = file.readline()

    # Convert JSON bytes into a Python dictionary
    product = orjson.loads(first_line)

    print(product)