import pandas as pd
from pathlib import Path

INPUT_FILE = "datasets/processed/ingredients_raw.csv"
OUTPUT_FILE = "datasets/processed/ingredients_clean.csv"

print("Loading ingredients...")

df = pd.read_csv(INPUT_FILE)

print("Cleaning...")

# Remove leading/trailing spaces
df["Ingredient"] = df["Ingredient"].str.strip()

# Convert to lowercase
df["Ingredient"] = df["Ingredient"].str.lower()

# Remove duplicate ingredients
df = df.groupby("Ingredient", as_index=False)["Count"].sum()

# Sort by count
df = df.sort_values(by="Count", ascending=False)

# Save
Path("datasets/processed").mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print()
print("="*40)
print("Normalization Complete")
print("="*40)

print("Unique Ingredients :", len(df))
print("Saved :", OUTPUT_FILE)
