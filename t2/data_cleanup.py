# %%
# Imports and args
import pandas as pd
import json
import argparse


INPUT_FILE = "data/conjunto_de_treinamento.csv"
OUTPUT_FILE = "train_ds.csv"
IS_TRAINING_FILE = True

# %%
# Read alternative files or modes from kwargs
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in", help="Input csv file", required=False)
parser.add_argument("-o", "--out", help="Output csv file", required=False)
parser.add_argument(
    "-t",
    "--test",
    help="Treat input as test file (instead of training)",
    required=False,
    action="store_true",
)
args = vars(parser.parse_args())

if args["in"]:
    INPUT_FILE = args["in"]

if args["out"]:
    OUTPUT_FILE = args["out"]

if args["test"]:
    IS_TRAINING_FILE = False

print(
    f"Starting script with INPUT={INPUT_FILE} and OUTPUT={OUTPUT_FILE} in {'training' if IS_TRAINING_FILE else 'test'} mode."
)

# %%
# Read data
df = pd.read_csv(INPUT_FILE)

# %%
# Check for nulls
df.info()


# %%
# Treat raw text in "diferenciais"
# Create score base
def score_amenities(el):
    score = 0
    amenities = {
        "piscina": 1,
        "copa": 1,
        "churrasqueira": 1,
        "sauna": 1,
        "quadra": 1,
        "campo": 1,
        "sala": 1,
        "playground": 1,
    }

    for feat in amenities:
        if feat in el:
            score += amenities[feat]

    return score


df["amenities"] = df.diferenciais.map(score_amenities)

# %%
# Drop diferenciais
df = df.drop(columns=["diferenciais"])

# %%
# Dump processed file
df.to_csv(OUTPUT_FILE, index=False)
