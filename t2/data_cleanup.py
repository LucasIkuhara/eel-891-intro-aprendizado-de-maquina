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
# One-hot encode common features
df["tem_piscina"] = df.diferenciais.map(lambda el: "piscina" in el)
df["tem_copa"] = df.diferenciais.map(lambda el: "copa" in el)
df["tem_churrasqueira"] = df.diferenciais.map(lambda el: "churrasqueira" in el)
df["tem_sauna"] = df.diferenciais.map(lambda el: "sauna" in el)
df["tem_quadra"] = df.diferenciais.map(lambda el: "quadra" in el)
df["tem_sala"] = df.diferenciais.map(lambda el: "sala" in el)

# %%
# Drop diferenciais
df = df.drop(columns=["diferenciais"])

# %%
# Dump processed file
df.to_csv(OUTPUT_FILE, index=False)
