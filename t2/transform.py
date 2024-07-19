# %%
# Imports and args
import pandas as pd
import json
import argparse


INPUT_FILE = "data/conjunto_de_treinamento.csv"
OUTPUT_FILE = "train_ds.csv"
TARGET_ENCODINGS = "bairro.json"
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
# One-hot encode categorical columns with few values
# Treat raw text in "tipo"
for tipo in ['Casa', 'Apartamento', 'Quitinete', 'Loft']:
    df[f"tipo_{tipo.lower()}"] = df["tipo"] == tipo

# Treat raw text in "tipo_vendedor"
df["tipo_vendedor"] = df["tipo_vendedor"] == 'Imobiliaria'

df = df.drop(columns=["tipo"])

# %%
# Treat "bairro" replacing the value by its mean value
if IS_TRAINING_FILE:
    bairro = {}
    df["bairro_media"] = df.groupby("bairro")["preco"].transform('mean')
    for val in df[["bairro", "bairro_media"]].drop_duplicates().iloc:
        bairro[val.bairro] = val.bairro_media

    # Pre-compute mean to use in missing 'bairro' values
    bairro["default"] = df["bairro_media"].mean()
    json.dump(bairro, open(TARGET_ENCODINGS, "w"))

    # Replace name by means
    df["bairro"] = df["bairro_media"]
    df = df.drop(columns=["bairro_media"])

else:
    encodings = json.load(open(TARGET_ENCODINGS, "r"))

    def apply_encoding(el):
        # Used saved encodings, if missing use 0.5
        try:
            return encodings[el]
        except KeyError:
            print(f"missing encoding for {el}.")
            return encodings["default"]

    df["bairro"] = df["bairro"].apply(apply_encoding)

# %%
# Merge features to reduce dimensionality
df = df.drop(columns=["diferenciais"])
feats = {
    "churrasqueira": 1,
    "estacionamento": 1,
    "piscina": 1,
    "playground": 1,
    "quadra": 1,
    "s_festas": 1,
    "s_jogos": 1,
    "s_ginastica": 1,
    "sauna": 1,
    "vista_mar": 1,
}

df["amenities"] = 0
for col in feats:
    w = feats[col]
    df["amenities"] = df["amenities"] + w * df[col]

df = df.drop(columns=list(feats.keys()))

# %%
# Dump processed file
df.to_csv(OUTPUT_FILE, index=False)
