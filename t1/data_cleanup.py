# %%
# Read data and examine it
import pandas as pd
import json


INPUT_FILE = "data/conjunto_de_teste.csv"
OUTPUT_FILE = "clean_test_ds.csv"
TARGET_ENCODINGS = "target_encodings.json"
IS_TEST_FILE = True

df = pd.read_csv(INPUT_FILE)

# %%
# Assess Nulls and missing
df.info()

# %%
# Check non-numerical columns for unique values
str_df = df.select_dtypes(include=[object])
uniques = []

for col in str_df.columns:
    uniques.append(
        [col, len(str_df[col].unique())]
    ) 

pd.DataFrame(uniques, columns=["column", "unique-count"])

# %%
# Transform sex into numbers
# Go from "sexo" (M, F, N, " ") to 0, 1
df = df[df["sexo"] != " "]  # Drop empty strings
df["sexo"] = df["sexo"].map(lambda e: 1 if e == "F" else 0)  # Cast to num, assuming N is typo for M

# %%
# Transform columns with 2 values as binary
bin_cols = [
    "possui_telefone_residencial",
    "vinculo_formal_com_empresa",
    "possui_telefone_trabalho"
]

for col in bin_cols:
    first_val = df[col].unique()[0]  # Get first value in series
    df[col] = df[col].map(
        lambda e: 1 if e == first_val else 0
    )

# %%
# Target-encode some categorical features using their relative frequency by category
cols = [
    "estado_onde_nasceu",
    "estado_onde_reside",
    "estado_onde_trabalha",
    "codigo_area_telefone_residencial",
    "codigo_area_telefone_trabalho"
]

# If it's a training file, create encodings
if not IS_TEST_FILE:
    encodings = dict([(name, {}) for name in cols])

    for col in cols:
        encoded_col = "TE_" + col
        df[encoded_col] = df.groupby(col)["inadimplente"].transform('mean')

        # Save all encoded values to encodings dict
        for val in df[[col, encoded_col]].drop_duplicates().iloc:
            encodings[col][val[col]] = val[encoded_col]

        # Replace original column by encoded
        df[col] = df[encoded_col]
        df.drop(columns=[encoded_col])

        # Save encodings to JSON
        json.dump(encodings, open(TARGET_ENCODINGS, "w"))

# If it's a test file, read encodings and apply them
else:
    encodings = json.load(open(TARGET_ENCODINGS, "r"))

    for col in cols:

        def apply_encoding(el):
            # Used saved encodings, if missing use 0.5
            try:
                return encodings[col][el]
            except KeyError:
                print(f"missing {el} in {col}, using 0.5")
                return 0.5

        df[col] = df[col].apply(apply_encoding)

# %%
# One-hot encode "forma_envio_solicitacao"
df["envio_presencial"] = (df["forma_envio_solicitacao"] == "presencial").astype(float)
df["envio_internet"] = (df["forma_envio_solicitacao"] == "internet").astype(float)
df["envio_correio"] = (df["forma_envio_solicitacao"] == "correio").astype(float)

# %%
# Drop columns which are mostly null, with no way to fill them
useless = [
    "possui_telefone_celular",    # All 'N's
    "grau_instrucao",             # All 0's
    "profissao_companheiro",      # ~2/3s empty
    "grau_instrucao_companheiro", # ~2/3s empty
    "forma_envio_solicitacao",    # one-hot encoded before
]

# Unique transactional id
if not IS_TEST_FILE:
    useless.append("id_solicitante")
df = df.drop(useless, axis=1)

# %%
# Get correlation of numerical columns
num_df = df.select_dtypes(exclude=[object])
corr = num_df.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=2)

# %%
# Drop redundant numerical columns (corr = 1)
redundant = [
    "qtde_contas_bancarias_especiais",
    "local_onde_trabalha"
]

df = df.drop(redundant, axis=1)

# %%
# Sort df by correlation with target column
if not IS_TEST_FILE:
    corr = df.corr()
    df = df[abs(corr["inadimplente"]).sort_values(ascending=False).index]

# %%
# Drop remaining NAs and dump df to csv file
df = df.dropna()
df.to_csv(OUTPUT_FILE, index=False)
