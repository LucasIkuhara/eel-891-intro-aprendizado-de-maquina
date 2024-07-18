# %%
# Read data and examine it
import pandas as pd

INPUT_FILE = "data/conjunto_de_treinamento.csv"
OUTPUT_FILE = "clean_training_ds.csv"

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
df["estado_onde_nasceu"] = df.groupby("estado_onde_nasceu")["inadimplente"].transform('mean')
df["estado_onde_reside"] = df.groupby("estado_onde_reside")["inadimplente"].transform('mean')
df["estado_onde_trabalha"] = df.groupby("estado_onde_trabalha")["inadimplente"].transform('mean')
df["codigo_area_telefone_residencial"] = df.groupby("codigo_area_telefone_residencial")["inadimplente"].transform('mean')
df["codigo_area_telefone_trabalho"] = df.groupby("codigo_area_telefone_trabalho")["inadimplente"].transform('mean')

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
    "id_solicitante"              # Unique transactional id  
]

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
corr = df.corr()
df = df[abs(corr["inadimplente"]).sort_values(ascending=False).index]

# %%
# Drop remaining NAs and dump df to csv file
df = df.dropna()
df.to_csv(OUTPUT_FILE, index=False)
