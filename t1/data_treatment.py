# %%
# Read data and examine it
import pandas as pd


df = pd.read_csv("data/conjunto_de_treinamento.csv")
df.info()

# %%
# Assess Nulls and missing
df[df.isna()].info()

# %%
# Drop "grau_instrucao" which only has 0's
df = df.drop(["grau_instrucao"], axis=1)

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
# Transform columns into numbers

# Go from "sexo" (M, F, N, " ") to 0, 1
df = df[df["sexo"] != " "]  # Drop empty strings
df["sexo"] = df["sexo"].map(lambda e: 1 if e == "F" else 0)  # Cast to num, assuming N is typo for M

# Drop "possui_telefone_celular" which only has one value
df = df.drop(["possui_telefone_celular"], axis=1)

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
# Get correlation of numerical columns
num_df = df.select_dtypes(exclude=[object])
corr = num_df.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=2)

# %%
# Drop redundant numerical columns (corr = 1)
df = df.drop(["qtde_contas_bancarias_especiais", "local_onde_reside"], axis=1)

# %%
# Dump df to csv
df.to_csv("clean_training_ds.csv")
