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
# Check non-numerical columns
str_df = df.select_dtypes(include=[object])
uniques = []

for col in str_df.columns:
    uniques.append(
        [col, len(str_df[col].unique())]
    ) 

pd.DataFrame(uniques, columns=["column", "unique-count"])

# %%
# Get correlation of numerical columns
num_df = df.select_dtypes(exclude=[object])
corr = num_df.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=2)

# %%
# Drop redundant columns 
df = df.drop(["qtde_contas_bancarias_especiais", ], axis=1)

