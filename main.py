import pandas as pd

df = pd.read_csv("data/tox21.csv")  # change name if needed

print("Shape:", df.shape)
print(df.head())
print(df.columns)