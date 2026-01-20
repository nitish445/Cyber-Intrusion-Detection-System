import pandas as pd

df = pd.read_csv("processed/user_daily.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
