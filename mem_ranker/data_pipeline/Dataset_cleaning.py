import pandas as pd

df = pd.read_csv("Breithaupt_original.csv")

keep = [
    "Original text",
    "G1 Story",
    "G2 Retelling",
    "G3 Retelling"
]

clean = df[keep]

clean.to_csv("Breithaupt_Cleaned.csv", index=False)

print("Saved:", len(clean), "rows")
print("Columns:", clean.columns.tolist())