import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop unnecessary columns
drop_cols = ["CustomerID"] + [c for c in df.columns if c.startswith("Unnamed")]
df = df.drop(columns=drop_cols)


# Trim stray whitespace in all text columns
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].apply(lambda s: s.str.strip())

# "Fe Male" is a typo for "Female"
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
# "Unmarried" is merged into "Single" so the app and model use the same labels
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# ---------------- Duplicates and missing target ----------------
n_before = len(df)
df = df.drop_duplicates()
print(f"Removed {n_before - len(df)} duplicate rows")

df = df.dropna(subset=["ProdTaken"])


X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) ProdTaken ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
