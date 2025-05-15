from utils import create_double_cv
import pandas as pd

df = pd.read_csv("../data/train.csv")
df = df[["study_id"]].dropna()  # Ensure no NaNs
df = create_double_cv(df, "study_id", 5, 5)
df.to_csv("../data/folds_cv5.csv", index=False)