import pandas as pd
from pathlib import Path

# === PATHS ===
BASE_DIR = Path("/home/lorenzods/Scrivania/finetuning_project")
DATA_PATH = BASE_DIR / "absolute_data/tp_2020conference.csv"
OUTPUT_PATH = BASE_DIR / "data_parquet/test_20.parquet"

# === SUPPORT FUNCTIONS ===
def clean_text(val):
    """Ensure the value is a clean string without leading/trailing spaces."""
    if not isinstance(val, str):
        return ""
    return val.strip()

# === LOAD CSV ===
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

# Define column names from CSV
title_col = "title"
abstract_col = "abstract"
review_col = "review"
decision_col = "paper_decision"

# Keep only the required columns
df = df[[title_col, abstract_col, review_col, decision_col]]

# Drop rows with missing required fields
df = df.dropna(subset=[title_col, abstract_col, review_col, decision_col])

# Clean text fields
df[title_col] = df[title_col].apply(clean_text)
df[abstract_col] = df[abstract_col].apply(clean_text)
df[review_col] = df[review_col].apply(clean_text)
df[decision_col] = df[decision_col]

# Remove duplicates
df_final = df.drop_duplicates()

# Save to Parquet
df_final.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Saved test set to {OUTPUT_PATH} ({len(df_final)} rows)")
