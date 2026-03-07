import pandas as pd
from pathlib import Path
from typing import List

# === 1. PATHS ===========================================================
BASE_DIR      = Path("/home/lorenzods/Scrivania/finetuning_project")
ABSOLUTE_DATA = BASE_DIR / "absolute_data"
PARQUET_DIR   = BASE_DIR / "data_parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATHS = {
    2017: ABSOLUTE_DATA / "tp_2017conference.csv",
    2018: ABSOLUTE_DATA / "tp_2018conference.csv",
    2019: ABSOLUTE_DATA / "tp_2019conference.csv",
}

# === 2. SUPPORT FUNCTIONS ==============================================

def get_col(df, keywords, required: bool = True) -> str:
    """Return first column whose name contains one of the given keywords (case-insensitive)."""
    if isinstance(keywords, str):
        keywords = [keywords]
    cols = {c.lower(): c for c in df.columns}
    for lc, orig in cols.items():
        for kw in keywords:
            if kw.lower() in lc:
                return orig
    if required:
        raise ValueError(f"❌ Column with keywords {keywords} not found. Available: {list(df.columns)}")
    return None

def clean_text(value: str, prefix: str = "") -> str:
    """Strip optional prefixes like 'Abstract:###'/'Decision:###' and surrounding spaces/punctuation."""
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if prefix and s.lower().startswith(prefix.lower()):
        s = s[len(prefix):].strip(" :#")
    return s

def normalize_decision_binary(decision: str) -> str:
    """
    Binary normalization:
      - if string contains 'accept' (in any form), return 'Accept'
      - else return 'Reject'
    This also handles things like 'Recommendation: Accept', 'accepted', etc.
    """
    s = clean_text(decision, "Decision:###")
    s = clean_text(s, "Recommendation:###").lower()
    return "Accept" if "accept" in s else "Reject"

def process_file(path: Path, year: int) -> pd.DataFrame:
    """Read one CSV and return a DataFrame with prompt & completion (binary labels)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()  # keep original case for get_col()

    title_col    = get_col(df, "title")
    abstract_col = get_col(df, "abstract")
    review_col   = get_col(df, "review")
    # decision columns vary: try common variants
    decision_col = get_col(df, ["paper_decision", "decision", "recommendation", "accept"], required=True)

    df = df.dropna(subset=[title_col, abstract_col, review_col, decision_col])

    # Build prompt/completion
    df["prompt"] = (
        df[title_col].astype(str).str.strip()
        + "\n\n"
        + df[abstract_col].astype(str).apply(lambda x: clean_text(x, "Abstract:###"))
    )

    decisions_bin = df[decision_col].astype(str).apply(normalize_decision_binary)

    df["completion"] = (
        df[review_col].astype(str).str.strip()
        + "\n\nDecision: "
        + decisions_bin
    )

    df["year"] = year
    out = df[["prompt", "completion", "year"]].drop_duplicates()

    # Quick sanity print (counts of labels)
    counts = decisions_bin.value_counts().to_dict()
    print(f"Year {year}: label counts {counts}")
    return out

# === 3. MAIN ===========================================================
def main() -> None:
    frames: List[pd.DataFrame] = []
    for year, path in CSV_PATHS.items():
        print(f"📂 Processing {path.name} ({year})")
        frames.append(process_file(path, year))

    train_df = pd.concat(frames, ignore_index=True)
    out_path = PARQUET_DIR / "train_17_18_19.parquet"
    train_df.to_parquet(out_path, index=False)

    print(f"\n✅ Saved: {out_path}  •  {len(train_df)} rows")
    print(train_df["completion"].str.extract(r"Decision:\s*(\w+)")[0].value_counts())

if __name__ == "__main__":
    main()
