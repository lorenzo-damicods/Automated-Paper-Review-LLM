import re, json, collections
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

BASELINE_JSONL = "BASELINE_llama32_test20_NORM.jsonl"
TRAINED_JSONL  = "TRAINED_llama32_test20_raw_output_NORM.jsonl"
TEST_PARQUET   = "test_2020.parquet"   

# ────────────────────────────────────────────────────────────────

def parse_decision(x):
    s = "" if x is None else str(x).strip()
    if s.lower() in ("accept", "reject"):
        return s.title()
    m = re.search(r"(?im)^\s*#{0,3}\s*decision\s*[:#\-]*\s*(accept|reject)\b", s)
    if m: return m.group(1).title()
    hits = [h.group(1).lower() for h in re.finditer(r"\b(accept|reject)\b", s, re.I)]
    if hits: return "Accept" if hits[-1] == "accept" else "Reject"
    return None

def norm(s): return re.sub(r"\s+", " ", (s or "")).strip().lower()

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            dec = (rec.get("parsed_decision") or rec.get("decision") or
                   parse_decision(rec.get("raw","")) or
                   parse_decision(rec.get("comment","")))
            rows.append({
                "title_key": norm(rec.get("title","")),
                "decision":  parse_decision(dec)
            })
    return rows

# ── ground truth ──
assert Path(TEST_PARQUET).exists(), f"Non trovo {TEST_PARQUET}"
test_df = pd.read_parquet(TEST_PARQUET)
test_df["title_key"] = test_df["title"].apply(norm)

def mode_dec(series):
    decs = [parse_decision(x) for x in series]
    decs = [d for d in decs if d in ("Accept","Reject")]
    if not decs: return None
    return collections.Counter(decs).most_common(1)[0][0]

gt = (test_df.groupby("title_key", as_index=False)
             .agg(gt_decision=("paper_decision", mode_dec)))

# ── Predictions ──
base_rows = {r["title_key"]: r["decision"] for r in load_jsonl(BASELINE_JSONL)}
trn_rows  = {r["title_key"]: r["decision"] for r in load_jsonl(TRAINED_JSONL)}

gt["base_pred"]    = gt["title_key"].map(base_rows)
gt["trained_pred"] = gt["title_key"].map(trn_rows)

# ── Report ──
for label, pred_col in [("BASELINE", "base_pred"), ("QLORA", "trained_pred")]:
    sub = gt[gt[pred_col].isin(["Accept","Reject"]) &
             gt["gt_decision"].isin(["Accept","Reject"])].copy()
    print(f"\n{'='*55}")
    print(f"  {label}   (n={len(sub)})")
    print(f"{'='*55}")
    print(classification_report(sub["gt_decision"], sub[pred_col],
                                 target_names=["Accept","Reject"], digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(sub["gt_decision"], sub[pred_col],
                          labels=["Accept","Reject"])
    print(f"               Pred Accept  Pred Reject")
    print(f"  True Accept  {cm[0,0]:>10}  {cm[0,1]:>10}")
    print(f"  True Reject  {cm[1,0]:>10}  {cm[1,1]:>10}")

