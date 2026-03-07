# Automated Paper Reviewing with a Compact LLM: QLoRA Fine-Tuning

> **Course:** Introduction to Machine Learning — University of Trento, Department of Sociology and Social Research  
> **Author:** Lorenzo D'Amico (238684)  
> **Full report:** [`report_introduction_to_ml_LORENZO_D_AMICO_238684.pdf`](./report_introduction_to_ml_LORENZO_D_AMICO_238684.pdf)  
> **All large files (data, model outputs):** [Google Drive](https://drive.google.com/drive/folders/1XQKbqUhGCq4b5yfLhBnxhAM3o07K3kSm)

---

## Overview

This project investigates whether a compact ~3B-parameter language model (**Llama-3.2-3B-Instruct**), adapted with **QLoRA**, can assist peer review by generating structured feedback and predicting an explicit **Accept / Reject** decision.

- **Training data:** OpenReview ICLR 2017–2019 (`train_17_18_19.parquet`)
- **Test data:** ICLR 2020 — strictly out-of-distribution (`test_2020.parquet`)
- **Task:** Binary classification (Accept / Reject) + structured JSON review generation
- **Training hardware:** NVIDIA A40 48 GB (university GPU cluster)

---

## Key Results

| Metric | Zero-Shot Baseline | QLoRA Fine-Tuned |
|---|---|---|
| Accuracy (OOD, ICLR 2020) | 30.9% | **41.7%** |
| Macro F1 | 0.237 | **0.415** |
| Accept predictions | 2199 / 2203 | ~1650 / 2196 |
| Reject predictions | 4 / 2203 | ~546 / 2196 |
| Reject Precision | — | **0.716** |
| Validation Loss | ~3.0 | **~2.27** |
| Perplexity | ~20 | **~9.7** |
| Parameters trained | 0% | **1.33%** (24.3M / 1.83B) |

> ⚠️ **Reference point:** a naive always-Reject classifier achieves **69.1% accuracy** on this test set (69% of ICLR 2020 submissions were rejected). Both models fall below this ceiling — **macro F1 is the meaningful metric** on this imbalanced dataset, and it nearly doubles with fine-tuning (0.237 → 0.415).

### Review Quality (Lexical & Semantic Overlap)

| Metric | Baseline | QLoRA |
|---|---|---|
| Token-Jaccard (mean) | 0.130 | **0.146** |
| Weaknesses ROUGE-1 | 0.149 | **0.169** |
| Strengths chrF | 0.205 | **0.217** |
| SBERT cosine (mean) | **0.633** | 0.585 |
| Strengths ROUGE-L | **0.124** | 0.110 |

QLoRA gains in lexical specificity (Jaccard, ROUGE, chrF); the baseline retains a slight edge on semantic similarity (SBERT). This trade-off reflects domain adaptation shifting the model from abstract fluency toward reviewer-style vocabulary.

---

## Repository Structure

```
.
├── 0_preprocessing_for_training_dataset.py   # Build train_17_18_19.parquet from raw dumps
├── 1_preprocessing_test_20_parquet.py        # Build test_2020.parquet (no decision leakage)
├── Baseline_Training_and_Evaluation.ipynb    # Full pipeline: baseline → QLoRA → KL → eval
│                                             # Pre-computed outputs included; GPU needed to re-run
├── confusion_matrix.py                       # Confusion matrix plotting (standalone)
├── figures/                                  # All plots used in the report
├── review_similarity_metrics.csv             # Full lexical + semantic similarity results (504 KB)
├── rougeL_pairs.csv                          # ROUGE-L pairwise scores: human vs. generated (84 KB)
├── report_introduction_to_ml_LORENZO_D_AMICO_238684.pdf
└── README.md
```

---

## Notebook Structure (`Baseline_Training_and_Evaluation.ipynb`)

The notebook is a single end-to-end pipeline with five sections:

**1. Baseline** — Zero-shot inference with `Llama-3.2-3B-Instruct` using a two-prompt strategy: the first prompt elicits a structured review (Decision / Comment / Strengths / Weaknesses), the second forces a single-token Accept/Reject verdict via greedy decoding. Outputs saved to `BASELINE_llama32_test20.jsonl`.

**2. Fine-Tuning with LoRA** — QLoRA training on `train_17_18_19.parquet`. Completions are converted to structured JSON targets `{comment, strengths (×2), weaknesses (×2), decision}` before tokenization. Prompt tokens are masked (label = −100); loss is computed on completion tokens only. Best checkpoint selected by validation loss with early stopping. Adapters saved to `lora_json_outputs_es/`.

**3. Testing the Trained Model** — Loads the best LoRA adapter (auto-detected from `trainer_state.json`) and runs inference on `test_2020.parquet`. Outputs saved to `TRAINED_llama32_test20_raw_output.jsonl`.

**4. Evaluation** — Normalization of raw JSONL outputs (`*_NORM.jsonl`), then full evaluation suite: accuracy, macro F1, confusion matrix, token-Jaccard, ROUGE-1/2/L, chrF, SBERT cosine, BERTScore F1. Results written to `review_similarity_metrics.csv` and `rougeL_pairs.csv`.

**5. KL Anchoring Experiment** — Continued training from `lora_json_outputs_es/checkpoint-1000` with KL-divergence regularisation (λ = 0.5, applied to last 256 tokens per sequence). Adapters saved to `lora_json_kl/`. Result: validation loss rose to ~2.50 and perplexity to ~10.1 — this direction was ruled out.

---

## Architecture & Hyperparameters

### Zero-Shot Baseline

| Parameter | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Strategy | Two-prompt: structured review + forced binary decision |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Repetition penalty | 1.15 |
| No-repeat n-gram size | 6 |
| Max new tokens | 220 (review) / 3 (decision) |
| Decision decoding | Greedy |

### QLoRA Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Quantization | 4-bit NF4, double quantization, bfloat16 compute |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 24,313,856 / 1,827,777,536 (~1.33%) |
| Loss | Token-level cross-entropy on completion tokens only |
| Completion format | JSON: `{comment, strengths (×2), weaknesses (×2), decision}` |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Weight decay | 0.1 |
| Warmup ratio | 0.03 |
| Batch size per device | 4 |
| Gradient accumulation | 8 steps (effective batch size = 32) |
| Max sequence length | 1,408 tokens |
| Max epochs | 10 |
| Early stopping patience | 3 (evaluated every 200 steps) |
| Validation split | 8% holdout of 2017–2019, deduplicated by (title, abstract) |
| Best checkpoint | Step 400 (val loss ≈ 2.273) |
| Random seed | 42 |
| Hardware | NVIDIA A40 48 GB |

---

## What is NOT in this Repository (and Why)

| File / Folder | Size | Reason |
|---|---|---|
| `train_17_18_19.parquet` | 14.1 MB | Derived from public data; regenerate with `0_preprocessing_for_training_dataset.py` |
| `test_2020.parquet` | 10.5 MB | Same — regenerate with `1_preprocessing_test_20_parquet.py` |
| `absolute_data/` | — | Raw OpenReview dumps; available at [github.com/Seafoodair/Openreview](https://github.com/Seafoodair/Openreview) |
| `BASELINE_llama32_test20.jsonl` | 5.3 MB | Large model output — on Google Drive |
| `BASELINE_llama32_test20_NORM.jsonl` | 5.4 MB | Same |
| `TRAINED_llama32_test20_raw_output.jsonl` | 4.8 MB | Same |
| `TRAINED_llama32_test20_raw_output_NORM.jsonl` | 4.9 MB | Same |
| `lora_json_outputs_es/` | — | LoRA adapter checkpoints — on Google Drive |
| `lora_json_kl/` | — | KL experiment checkpoints — on Google Drive |
| `.ipynb_checkpoints/` | — | Jupyter auto-generated; never commit |
| `*.bin`, `*.safetensors` | — | Gated model (Meta Llama license); not redistributable |

All excluded files are available on [Google Drive](https://drive.google.com/drive/folders/1XQKbqUhGCq4b5yfLhBnxhAM3o07K3kSm).

---

## Setup & Reproducibility

> ⚠️ **The notebook contains pre-computed outputs and can be read without re-running.** Full re-execution requires access to `meta-llama/Llama-3.2-3B-Instruct` (gated on Hugging Face — request access first) and a GPU with ≥16 GB VRAM (trained on NVIDIA A40 48 GB).

### Dependencies

```bash
pip install torch transformers peft trl bitsandbytes
pip install pandas scikit-learn matplotlib
pip install rouge-score sentence-transformers bert-score
```

### Reproduce Data Preprocessing

```bash
# Step 1: build training parquet from raw OpenReview dumps
python 0_preprocessing_for_training_dataset.py

# Step 2: build test parquet — ICLR 2020, no decision appended (leakage prevention)
python 1_preprocessing_test_20_parquet.py
```

### Run the Notebook

Open `Baseline_Training_and_Evaluation.ipynb` in Jupyter or upload to Google Colab. Run sections in order. To skip re-training, download the pre-computed `.jsonl` files from Google Drive and place them in the project root before running the Evaluation section.

---

## Evaluation Protocol

- All evaluation on **ICLR 2020** (OOD — never seen during training)
- Papers matched by normalised `(title, abstract)` SHA-1 key — strict leakage prevention
- True decision never exposed in the model prompt
- Primary metrics: **macro F1**, per-class precision / recall / F1
- Lexical overlap: token-Jaccard, ROUGE-1/2/L, chrF (on Strengths and Weaknesses sections)
- Semantic similarity: SBERT cosine precision/recall/F1, BERTScore F1
- Diagnostic: validation loss, perplexity (not used for model selection)

---

## Limitations

- Both models fall below a naive always-Reject classifier (69.1% accuracy) — accuracy is misleading given class imbalance
- Results limited to ICLR-style submissions with a 3B backbone; larger models and cross-venue evaluation needed to generalise
- KL experiment not evaluated on the full test set due to GPU budget constraints
- OpenReview editorial decisions may reflect venue-specific norms rather than universal quality criteria

---

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
- Meta AI (2024). *The Llama 3 Herd of Models.*
- Yuan et al. (2022). *Can we automate scientific reviewing?* JAIR, 75, 171–212
- Zhou et al. (2024). *Is LLM a reliable reviewer?* LREC-COLING 2024
- Idahl & Ahmadi (2024). *OpenReviewer.* arXiv:2412.11948
- OpenReview ICLR dumps: [github.com/Seafoodair/Openreview](https://github.com/Seafoodair/Openreview)
