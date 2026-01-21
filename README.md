# Automated Paper Review with QLoRA Fine-Tuning 

### 📋 Project Overview
This project investigates the potential of **Compact Language Models** (specifically **Llama-3.2 3B**) to assist in the academic peer-review process.
The goal was to automate the initial screening of conference papers by generating structured feedback and predicting an **Accept/Reject** decision based on the paper's content.

Using **QLoRA (Quantized Low-Rank Adaptation)**, I fine-tuned the model on data from ICLR conferences, focusing on minimizing bias and improving decision accuracy on Out-Of-Distribution (OOD) data.

### 🚀 Key Results
* **Accuracy Improvement:** The fine-tuned model achieved an accuracy of **41.7%** on the OOD test set (ICLR 2020), significantly outperforming the zero-shot baseline of **30.9%**.
* **Bias Reduction:** The baseline model was heavily biased towards acceptance (accepting 2199/2203 papers). The QLoRA adaptation successfully corrected this, producing a much more realistic distribution (~1650 Accepts / 546 Rejects).
* **Efficiency:** Fine-tuning updated only **~1.33%** of parameters, making the training computationally feasible while achieving lower validation loss (2.27 vs 3.0).

---

### 🧠 Methodology

#### 1. Data Pipeline & Normalization
* **Source:** **OpenReview** dataset (ICLR 2017-2019 for training, ICLR 2020 for testing).
* **Preprocessing:**
    * Extracted review text and editorial decisions.
    * Binarized decisions into `Accept` / `Reject` labels.
    * Normalized text input for consistency (scripts in `data_preparation/`).

#### 2. Model & Fine-Tuning
* **Base Model:** `Llama-3.2-3B-Instruct` (chosen for its efficiency/performance ratio).
* **Technique:** **QLoRA** (4-bit quantization) using the `PEFT` library.
* **Objective:** Optimized token-level cross-entropy on the review text + decision token.
* **Regularization Experiment:** Tested KL-divergence regularization to prevent mode collapse, though standard QLoRA proved more effective for this specific task.

#### 3. Evaluation
Evaluation was performed strictly **Out-Of-Distribution (OOD)** using papers from ICLR 2020 to test the model's ability to generalize to new research trends. Metrics included Accuracy, Validation Loss, and Perplexity.

---

### 🛠️ Tech Stack
* **LLM Frameworks:** PyTorch, Hugging Face Transformers, BitsAndBytes
* **Fine-Tuning:** PEFT (Parameter-Efficient Fine-Tuning), TRL
* **Analysis:** Pandas, Scikit-learn, Matplotlib
* **Environment:** Jupyter Notebooks

---

### 📂 Repository Structure
* `notebooks/`: Contains the main training and evaluation logic (`Training_and_evaluation.ipynb`).
* `Training_and_Evaluation/`: Scripts used for cleaning and normalizing the OpenReview dataset.
* `Automated_Paper_Rewiew/`: Full academic report with detailed analysis and references.

### 🔮 Future Work
* Integration of **SBERT** to better measure semantic similarity between generated and human reviews.
* Experimenting with **KL-regularization** on larger, cleaner datasets to better control the teacher distribution.
* Expanding the training set to include other top-tier conferences (NeurIPS, ICML).

---
*Author: Lorenzo D'Amico*
