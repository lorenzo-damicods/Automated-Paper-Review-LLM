## Inference Demo (Batch Scoring)

### 0) Prerequisites
You need access to the base model on Hugging Face (Llama-3.2-3B-Instruct).
Make sure you are logged in (`huggingface-cli login`) and have accepted the model terms.

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_demo.txt
