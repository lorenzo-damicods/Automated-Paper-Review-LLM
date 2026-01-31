import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(paper_text: str) -> str:
    # Se nel training avevi un template diverso, possiamo rimpiazzarlo 1:1.
    return (
        "You are an academic reviewer.\n"
        "Task: provide structured feedback and end with a single decision token: ACCEPT or REJECT.\n\n"
        "Paper:\n"
        f"{paper_text}\n\n"
        "Return the following format:\n"
        "Summary:\n"
        "- ...\n"
        "Strengths:\n"
        "- ...\n"
        "Weaknesses:\n"
        "- ...\n"
        "Decision: <ACCEPT|REJECT>\n"
    )


@torch.no_grad()
def generate_one(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_decision(text: str) -> str:
    marker = "Decision:"
    i = text.rfind(marker)
    if i == -1:
        return "UNKNOWN"
    tail = text[i + len(marker):].strip().upper()
    if "ACCEPT" in tail:
        return "ACCEPT"
    if "REJECT" in tail:
        return "REJECT"
    return "UNKNOWN"


def load_model_with_adapter(base_model: str, adapter_dir: str, load_4bit: bool):
    # Tokenizer dal base model (in genere l’adapter non contiene tokenizer completo)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}
    if load_4bit:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = qcfg
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter_dir", default="lora_json_outputs_es")
    ap.add_argument("--input_json", default="demo/sample_papers.json")
    ap.add_argument("--output_json", default="outputs/predictions.json")
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    tokenizer, model = load_model_with_adapter(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        load_4bit=args.load_4bit
    )

    items = read_json(args.input_json)
    if not isinstance(items, list):
        raise RuntimeError("Input JSON must be a list of objects: [{paper_id, paper_text}, ...]")

    results: List[Dict[str, Any]] = []
    for it in items:
        paper_id = it.get("paper_id", "")
        paper_text = it.get("paper_text", "")

        prompt = build_prompt(paper_text)
        gen = generate_one(tokenizer, model, prompt, args.max_new_tokens)
        decision = extract_decision(gen)

        results.append({
            "paper_id": paper_id,
            "decision": decision,
            "generated_review": gen
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.output_json} | n={len(results)}")


if __name__ == "__main__":
    main()
