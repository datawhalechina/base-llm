# -*- coding: utf-8 -*-
"""VQA-RAD zero-shot VQA 推理与评估（Qwen2.5-VL-3B-Instruct, 4bit）

用法:
  python vqa_inference.py --max-samples 100 --out results/vqa_zeroshot.json
"""
import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_ID = os.environ.get(
    "VQA_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen25vl-3b"),
)


def build_messages(image, question):
    closed = question.strip().lower().startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were ", "has ", "have "))
    if closed:
        prompt = (
            "You are answering questions about a radiology image. "
            "Answer with only one word: yes or no.\n\nQuestion: " + question
        )
    else:
        prompt = (
            "You are answering questions about a radiology image. "
            "Answer in one short phrase, no sentences.\n\nQuestion: " + question
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.inference_mode()
def answer_one(model, processor, image, question, max_new_tokens=16):
    messages = build_messages(image, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def normalize(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def is_correct(pred, gt):
    p, g = normalize(pred), normalize(gt)
    if not p:
        return False
    return p == g or p in g or g in p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/vqa_zeroshot.json")
    ap.add_argument("--adapter", type=str, default=None, help="QLoRA adapter 路径（微调后评估用）")
    args = ap.parse_args()

    ds = load_dataset("flaviagiammarino/vqa-rad", split="test")
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    print(f"eval samples: {len(ds)}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("adapter loaded:", args.adapter)

    records, n_closed, n_closed_ok, n_open, n_open_ok = [], 0, 0, 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        pred = answer_one(model, processor, ex["image"], ex["question"])
        closed = ex["question"].strip().lower().startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were ", "has ", "have "))
        ok = is_correct(pred, ex["answer"])
        if closed:
            n_closed += 1; n_closed_ok += ok
        else:
            n_open += 1; n_open_ok += ok
        records.append({
            "idx": i, "question": ex["question"], "gt": ex["answer"],
            "pred": pred, "closed": closed, "correct": bool(ok),
        })
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(ds)}] closed acc {n_closed_ok}/{n_closed}, open acc {n_open_ok}/{n_open}")

    result = {
        "model": MODEL_ID, "adapter": args.adapter, "n": len(ds),
        "closed_acc": n_closed_ok / max(n_closed, 1), "n_closed": n_closed,
        "open_acc": n_open_ok / max(n_open, 1), "n_open": n_open,
        "overall_acc": (n_closed_ok + n_open_ok) / len(ds),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": result, "records": records}, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
