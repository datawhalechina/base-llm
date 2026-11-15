# -*- coding: utf-8 -*-
"""IU-Xray 报告生成（zero-shot）与 BLEU-4 / ROUGE-L 评估

用法:
  python report_inference.py --max-samples 30 --out results/report_zeroshot.json
前提: IU-Xray image.zip 已解压，IU_XRAY_IMAGE_DIR 指向图片根目录
"""
import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_ID = os.environ.get(
    "VQA_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen25vl-3b"),
)
PROMPT = (
    "You are a radiologist writing a chest X-ray report. Based on the chest X-ray images, "
    "write a concise radiology report covering: heart size, mediastinum, lungs, pleura, and any "
    "abnormal findings. Write 3-6 short sentences in the style of a Findings section. "
    "Do not include patient identity information."
)

# 非临床用途免责声明（评估输出中保留）
DISCLAIMER = "For research/demo use only. NOT for clinical use."


def load_iuxray_test(jsonl_path, image_dir):
    """读 test.jsonl，把 images 相对路径映射到本地解压目录。"""
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            # query 形如 "<image><image>Please review ..." —— 去掉占位符得到指令
            instruction = re.sub(r"<image>", "", ex["query"]).strip()
            # response 可能带 'Findings:...' 前缀结构，保留全文作为参考报告
            ref = ex["response"].strip()
            imgs = []
            for rel in ex["images"]:
                # rel 形如 /iu_xray/image/CXR2384_IM-0942/0.png
                # zip 内实际路径 .../iu_xray/iu_xray/images/<CXRxxx>/<N>.png
                parts = [p for p in rel.split("/") if p]
                cand = os.path.join(image_dir, parts[-2], parts[-1]) if len(parts) >= 2 else os.path.join(image_dir, parts[-1])
                if not os.path.exists(cand):
                    cand = os.path.join(image_dir, *parts)
                imgs.append(cand)
            imgs = [p for p in imgs if os.path.exists(p)]
            if imgs:
                items.append({"instruction": instruction, "reference": ref, "images": imgs})
    return items


@torch.inference_mode()
def generate_report(model, processor, images, max_new_tokens=256):
    content = [{"type": "image", "image": p} for p in images]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="data/iu-xray/test.jsonl")
    ap.add_argument("--image-dir", type=str, default=os.environ.get("IU_XRAY_IMAGE_DIR", "data/iu-xray/images"))
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--out", type=str, default="results/report_zeroshot.json")
    ap.add_argument("--adapter", type=str, default=None)
    args = ap.parse_args()

    items = load_iuxray_test(args.jsonl, args.image_dir)
    if args.max_samples < len(items):
        items = items[: args.max_samples]
    print(f"report samples: {len(items)}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("adapter loaded:", args.adapter)

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    records, bleus, rouges = [], [], []
    t0 = time.time()
    for i, ex in enumerate(items):
        try:
            hyp = generate_report(model, processor, ex["images"])
        except Exception as e:
            hyp = ""
            print(f"[{i}] generate error: {e}")
        ref = ex["reference"]
        if hyp and ref:
            bleus.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth))
            rouges.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
        records.append({"idx": i, "reference": ref, "generated": hyp})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(items)}] BLEU-4 {sum(bleus)/max(len(bleus),1):.4f}, ROUGE-L {sum(rouges)/max(len(rouges),1):.4f}")

    result = {
        "model": MODEL_ID, "adapter": args.adapter, "n": len(items),
        "bleu4": sum(bleus) / max(len(bleus), 1),
        "rougeL_f": sum(rouges) / max(len(rouges), 1),
        "elapsed_sec": round(time.time() - t0, 1),
        "disclaimer": DISCLAIMER,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": result, "records": records}, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
