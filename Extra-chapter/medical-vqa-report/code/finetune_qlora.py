# -*- coding: utf-8 -*-
"""VQA-RAD QLoRA 微调（Qwen2.5-VL-3B-Instruct, 4bit, 6GB 显存约束）

用法:
  python finetune_qlora.py --epochs 1 --max-samples 800 --out adapters/vqa-rad-lora
仅调 LLM 部分 attention/MLP 投影，视觉塔冻结。
"""
import argparse
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_example(ex, processor):
    """把一条 VQA 样本转成 chat 格式，返回 {'messages': ...}（图像由 collator 处理）"""
    closed = ex["question"].strip().lower().startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were ", "has ", "have "))
    if closed:
        q = "Answer with only one word: yes or no.\n\nQuestion: " + ex["question"]
    else:
        q = "Answer in one short phrase, no sentences.\n\nQuestion: " + ex["question"]
    return {
        "image": ex["image"],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--out", type=str, default="adapters/vqa-rad-lora")
    args = ap.parse_args()

    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
    if args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    examples = [build_example(ex, processor) for ex in ds]

    # 手写训练循环（数据含图像，自定义 collator 与 Trainer 兼容性差，直接最简循环）
    import random
    from torch.utils.data import DataLoader

    def collate(batch):
        texts, img_lists = [], []
        for ex in batch:
            msgs = [
                {"role": m["role"], "content": [
                    {"type": "image", "image": ex["image"]} if c["type"] == "image" else {"type": "text", "text": c["text"]}
                    for c in m["content"]
                ]}
                for m in ex["messages"]
            ]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            img_lists.append([ex["image"]])
        enc = processor(text=texts, images=[im for lst in img_lists for im in lst] or None,
                        padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        labels = enc.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # 图像 token 位置也置 -100（Qwen 视觉占位符 id）
        img_tok = getattr(processor, "image_token_id", None)
        if img_tok is not None:
            labels[labels == img_tok] = -100
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask,
                "pixel_values": enc.get("pixel_values"), "image_grid_thw": enc.get("image_grid_thw"),
                "labels": labels}

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    model.train()
    step = 0
    import math
    total_steps = math.ceil(len(loader) * args.epochs / args.grad_accum)
    print(f"loader batches/epoch: {len(loader)}, total optimizer steps ~ {total_steps}")

    from torch.nn.utils.rnn import pad_sequence

    for epoch in range(int(args.epochs) + (1 if args.epochs % 1 else 0)):
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            pixel_values = batch.pop("pixel_values")
            image_grid_thw = batch.pop("image_grid_thw")
            kwargs = {}
            if pixel_values is not None:
                kwargs = {"pixel_values": pixel_values.to(model.device, dtype=torch.bfloat16),
                          "image_grid_thw": image_grid_thw.to(model.device)}
            out = model(input_ids=batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                        labels=batch["labels"].to(model.device), **kwargs)
            (out.loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); optimizer.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {out.loss.item():.4f}")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print("saved adapter to", args.out)


if __name__ == "__main__":
    sys.exit(main())
