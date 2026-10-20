"""以 Qwen2.5-VL 直接閱讀票據影像並輸出結構化 JSON。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from common import PROMPT, load_records, parse_model_json, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-samples", type=int, default=0, help="0 代表執行全部樣本")
    args = parser.parse_args()

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization,
        attn_implementation="sdpa",
    )
    model.eval()

    records = load_records(args.data_dir / "dataset.jsonl")
    if args.max_samples > 0:
        records = records[: args.max_samples]
    predictions = []
    for index, record in enumerate(records, start=1):
        image_path = args.data_dir / record["image"]
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image", "image": str(image_path),
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 768 * 28 * 28,
                },
                {"type": "text", "text": PROMPT},
            ],
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=14)
        inputs = processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prediction, parseable = parse_model_json(raw)
        predictions.append({"id": record["id"], "raw_text": raw, "json_parseable": parseable, "prediction": prediction})
        print(f"[{index}/{len(records)}] {record['id']} parseable={parseable}")
        del inputs, output_ids, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_jsonl(args.output, predictions)
    print(f"samples={len(predictions)} output={args.output}")


if __name__ == "__main__":
    main()
