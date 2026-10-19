"""用 RefCOCOg 小樣本計算平均 IoU 與 Acc@0.5。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from grounding import DEFAULT_MODEL_ID, box_iou, load_detector, predict_bbox, xywh_to_xyxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/refcocog-mini"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path, help="選填；將逐筆預測與指標寫入 JSON")
    args = parser.parse_args()

    manifest = args.dataset_dir / "samples.jsonl"
    if not manifest.exists():
        parser.error(f"找不到 {manifest}；請先執行 prepare_refcocog_subset.py")
    samples = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        parser.error("樣本清單為空")

    processor, model, device = load_detector(args.model_id, args.device)
    details: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image_path = args.dataset_dir / sample["image"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        prediction = predict_bbox(
            image,
            sample["sentence"],
            processor,
            model,
            device,
            threshold=args.threshold,
            text_threshold=args.text_threshold,
        )
        iou = box_iou(prediction["bbox"], xywh_to_xyxy(sample["bbox_xywh"]))
        result = {
            "image_id": sample["image_id"],
            "sentence": sample["sentence"],
            "ground_truth_xywh": sample["bbox_xywh"],
            "prediction_xyxy": prediction["bbox"],
            "score": prediction["score"],
            "matched_text": prediction["label"],
            "iou": round(iou, 4),
            "acc_at_0_5": iou >= 0.5,
        }
        details.append(result)
        print(f"[{index}/{len(samples)}] IoU={iou:.3f} Acc@0.5={iou >= 0.5} | {sample['sentence']}")

    metrics = {
        "model_id": args.model_id,
        "device": str(device),
        "threshold": args.threshold,
        "text_threshold": args.text_threshold,
        "sample_count": len(details),
        "mean_iou": sum(item["iou"] for item in details) / len(details),
        "acc_at_0_5": sum(item["acc_at_0_5"] for item in details) / len(details),
        "no_detection_count": sum(item["prediction_xyxy"] is None for item in details),
        "samples": details,
    }
    print(json.dumps({key: value for key, value in metrics.items() if key != "samples"}, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"逐筆結果已寫入：{args.output}")


if __name__ == "__main__":
    main()
