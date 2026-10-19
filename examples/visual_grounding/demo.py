"""對一張影像執行 Grounding DINO 指代表達定位。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from grounding import draw_prediction, load_detector, predict_bbox


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="本機影像路徑")
    parser.add_argument("--text", required=True, help="要定位的自然語言指代，建議使用英文")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("grounding-result.jpg"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    processor, model, device = load_detector(args.model_id, args.device)
    prediction = predict_bbox(
        image,
        args.text,
        processor,
        model,
        device,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
    )
    draw_prediction(image, prediction, args.output)
    print(json.dumps({"text": args.text, **prediction, "device": str(device), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
