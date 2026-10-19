"""Grounding DINO 的單句視覺定位與 bbox 指標工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_detector(model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> tuple[Any, Any, torch.device]:
    """載入處理器與模型；HF_HOME 可用來將權重快取放在指定磁碟。"""
    target = choose_device(device)
    is_local_path = Path(model_id).expanduser().exists()
    revision = None if is_local_path else DEFAULT_MODEL_REVISION
    processor_kwargs = {"revision": revision} if revision else {}
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model_kwargs = {"use_safetensors": True}
    if revision:
        model_kwargs["revision"] = revision
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        **model_kwargs,
    ).to(target)
    model.eval()
    return processor, model, target


@torch.inference_mode()
def predict_bbox(
    image: Image.Image,
    text: str,
    processor: Any,
    model: Any,
    device: torch.device,
    threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> dict[str, Any]:
    """回傳最高分候選框，座標為原始影像像素的 xyxy。"""
    image = image.convert("RGB")
    inputs = processor(images=image, text=[text], return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    if len(results["scores"]) == 0:
        return {"bbox": None, "score": 0.0, "label": ""}

    best = int(results["scores"].argmax().item())
    labels = results["text_labels"] if "text_labels" in results else results["labels"]
    return {
        "bbox": [round(float(value), 2) for value in results["boxes"][best].tolist()],
        "score": round(float(results["scores"][best].item()), 4),
        "label": str(labels[best]),
    }


def draw_prediction(image: Image.Image, prediction: dict[str, Any], output_path: str | Path) -> None:
    """在影像上繪製預測框並儲存。"""
    canvas = image.convert("RGB").copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bbox = prediction["bbox"]
    if bbox is not None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(bbox, outline=(255, 48, 48), width=max(2, round(min(canvas.size) / 250)))
        caption = f'{prediction["label"]} {prediction["score"]:.3f}'
        draw.text((bbox[0] + 3, max(0, bbox[1] - 18)), caption, fill=(255, 48, 48))
    canvas.save(output_path)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def box_iou(predicted: list[float] | None, target: list[float]) -> float:
    """計算兩個 xyxy 邊界框的 IoU；缺少預測框時回傳 0。"""
    if predicted is None:
        return 0.0
    px1, py1, px2, py2 = predicted
    tx1, ty1, tx2, ty2 = target
    intersection_width = max(0.0, min(px2, tx2) - max(px1, tx1))
    intersection_height = max(0.0, min(py2, ty2) - max(py1, ty1))
    intersection = intersection_width * intersection_height
    predicted_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = predicted_area + target_area - intersection
    return intersection / union if union > 0 else 0.0
