"""下載 RefCOCOg validation parquet，抽樣並取得 COCO 影像。"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PARQUET_URL = (
    "https://huggingface.co/datasets/jhkwak-bp/refcocog-coco2017/resolve/702f10ca47c4d9b5a6479f180eabbe113d26bf6f/"
    "data/validation-00000-of-00001.parquet?download=true"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "base-llm-visual-grounding/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
        output.write(response.read())


def sample_fields(row: dict[str, Any]) -> tuple[str, str, list[float], str, str] | None:
    sentence = str(row.get("sentence") or row.get("raw_sentence") or "").strip()
    annotation = row.get("annotation") or {}
    image_info = row.get("image_info") or {}
    bbox = annotation.get("bbox") if isinstance(annotation, dict) else None
    image_path = str(row.get("image_path") or image_info.get("file_name") or "")
    match = re.search(r"(?:COCO_)?(train|val)2017[_/](\d{12})\.jpg$", image_path)
    if not match or len(bbox or []) != 4 or not sentence:
        return None
    split, image_id = match.groups()
    try:
        box = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if box[2] <= 0 or box[3] <= 0:
        return None
    return f"{split}2017/{image_id}.jpg", sentence, box, image_id, str(row.get("split", "validation"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/refcocog-mini"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parquet", type=Path, help="已下載的 validation parquet；省略時自動下載")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples 必須大於 0")

    output_dir = args.output_dir.resolve()
    parquet_path = args.parquet or output_dir / "refcocog-validation.parquet"
    if not parquet_path.exists():
        print(f"下載 RefCOCOg validation annotations：{PARQUET_URL}")
        download_file(PARQUET_URL, parquet_path)

    rows = pq.read_table(parquet_path).to_pylist()
    candidates = [candidate for row in rows if (candidate := sample_fields(row)) is not None]
    by_image: dict[str, tuple[str, str, list[float], str, str]] = {}
    for candidate in candidates:
        by_image.setdefault(candidate[3], candidate)
    unique_candidates = sorted(by_image.values(), key=lambda item: (item[3], item[1]))
    selected = random.Random(args.seed).sample(unique_candidates, min(args.samples, len(unique_candidates)))
    if not selected:
        raise RuntimeError("parquet 中找不到含 COCO 2017 路徑、bbox 與句子的樣本")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (relative_url_path, sentence, bbox, image_id, source_split) in enumerate(selected, start=1):
            filename = Path(relative_url_path).name
            image_path = image_dir / filename
            if not image_path.exists():
                # Path-style S3 keeps TLS verification valid for the dotted bucket name.
                url = f"https://s3.amazonaws.com/images.cocodataset.org/{relative_url_path}"
                print(f"[{index}/{len(selected)}] 下載影像 {filename}")
                download_file(url, image_path)
            record = {
                "image": f"images/{filename}",
                "image_id": image_id,
                "sentence": sentence,
                "bbox_xywh": bbox,
                "source_split": source_split,
                "annotation_source": "RefCOCOg validation, jhkwak-bp/refcocog-coco2017",
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"完成：{len(selected)} 筆樣本，manifest={manifest_path}")


if __name__ == "__main__":
    main()
