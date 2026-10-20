"""RapidOCR 傳統辨識基線：影像 -> OCR 文字 -> 標籤解析 -> JSON。"""

from __future__ import annotations

import argparse
from pathlib import Path

from rapidocr import RapidOCR

from common import extract_ocr_fields, load_records, save_jsonl


def result_text(result) -> str:
    texts = getattr(result, "txts", None)
    if texts is None and isinstance(result, dict):
        texts = result.get("txts", [])
    if texts is None:
        return str(result)
    return "\n".join(str(text) for text in texts if text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    engine = RapidOCR()
    for record in load_records(args.data_dir / "dataset.jsonl"):
        raw = result_text(engine(str(args.data_dir / record["image"])))
        rows.append({"id": record["id"], "raw_text": raw, "prediction": extract_ocr_fields(raw)})
    save_jsonl(args.output, rows)
    print(f"samples={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
