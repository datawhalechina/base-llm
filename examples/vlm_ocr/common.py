"""合成票據 OCR/VLM 示範共用邏輯。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIELDS = ("merchant", "document_id", "date", "currency", "subtotal", "tax", "total", "items")
ITEM_FIELDS = ("name", "quantity", "unit_price", "amount")

PROMPT = """請閱讀這張虛構票據，僅輸出一個可解析的 JSON 物件，不要 Markdown 程式碼框或解說。
欄位必須是：merchant、document_id、date、currency、subtotal、tax、total、items。
除 items 外的值都輸出字串；items 是物件陣列，每個物件包含 name、quantity、unit_price、amount，值均為字串。
保留票面原文與數字小數位；看不清楚的值填空字串，不要猜。金額不得自行重算。
範例：{"merchant":"晴空文具","document_id":"INV-2026-001","date":"2026-10-01","currency":"CNY","subtotal":"10.00","tax":"1.30","total":"11.30","items":[{"name":"筆記本","quantity":"1","unit_price":"10.00","amount":"10.00"}]}"""


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def extract_ocr_fields(text: str) -> dict[str, Any]:
    """以明確標籤解析傳統 OCR 文字；不對缺失欄位做推測。"""
    normalized = text.replace("：", ":").replace("｜", "|")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    result: dict[str, Any] = {key: "" for key in FIELDS if key != "items"}
    result["items"] = []
    labels = {
        "merchant": r"(?:商(?:戶|户)|Merchant)\s*:\s*(.+)",
        "document_id": r"(?:(?:單|单)據(?:編|编)(?:號|号)|Document ID)\s*:\s*([A-Z0-9-]+)",
        "date": r"(?:日期|Date)\s*:\s*(\d{4}-\d{2}-\d{2})",
        "currency": r"(?:(?:幣|币)(?:別|别)|Currency)\s*:\s*([A-Z]{3})",
        "subtotal": r"(?:小(?:計|计)|Subtotal)\s*:\s*([\d,]+\.\d{2})",
        "tax": r"(?:(?:稅|税)(?:額|额)|Tax)\s*:\s*([\d,]+\.\d{2})",
        "total": r"(?:(?:總|总)(?:計|计)|Total)\s*:\s*([\d,]+\.\d{2})",
    }
    for key, pattern in labels.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip().rstrip("|").strip()
    item_pattern = re.compile(
        r"(?:品(?:項|项)|Item)\s*:\s*(.*?)\s*\|\s*(?:(?:數|数)量|Qty)\s*:\s*(\d+)\s*\|\s*"
        r"(?:(?:單|单)(?:價|价)|Unit price)\s*:\s*([\d,]+\.\d{2})\s*\|\s*"
        r"(?:金(?:額|额)|Amount)\s*:\s*([\d,]+\.\d{2})",
        flags=re.IGNORECASE,
    )
    for match in item_pattern.finditer(normalized):
        result["items"].append(dict(zip(ITEM_FIELDS, (part.strip() for part in match.groups()))))
    return result


def parse_model_json(raw: str) -> tuple[dict[str, Any], bool]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}, False
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}, False
    if not isinstance(value, dict):
        return {}, False
    output: dict[str, Any] = {key: str(value.get(key, "")).strip() for key in FIELDS if key != "items"}
    items = value.get("items", [])
    if not isinstance(items, list):
        items = []
    output["items"] = [
        {key: str(item.get(key, "")).strip() for key in ITEM_FIELDS}
        for item in items if isinstance(item, dict)
    ]
    return output, True


def canonical(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip()

\n