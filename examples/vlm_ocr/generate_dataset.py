"""以虛構商戶資料產生可重現的票據影像與 JSON 標註。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


SAMPLES = [
    ("晴空文具", "INV-2026-001", "2026-10-01", [("筆記本", 2, "18.00"), ("中性筆", 3, "6.50")]),
    ("橘子雲咖啡", "RC-2026-002", "2026-10-02", [("手沖咖啡", 2, "32.00"), ("燕麥餅乾", 1, "15.00")]),
    ("遠山數位", "INV-2026-003", "2026-10-03", [("USB-C 轉接器", 1, "89.00"), ("收納袋", 2, "24.50")]),
    ("青禾生活", "RC-2026-004", "2026-10-04", [("玻璃水瓶", 1, "56.00"), ("棉布袋", 2, "12.00")]),
    ("星河印務", "INV-2026-005", "2026-10-05", [("海報印刷", 10, "8.80"), ("裝訂服務", 1, "35.00")]),
    ("木槿書屋", "RC-2026-006", "2026-10-06", [("資料科學入門", 1, "72.00"), ("索引貼", 2, "9.90")]),
    ("微光烘焙", "RC-2026-007", "2026-10-07", [("檸檬磅蛋糕", 1, "48.00"), ("可可司康", 2, "16.00")]),
    ("北辰辦公", "INV-2026-008", "2026-10-08", [("A4 紙張", 5, "22.00"), ("標籤紙", 2, "14.50")]),
    ("小島植物", "RC-2026-009", "2026-10-09", [("迷迭香盆栽", 2, "38.00"), ("陶盆", 2, "25.00")]),
    ("雲杉器物", "INV-2026-010", "2026-10-10", [("陶瓷杯", 3, "42.00"), ("茶匙", 3, "8.00")]),
    ("晨霧茶舖", "RC-2026-011", "2026-10-11", [("烏龍茶", 1, "128.00"), ("牛皮紙袋", 2, "3.50")]),
    ("回聲工作室", "INV-2026-012", "2026-10-12", [("場地租用", 1, "360.00"), ("設備清潔", 1, "45.00")]),
]


def find_font() -> str:
    candidates = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("找不到支援中文的字型；請安裝 Noto Sans CJK 或指定字型路徑")


def render(index: int, record: dict, target: Path, font_path: str) -> None:
    image = Image.new("RGB", (1240, 1540), "#f9f7f1")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(font_path, 54)
    body = ImageFont.truetype(font_path, 34)
    small = ImageFont.truetype(font_path, 27)
    draw.rectangle((55, 55, 1185, 1485), outline="#242424", width=4)
    draw.text((100, 92), "虛構票據 / SYNTHETIC INVOICE", font=title, fill="#171717")
    draw.line((95, 176, 1145, 176), fill="#333333", width=3)
    lines = [
        f"商戶: {record['merchant']}",
        f"單據編號: {record['document_id']}",
        f"日期: {record['date']}",
        f"幣別: {record['currency']}",
    ]
    y = 220
    for line in lines:
        draw.text((105, y), line, font=body, fill="#202020")
        y += 72
    draw.line((95, 535, 1145, 535), fill="#999999", width=2)
    draw.text((105, 560), "品項明細", font=body, fill="#202020")
    y = 630
    for item in record["items"]:
        text = f"品項: {item['name']} | 數量: {item['quantity']} | 單價: {item['unit_price']} | 金額: {item['amount']}"
        draw.text((105, y), text, font=small, fill="#202020")
        y += 82
    draw.line((95, 850, 1145, 850), fill="#999999", width=2)
    y = 850
    for key, label in (("subtotal", "小計"), ("tax", "稅額"), ("total", "總計")):
        y += 74
        draw.text((770, y), f"{label}: {record[key]}", font=body, fill="#171717")
    draw.text((105, 1370), "本文件由程式隨機種子資料產生，不代表真實交易或個人資訊。", font=small, fill="#555555")
    if index in (3, 8):
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
        draw = ImageDraw.Draw(image)
        draw.line((55, 55, 1185, 1485), fill="#dedbd4", width=1)
    if index == 10:
        image = image.rotate(0.7, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="#f9f7f1")
    image.save(target, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "sample_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_path = find_font()
    annotations = []
    for index, (merchant, document_id, date, items) in enumerate(SAMPLES, start=1):
        normalized_items = [
            {"name": name, "quantity": str(quantity), "unit_price": unit_price, "amount": f"{quantity * float(unit_price):.2f}"}
            for name, quantity, unit_price in items
        ]
        subtotal = sum(float(item["amount"]) for item in normalized_items)
        tax = round(subtotal * 0.13, 2)
        record = {
            "id": f"invoice-{index:02d}",
            "image": f"invoice-{index:02d}.png",
            "merchant": merchant,
            "document_id": document_id,
            "date": date,
            "currency": "CNY",
            "subtotal": f"{subtotal:.2f}",
            "tax": f"{tax:.2f}",
            "total": f"{subtotal + tax:.2f}",
            "items": normalized_items,
        }
        render(index, record, args.out_dir / record["image"], font_path)
        annotations.append(record)
    (args.out_dir / "dataset.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in annotations), encoding="utf-8"
    )
    print(f"samples={len(annotations)} images={args.out_dir}")


if __name__ == "__main__":
    main()
