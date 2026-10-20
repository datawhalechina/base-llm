# OCR / VLM 評估結果

- 樣本數：12 張虛構票據
- 欄位數：每張 8 個頂層欄位（品項陣列視為一欄）
- 字元準確率：以各欄位序列化後的 Levenshtein 編輯距離計算，`1 - CER`。

| 方法 | JSON 可解析率 | 欄位準確率 | 文件完全匹配 | 字元準確率 |
|---|---:|---:|---:|---:|
| RapidOCR + label parsing | 100.00% | 97.92% | 83.33% | 99.86% |
| Qwen2.5-VL-3B-Instruct | 100.00% | 100.00% | 100.00% | 100.00% |

## RapidOCR + label parsing 各欄位準確率

| 欄位 | 命中 | 總數 | 準確率 |
|---|---:|---:|---:|
| merchant | 12 | 12 | 100.00% |
| document_id | 12 | 12 | 100.00% |
| date | 12 | 12 | 100.00% |
| currency | 12 | 12 | 100.00% |
| subtotal | 12 | 12 | 100.00% |
| tax | 12 | 12 | 100.00% |
| total | 12 | 12 | 100.00% |
| items | 10 | 12 | 83.33% |

## Qwen2.5-VL-3B-Instruct 各欄位準確率

| 欄位 | 命中 | 總數 | 準確率 |
|---|---:|---:|---:|
| merchant | 12 | 12 | 100.00% |
| document_id | 12 | 12 | 100.00% |
| date | 12 | 12 | 100.00% |
| currency | 12 | 12 | 100.00% |
| subtotal | 12 | 12 | 100.00% |
| tax | 12 | 12 | 100.00% |
| total | 12 | 12 | 100.00% |
| items | 12 | 12 | 100.00% |

## RapidOCR + label parsing 失敗欄位

| 樣本 | 欄位 | 標註值 | 預測值 |
|---|---|---|---|
| invoice-01 | items | `[{"name": "筆記本", "quantity": "2", "unit_price": "18.00", "amount": "36.00"}, {"name": "中性筆", "quantity": "3", "unit_price": "6.50", "amount": "19.50"}]` | `[{"name": "笔記本", "quantity": "2", "unit_price": "18.00", "amount": "36.00"}, {"name": "中性笔", "quantity": "3", "unit_price": "6.50", "amount": "19.50"}]` |
| invoice-05 | items | `[{"name": "海報印刷", "quantity": "10", "unit_price": "8.80", "amount": "88.00"}, {"name": "裝訂服務", "quantity": "1", "unit_price": "35.00", "amount": "35.00"}]` | `[{"name": "海報印刷", "quantity": "10", "unit_price": "8.80", "amount": "88.00"}, {"name": "裝訂服务", "quantity": "1", "unit_price": "35.00", "amount": "35.00"}]` |

## Qwen2.5-VL-3B-Instruct 失敗欄位

本次樣本未出現欄位不匹配。
