"""欄位精確率、文件完全匹配率、字元準確率與錯誤明細。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import FIELDS, canonical, load_records


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {row["id"]: row for row in predictions}
    counts = {field: {"correct": 0, "total": 0} for field in FIELDS}
    correct_docs = 0
    char_errors = 0
    char_total = 0
    parseable = 0
    failures = []
    for expected in ground_truth:
        result = indexed.get(expected["id"], {})
        actual = result.get("prediction", {})
        parseable += int(bool(result.get("json_parseable", bool(actual))))
        doc_correct = True
        for field in FIELDS:
            expected_text = canonical(expected[field])
            actual_text = canonical(actual.get(field, [] if field == "items" else ""))
            counts[field]["total"] += 1
            is_correct = expected_text == actual_text
            counts[field]["correct"] += int(is_correct)
            doc_correct &= is_correct
            char_errors += edit_distance(expected_text, actual_text)
            char_total += len(expected_text)
            if not is_correct:
                failures.append({"id": expected["id"], "field": field, "expected": expected[field], "actual": actual.get(field)})
        correct_docs += int(doc_correct)
    field_rates = {key: value["correct"] / value["total"] if value["total"] else 0.0 for key, value in counts.items()}
    total_fields = sum(value["total"] for value in counts.values())
    total_correct = sum(value["correct"] for value in counts.values())
    return {
        "sample_count": len(ground_truth),
        "parseable_json_rate": parseable / len(ground_truth) if ground_truth else 0.0,
        "field_accuracy": total_correct / total_fields if total_fields else 0.0,
        "field_accuracy_by_name": field_rates,
        "document_exact_match_rate": correct_docs / len(ground_truth) if ground_truth else 0.0,
        "character_accuracy": 1 - char_errors / char_total if char_total else 0.0,
        "character_error_rate": char_errors / char_total if char_total else 0.0,
        "character_count": char_total,
        "field_counts": counts,
        "failure_examples": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True, help="可同時比較兩個以上的 JSONL")
    parser.add_argument("--names", nargs="*", help="每個預測檔的顯示名稱")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    truth = load_records(args.ground_truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = args.names or [path.stem for path in args.predictions]
    if len(names) != len(args.predictions):
        raise ValueError("--names 數量必須與 --predictions 一致")
    results = {name: evaluate(truth, load_records(path)) for name, path in zip(names, args.predictions)}
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# OCR / VLM 評估結果", "", f"- 樣本數：{len(truth)} 張虛構票據", "- 欄位數：每張 8 個頂層欄位（品項陣列視為一欄）", "- 字元準確率：以各欄位序列化後的 Levenshtein 編輯距離計算，`1 - CER`。", "", "| 方法 | JSON 可解析率 | 欄位準確率 | 文件完全匹配 | 字元準確率 |", "|---|---:|---:|---:|---:|"]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['parseable_json_rate']:.2%} | {metrics['field_accuracy']:.2%} "
            f"| {metrics['document_exact_match_rate']:.2%} | {metrics['character_accuracy']:.2%} |"
        )
    for name, metrics in results.items():
        lines.extend(
            [
                "",
                f"## {name} 各欄位準確率",
                "",
                "| 欄位 | 命中 | 總數 | 準確率 |",
                "|---|---:|---:|---:|",
            ]
        )
        for field, counts in metrics["field_counts"].items():
            rate = counts["correct"] / counts["total"] if counts["total"] else 0.0
            lines.append(
                f"| {field} | {counts['correct']} | {counts['total']} | {rate:.2%} |"
            )
    for name, metrics in results.items():
        lines.extend(["", f"## {name} 失敗欄位", ""])
        if not metrics["failure_examples"]:
            lines.append("本次樣本未出現欄位不匹配。")
        else:
            lines.extend(["| 樣本 | 欄位 | 標註值 | 預測值 |", "|---|---|---|---|"])
            for failure in metrics["failure_examples"]:
                expected = json.dumps(failure["expected"], ensure_ascii=False)
                actual = json.dumps(failure["actual"], ensure_ascii=False)
                lines.append(f"| {failure['id']} | {failure['field']} | `{expected}` | `{actual}` |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
