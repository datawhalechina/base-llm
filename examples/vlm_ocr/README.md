# VLM OCR 與文檔結構化

本範例用同一批 12 張程式生成的虛構票據比較兩種流程：RapidOCR 偵測與辨識後再用標籤解析，以及 Qwen2.5-VL-3B-Instruct 直接看圖輸出 JSON。資料不含真實商戶或個人資訊。

## 環境

建議使用 Python 3.10+。先依 [PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/) 安裝與 CUDA 相符的 `torch` / `torchvision`，再安裝本目錄依賴。8 GB 顯示卡使用 4-bit NF4 推理；模型快取可設定到 D 槽，避免佔用系統碟。

```powershell
Set-Location examples/vlm_ocr
$env:HF_HOME = 'D:\packages\dw-task125-ocr\hf-cache'
python -m pip install -r requirements.txt
```

若已在 D 槽建立虛擬環境，請將 `python` 換成該環境 `Scripts\python.exe` 的完整路徑。

## 執行端到端示範

```powershell
Set-Location examples/vlm_ocr
$env:HF_HOME = 'D:\packages\dw-task125-ocr\hf-cache'
$data = 'D:\packages\dw-task125-ocr\data'
$results = 'D:\packages\dw-task125-ocr\results'

python generate_dataset.py --out-dir $data
python ocr_baseline.py --data-dir $data --output "$results\rapidocr.jsonl"
python vlm_extract.py --data-dir $data --output "$results\qwen_vl.jsonl"
python evaluate.py --ground-truth "$data\dataset.jsonl" `
  --predictions "$results\rapidocr.jsonl" "$results\qwen_vl.jsonl" `
  --names RapidOCR Qwen2.5-VL --output-dir "$results\evaluation"
```

首次執行會從 Hugging Face 下載模型權重，快取位置由 `HF_HOME` 決定。目前 12 張樣本由固定清單及固定稅率生成，每次產生相同的標註與影像。

## 流程與後處理

1. `generate_dataset.py` 產生 PNG 票據和 JSONL 標註，使用明確的欄位標籤及兩種單據編號前綴。
2. `ocr_baseline.py` 用 RapidOCR 得到逐行文字，再以正則式讀取欄位，缺失內容保留空值，不做猜測。
3. `vlm_extract.py` 將圖像與 JSON schema 提示詞送入 Qwen2.5-VL，限制輸出 token 數，以 4-bit NF4 降低顯示卡記憶體需求。
4. `common.py` 移除 JSON 程式碼框，解析物件並把欄位轉成固定字串格式。JSON 無法解析時記錄為失敗；不會用票面合計去反推或覆寫模型輸出。
5. `evaluate.py` 比較 8 個頂層欄位（品項陣列作為一欄），輸出 JSON 可解析率、欄位精確率、整張文件完全匹配率、字元準確率與逐欄失敗明細。

本目錄的 `results/` 收錄兩種方法的逐樣本 JSONL、逐欄指標（`metrics.json`）和可閱讀報告（`report.md`），均由上述指令生成。

## 提示詞設計

提示詞限定固定 JSON schema、指定所有值的型別、要求保留原始小數位、禁止自行重算，且看不清的欄位回傳空字串。這能減少額外說明混入輸出，也讓解析失敗與辨識錯誤可以分開計數。正式文件若有不同欄位或表格版式，需先調整 schema 和提示詞，再用獨立樣本驗證。

## 指標定義與限制

- **欄位準確率**：正確欄位數 / 所有樣本的欄位總數；items 陣列按精確內容比較。
- **文件完全匹配率**：8 個欄位全部正確的文件比例。
- **字元準確率**：`1 - CER`，將各欄位序列化後，以 Levenshtein 編輯距離計算。
- **失敗分析**：`report.md` 逐項列出標註值與預測值；可觀察文字辨識、欄位切分和 JSON 輸出的不同錯誤來源。

這是小型合成資料實驗，不代表真實票據的泛化準確度。評估集與模板固定，且模型可能受影像解析度、字型、傾斜、欄位新增或提示詞影響。應在取得授權且移除敏感資料的真實樣本上另行測試，並人工覆核金額等關鍵欄位。

## 來源

- [Qwen2.5-VL-3B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Transformers Qwen2.5-VL 文件](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_vl)
- [RapidOCR 官方專案](https://github.com/RapidAI/RapidOCR)
