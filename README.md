# TranslateGemma UI

本地翻譯工具，使用開放權重模型 [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) 驅動，提供基於瀏覽器的圖形介面，支援文字翻譯與 SRT 字幕翻譯。

翻譯完全在本地執行，無需依賴雲端服務，保護你的內容隱私。

## 功能特色

- **文字翻譯** — 輸入文字、選擇語言，即時取得翻譯結果
- **SRT 字幕翻譯** — 上傳 SRT 字幕檔，支援整檔模式與批次模式，翻譯後保留原始時間軸
- **詞彙表** — 上傳 CSV 詞彙對照表，確保翻譯用詞一致（支援翻譯前/後替換）
- **自動硬體偵測** — 自動偵測 GPU/CPU 並顯示裝置資訊，無 GPU 時自動退回 CPU 模式

### 支援語言

英語（en）、繁體中文（zh-TW）、日語（ja）

## 系統需求

- Python >= 3.12
- 作業系統：macOS、Windows
- [Hugging Face Token](https://huggingface.co/settings/tokens)（首次下載模型時需要）

## 安裝方式

### 方法一：使用 pip（推薦給一般使用者）

```bash
# 1. 下載專案
git clone https://github.com/user/translate-gemma-ui.git
cd translate-gemma-ui

# 2. 建立虛擬環境（建議）
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. 安裝相依套件
pip install -r requirements.txt
```

> **Windows 使用者提示：** 如果尚未安裝 Python，請從 [python.org](https://www.python.org/downloads/) 下載安裝，安裝時請勾選「Add Python to PATH」。

### 方法二：使用 uv（推薦給開發者）

```bash
# 安裝 uv（如尚未安裝）
# https://docs.astral.sh/uv/getting-started/installation/

# 下載專案並安裝
git clone https://github.com/user/translate-gemma-ui.git
cd translate-gemma-ui
uv sync
```

## 使用說明

### 啟動應用程式

```bash
# 使用 pip 安裝的使用者
python main.py

# 使用 uv 的使用者
uv run python main.py
```

啟動後，瀏覽器會自動開啟 Gradio 介面。如未自動開啟，請手動前往終端機顯示的網址（預設為 `http://127.0.0.1:7860`）。

### 首次使用：取得模型存取權限

TranslateGemma 是 Google 的受限模型，下載前需要先完成以下步驟：

1. 前往 [TranslateGemma 4B 模型頁面](https://huggingface.co/google/translategemma-4b-it)，點擊「Agree and access repository」接受 Google 的使用授權
2. 取得 [Hugging Face Token](https://huggingface.co/settings/tokens)

完成後，在應用程式介面中的 HF Token 欄位輸入權杖，點擊「載入模型」按鈕。模型會下載並快取至本地，後續啟動不需重新下載。

### 文字翻譯

1. 在文字翻譯分頁中輸入要翻譯的文字
2. 選擇來源語言與目標語言
3. 點擊「翻譯」，翻譯結果會以串流方式逐步顯示

### SRT 字幕翻譯

1. 在 SRT 翻譯分頁中上傳 SRT 字幕檔
2. 選擇目標語言與翻譯模式：
   - **整檔模式** — 適合字幕量較少的檔案，一次翻譯完成
   - **批次模式** — 適合較大的字幕檔，可設定每批翻譯的句數
3. （選填）上傳詞彙表 CSV 檔案，並選擇替換方式（翻譯前或翻譯後）
4. 點擊「翻譯」，完成後下載翻譯好的 SRT 檔案

### 詞彙表格式

詞彙表為 CSV 檔案（UTF-8 編碼），無標頭列，每行格式為：

```
來源詞,目標詞
```

範例：

```csv
Hello,你好
World,世界
```

## 開發

```bash
# 執行測試
uv run pytest

# 程式碼檢查
uv run ruff check .

# 程式碼格式化
uv run ruff format .
```

## 授權條款

本專案採用 [MIT License](LICENSE) 授權。
