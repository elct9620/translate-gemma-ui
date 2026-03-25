@echo off
chcp 65001 >nul
echo TranslateGemma UI
echo.

echo 檢查 PyTorch CUDA 支援...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %errorlevel% equ 0 (
    echo CUDA 已啟用
) else (
    echo.
    echo [警告] 偵測到 PyTorch 未支援 CUDA。
    echo 若您的電腦有 NVIDIA 顯示卡，可自動重新安裝支援 CUDA 的 PyTorch。
    echo.
    set /p FIX_CUDA="是否重新安裝 CUDA 版 PyTorch？(y/N): "
    if /i "%FIX_CUDA%"=="y" (
        echo.
        echo 正在安裝 CUDA 版 PyTorch，請稍候...
        pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
        echo.
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
        if %errorlevel% equ 0 (
            echo CUDA 版 PyTorch 安裝成功！
        ) else (
            echo [警告] 安裝後仍無法偵測 CUDA，請確認已安裝 NVIDIA 驅動程式。
        )
    )
)
echo.

set /p OFFLINE="是否使用離線模式？(y/N): "
if /i "%OFFLINE%"=="y" (
    echo 啟用離線模式...
    set HF_HUB_OFFLINE=1
) else (
    echo 啟用一般模式...
)

echo.
python main.py
pause
