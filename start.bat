@echo off
chcp 65001 >nul
echo TranslateGemma UI
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
