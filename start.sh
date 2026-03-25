#!/usr/bin/env bash
echo "TranslateGemma UI"
echo

read -rp "是否使用離線模式？(y/N): " OFFLINE

if [[ "${OFFLINE,,}" == "y" ]]; then
    echo "啟用離線模式..."
    export HF_HUB_OFFLINE=1
else
    echo "啟用一般模式..."
fi

echo
exec python main.py
