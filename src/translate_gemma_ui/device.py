import logging
import os
import sys
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device_name: str
    memory_info: str
    is_cpu: bool
    vram_bytes: int | None = None


def _format_memory(total_bytes: int, label: str) -> str:
    return f"{total_bytes / (1024**3):.2f} GB {label}"


def _get_system_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError):
        # Windows: os.sysconf is not available
        import ctypes

        kernel32 = ctypes.windll.kernel32
        mem_status = ctypes.c_ulonglong()
        kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_status))
        return mem_status.value * 1024


def get_device_info() -> DeviceInfo:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_name=torch.cuda.get_device_name(0),
            memory_info=_format_memory(props.total_mem, "VRAM"),
            is_cpu=False,
            vram_bytes=props.total_mem,
        )

    if torch.backends.mps.is_available():
        return DeviceInfo(
            device_name="Apple Silicon GPU",
            memory_info=_format_memory(_get_system_memory_bytes(), "RAM"),
            is_cpu=False,
        )

    if sys.platform == "win32":
        logger.warning(
            "偵測到 Windows 系統但 CUDA 無法使用。"
            "您可能安裝了 CPU 版本的 PyTorch。"
            "請使用 start.bat 啟動程式以自動修復，"
            "或執行: pip install torch --extra-index-url https://download.pytorch.org/whl/cu128"
        )

    return DeviceInfo(
        device_name="CPU",
        memory_info=_format_memory(_get_system_memory_bytes(), "RAM"),
        is_cpu=True,
    )
