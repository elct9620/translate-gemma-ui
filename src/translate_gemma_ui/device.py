import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device_name: str
    memory_info: str
    is_cpu: bool


def _format_memory(total_bytes: int, label: str) -> str:
    return f"{total_bytes / (1024**3):.2f} GB {label}"


def _get_system_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def get_device_info() -> DeviceInfo:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_name=torch.cuda.get_device_name(0),
            memory_info=_format_memory(props.total_mem, "VRAM"),
            is_cpu=False,
        )

    if torch.backends.mps.is_available():
        return DeviceInfo(
            device_name="Apple Silicon GPU",
            memory_info=_format_memory(_get_system_memory_bytes(), "RAM"),
            is_cpu=False,
        )

    return DeviceInfo(
        device_name="CPU",
        memory_info=_format_memory(_get_system_memory_bytes(), "RAM"),
        is_cpu=True,
    )
