from types import SimpleNamespace
from unittest.mock import patch

from translate_gemma_ui.device import DeviceInfo, get_device_info


def test_cuda_device_detection():
    props = SimpleNamespace(total_mem=8 * 1024**3)
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value = props

        info = get_device_info()

    assert info == DeviceInfo(device_name="NVIDIA RTX 4090", memory_info="8.00 GB VRAM", is_cpu=False)


def test_mps_device_detection():
    system_mem = 16 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        info = get_device_info()

    assert info == DeviceInfo(device_name="Apple Silicon GPU", memory_info="16.00 GB RAM", is_cpu=False)


def test_cpu_fallback():
    system_mem = 32 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        info = get_device_info()

    assert info == DeviceInfo(device_name="CPU", memory_info="32.00 GB RAM", is_cpu=True)


def test_device_info_is_frozen():
    info = DeviceInfo(device_name="CPU", memory_info="8.00 GB RAM", is_cpu=True)
    try:
        info.device_name = "GPU"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
