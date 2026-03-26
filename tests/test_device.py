import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from translate_gemma_ui.device import DeviceInfo, get_device_info


def test_cuda_device_detection():
    props = SimpleNamespace(total_memory=8 * 1024**3)
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value = props

        info = get_device_info()

    assert info == DeviceInfo(
        device_name="NVIDIA RTX 4090", memory_info="8.00 GB VRAM", is_cpu=False, vram_bytes=8 * 1024**3
    )


def test_cuda_device_has_vram_bytes():
    props = SimpleNamespace(total_memory=4 * 1024**3)
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GTX 3050"
        mock_torch.cuda.get_device_properties.return_value = props

        info = get_device_info()

    assert info.vram_bytes == 4 * 1024**3


def test_mps_device_has_no_vram_bytes():
    system_mem = 16 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        info = get_device_info()

    assert info.vram_bytes is None


def test_cpu_device_has_no_vram_bytes():
    system_mem = 32 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        info = get_device_info()

    assert info.vram_bytes is None


def test_device_info_default_vram_bytes_is_none():
    info = DeviceInfo(device_name="CPU", memory_info="8.00 GB RAM", is_cpu=True)
    assert info.vram_bytes is None


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
    with pytest.raises(AttributeError):
        info.device_name = "GPU"


def test_system_memory_windows_fallback():
    from unittest.mock import MagicMock

    from translate_gemma_ui.device import _get_system_memory_bytes

    mock_windll = MagicMock()
    # Simulate GetPhysicallyInstalledSystemMemory setting value to 16GB in KB
    mock_windll.kernel32.GetPhysicallyInstalledSystemMemory.side_effect = lambda ptr: setattr(
        ptr._obj, "value", 16 * 1024 * 1024
    )

    import ctypes

    mock_ctypes = MagicMock(windll=mock_windll, c_ulonglong=ctypes.c_ulonglong, byref=ctypes.byref)

    with (
        patch("os.sysconf", side_effect=AttributeError("not available on Windows")),
        patch.dict("sys.modules", {"ctypes": mock_ctypes}),
    ):
        result = _get_system_memory_bytes()
        assert result == 16 * 1024 * 1024 * 1024


def test_windows_cpu_fallback_logs_cuda_install_hint(caplog):
    system_mem = 16 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
        patch("translate_gemma_ui.device.sys") as mock_sys,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_sys.platform = "win32"

        with caplog.at_level(logging.WARNING, logger="translate_gemma_ui.device"):
            get_device_info()

    assert any("CUDA" in record.message for record in caplog.records)


def test_force_cpu_via_env_var():
    system_mem = 32 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
        patch.dict("os.environ", {"DEVICE": "cpu"}),
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        info = get_device_info()

    assert info.is_cpu is True
    assert info.device_name == "CPU"
    assert info.forced_cpu is True


def test_force_cpu_env_var_case_insensitive():
    system_mem = 32 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
        patch.dict("os.environ", {"DEVICE": "CPU"}),
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        info = get_device_info()

    assert info.is_cpu is True


def test_force_cpu_no_gpu_sets_forced_cpu_false():
    system_mem = 32 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
        patch.dict("os.environ", {"DEVICE": "cpu"}),
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        info = get_device_info()

    assert info.is_cpu is True
    assert info.forced_cpu is False


def test_device_env_var_non_cpu_does_not_override():
    props = SimpleNamespace(total_memory=8 * 1024**3)
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch.dict("os.environ", {"DEVICE": "cuda"}),
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value = props

        info = get_device_info()

    assert info.is_cpu is False


def test_empty_device_env_var_does_not_override():
    props = SimpleNamespace(total_memory=8 * 1024**3)
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch.dict("os.environ", {"DEVICE": ""}),
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value = props

        info = get_device_info()

    assert info.is_cpu is False


def test_non_windows_cpu_fallback_no_cuda_hint(caplog):
    system_mem = 16 * 1024**3
    with (
        patch("translate_gemma_ui.device.torch") as mock_torch,
        patch("translate_gemma_ui.device._get_system_memory_bytes", return_value=system_mem),
        patch("translate_gemma_ui.device.sys") as mock_sys,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_sys.platform = "darwin"

        with caplog.at_level(logging.WARNING, logger="translate_gemma_ui.device"):
            get_device_info()

    assert not any("CUDA" in record.message for record in caplog.records)
