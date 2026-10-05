import os
import sys

import pytest

import constants
import gpu_pack

on_mac = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


@on_mac
def test_mac_paths_follow_macos_conventions():
    home = os.path.expanduser("~")
    assert constants.APP_DATA_DIR == os.path.join(home, "Library", "Application Support", "Sotvox")
    assert constants.LOG_DIR == os.path.join(home, "Library", "Logs", "Sotvox")
    assert constants.DEFAULT_OUTPUT_DIR == os.path.join(home, "Documents", "sotvox-transcripts")


def test_resources_are_found():
    assert os.path.isdir(constants.ASSETS_DIR)
    for size in (16, 32, 48):
        assert os.path.exists(os.path.join(constants.ASSETS_DIR, f"sotvox_{size}.png"))


@on_mac
def test_gpu_pack_reports_no_cuda_on_mac():
    assert gpu_pack.libraries_available() is False
    assert gpu_pack.detect_nvidia_gpu() is None


@on_mac
def test_cpu_backend_supports_int8():
    import ctranslate2
    assert "int8" in ctranslate2.get_supported_compute_types("cpu")
    assert ctranslate2.get_cuda_device_count() == 0


@on_mac
def test_notification_sounds_exist():
    import ui
    assert os.path.exists(ui.SOUND_SUCCESS)
    assert os.path.exists(ui.SOUND_FAILURE)

