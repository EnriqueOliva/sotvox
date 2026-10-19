"""Unit tests for ui.SotvoxApp._resolve_device, the CPU/CUDA selection gate."""

import re
import sys

import pytest

import ui

CUDA_MODULE_NAME = "ctranslate2"

AUTO_DEVICE_LABELS = (
    "Auto",
    "auto",
    "AUTO",
    "",
    "Automatic",
    "GPU",
    "gpu (cuda)",
    "cpu",
    " CPU",
    "CPU ",
    "GPU (Cuda)",
)


class StubCTranslate2:
    def __init__(self, device_count=0, error=None):
        self.device_count = device_count
        self.error = error
        self.device_count_calls = 0

    def get_cuda_device_count(self):
        self.device_count_calls += 1
        if self.error is not None:
            raise self.error
        return self.device_count


class StubCTranslate2WithoutDeviceCount:
    def __init__(self):
        self.device_count_calls = 0


class RecordingLibraryProbe:
    def __init__(self, available):
        self.available = available
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        return self.available


def session_log_text(app):
    return "\n".join(app._session_log)


def install_library_probe(monkeypatch, available):
    probe = RecordingLibraryProbe(available)
    monkeypatch.setattr(ui.gpu_pack, "libraries_available", probe)
    return probe


def install_cuda_module(monkeypatch, module):
    monkeypatch.setitem(sys.modules, CUDA_MODULE_NAME, module)
    return module


class TestExplicitCpuSelection:
    def test_returns_cpu_with_int8_quantisation(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("CPU")

        assert app._resolve_device() == ("cpu", "int8")

    def test_never_consults_ctranslate2_even_when_cuda_devices_exist(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        stub = install_cuda_module(monkeypatch, StubCTranslate2(device_count=8))
        app.device_var.set("CPU")

        result = app._resolve_device()

        assert result == ("cpu", "int8")
        assert stub.device_count_calls == 0

    def test_returns_cpu_without_importing_ctranslate2_at_all(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, None)
        app.device_var.set("CPU")

        result = app._resolve_device()

        assert result == ("cpu", "int8")
        assert "CUDA detection failed" not in session_log_text(app)

    def test_records_that_the_user_asked_for_cpu(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("CPU")

        app._resolve_device()

        assert "Device resolved: CPU (user selected)" in session_log_text(app)

    def test_leaves_the_visible_log_untouched(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("CPU")

        app._resolve_device()

        assert app.log_text.value == ""

    def test_probes_the_gpu_libraries_before_branching_even_for_cpu(self, app, monkeypatch):
        probe = install_library_probe(monkeypatch, available=False)
        app.device_var.set("CPU")

        app._resolve_device()

        assert probe.call_count == 1

    def test_adds_exactly_one_breadcrumb(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("CPU")

        app._resolve_device()

        assert len(app._session_log) == 1

    def test_breadcrumb_carries_a_wall_clock_timestamp(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("CPU")

        app._resolve_device()

        assert re.match(r"^\[\d{2}:\d{2}:\d{2}\] Device resolved: CPU", app._session_log[0])


class TestExplicitGpuSelectionWithLibrariesReady:
    def test_returns_cuda_with_float16_quantisation(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("GPU (CUDA)")

        assert app._resolve_device() == ("cuda", "float16")

    def test_does_not_count_cuda_devices_because_the_user_already_chose(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        stub = install_cuda_module(monkeypatch, StubCTranslate2(device_count=0))
        app.device_var.set("GPU (CUDA)")

        result = app._resolve_device()

        assert result == ("cuda", "float16")
        assert stub.device_count_calls == 0

    def test_records_that_the_user_asked_for_cuda(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert "Device resolved: CUDA (user selected)" in session_log_text(app)

    def test_says_nothing_to_the_user_when_the_request_is_honoured(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert app.log_text.value == ""


class TestExplicitGpuSelectionWithLibrariesMissing:
    def test_falls_back_to_cpu_with_int8_quantisation(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app.device_var.set("GPU (CUDA)")

        assert app._resolve_device() == ("cpu", "int8")

    def test_does_not_count_cuda_devices_before_giving_up(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        stub = install_cuda_module(monkeypatch, StubCTranslate2(device_count=4))
        app.device_var.set("GPU (CUDA)")

        result = app._resolve_device()

        assert result == ("cpu", "int8")
        assert stub.device_count_calls == 0

    def test_records_the_missing_libraries_in_the_session_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert "GPU requested but CUDA libraries are not installed — using CPU" in session_log_text(app)

    def test_tells_the_user_to_click_enable_gpu_acceleration(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert app.log_text.value == (
            "GPU acceleration is not installed yet. Using CPU — "
            "click 'Enable GPU Acceleration...' to enable it.\n"
        )

    def test_user_facing_line_is_mirrored_into_the_session_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert "Enable GPU Acceleration" in session_log_text(app)
        assert len(app._session_log) == 2

    def test_visible_log_line_is_scrolled_into_view(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert app.log_text.see_calls == ["end"]

    def test_log_widget_is_left_disabled_after_writing(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()

        assert app.log_text.cget("state") == "disabled"


class TestAutomaticDetection:
    @pytest.mark.parametrize(
        ("device_count", "libraries_ready", "expected"),
        [
            (0, False, ("cpu", "int8")),
            (0, True, ("cpu", "int8")),
            (1, False, ("cpu", "int8")),
            (1, True, ("cuda", "float16")),
            (2, True, ("cuda", "float16")),
            (8, True, ("cuda", "float16")),
            (-1, True, ("cpu", "int8")),
        ],
    )
    def test_device_matrix(self, app, monkeypatch, device_count, libraries_ready, expected):
        install_library_probe(monkeypatch, available=libraries_ready)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=device_count))
        app.device_var.set("Auto")

        assert app._resolve_device() == expected

    @pytest.mark.parametrize("label", AUTO_DEVICE_LABELS)
    def test_any_label_other_than_the_two_exact_choices_auto_detects(self, app, monkeypatch, label):
        install_library_probe(monkeypatch, available=True)
        stub = install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))
        app.device_var.set(label)

        result = app._resolve_device()

        assert result == ("cuda", "float16")
        assert stub.device_count_calls == 1

    def test_counts_cuda_devices_exactly_once(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        stub = install_cuda_module(monkeypatch, StubCTranslate2(device_count=3))
        app.device_var.set("Auto")

        app._resolve_device()

        assert stub.device_count_calls == 1

    def test_probes_the_gpu_libraries_exactly_once(self, app, monkeypatch):
        probe = install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))
        app.device_var.set("Auto")

        app._resolve_device()

        assert probe.call_count == 1

    @pytest.mark.parametrize("libraries_ready", [True, False])
    def test_records_the_detection_outcome_in_the_session_log(self, app, monkeypatch, libraries_ready):
        install_library_probe(monkeypatch, available=libraries_ready)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=2))
        app.device_var.set("Auto")

        app._resolve_device()

        assert f"Auto-detect: 2 CUDA device(s), libraries_ready={libraries_ready}" in session_log_text(app)

    def test_records_a_zero_device_count_verbatim(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=0))
        app.device_var.set("Auto")

        app._resolve_device()

        assert "Auto-detect: 0 CUDA device(s), libraries_ready=True" in session_log_text(app)

    def test_says_nothing_to_the_user_when_cuda_is_chosen(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))
        app.device_var.set("Auto")

        app._resolve_device()

        assert app.log_text.value == ""

    def test_falls_back_to_cpu_silently_when_a_gpu_exists_but_libraries_do_not(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))
        app.device_var.set("Auto")

        result = app._resolve_device()

        assert result == ("cpu", "int8")
        assert app.log_text.value == ""

    def test_the_detection_breadcrumb_is_the_last_entry_written(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))
        app.device_var.set("Auto")

        app._resolve_device()

        assert app._session_log[-1].endswith("Auto-detect: 1 CUDA device(s), libraries_ready=True")


class TestAutomaticDetectionFailures:
    def test_returns_cpu_when_importing_ctranslate2_is_impossible(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, None)
        app.device_var.set("Auto")

        assert app._resolve_device() == ("cpu", "int8")

    def test_records_the_import_failure_in_the_session_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, None)
        app.device_var.set("Auto")

        app._resolve_device()

        breadcrumbs = session_log_text(app)
        assert "CUDA detection failed" in breadcrumbs
        assert "defaulting to CPU" in breadcrumbs

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("no CUDA driver present"),
            OSError("cudart64_12.dll not found"),
            ValueError("unexpected device index"),
        ],
    )
    def test_returns_cpu_when_counting_devices_raises(self, app, monkeypatch, error):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=error))
        app.device_var.set("Auto")

        assert app._resolve_device() == ("cpu", "int8")

    def test_records_the_exception_text_in_the_session_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=RuntimeError("no CUDA driver present")))
        app.device_var.set("Auto")

        app._resolve_device()

        assert "CUDA detection failed (no CUDA driver present), defaulting to CPU" in session_log_text(app)

    def test_returns_cpu_when_the_module_has_no_device_count_helper(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2WithoutDeviceCount())
        app.device_var.set("Auto")

        result = app._resolve_device()

        assert result == ("cpu", "int8")
        assert "CUDA detection failed" in session_log_text(app)

    def test_does_not_claim_a_device_count_it_never_obtained(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=RuntimeError("boom")))
        app.device_var.set("Auto")

        app._resolve_device()

        assert "Auto-detect:" not in session_log_text(app)

    def test_keeps_the_failure_out_of_the_visible_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=RuntimeError("boom")))
        app.device_var.set("Auto")

        app._resolve_device()

        assert app.log_text.value == ""

    def test_the_library_probe_still_ran_before_the_failure(self, app, monkeypatch):
        probe = install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=RuntimeError("boom")))
        app.device_var.set("Auto")

        app._resolve_device()

        assert probe.call_count == 1

    def test_writes_exactly_one_breadcrumb_for_a_failed_detection(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=RuntimeError("boom")))
        app.device_var.set("Auto")

        app._resolve_device()

        assert len(app._session_log) == 1

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit()])
    def test_interpreter_level_interrupts_are_not_swallowed(self, app, monkeypatch, error):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(error=error))
        app.device_var.set("Auto")

        with pytest.raises(type(error)):
            app._resolve_device()

    def test_a_failing_library_probe_is_not_swallowed(self, app, monkeypatch):
        def explode():
            raise RuntimeError("WinDLL probe crashed")

        monkeypatch.setattr(ui.gpu_pack, "libraries_available", explode)
        app.device_var.set("Auto")

        with pytest.raises(RuntimeError, match="WinDLL probe crashed"):
            app._resolve_device()


class TestReturnContract:
    @pytest.mark.parametrize(
        ("label", "libraries_ready", "device_count"),
        [
            ("CPU", True, 4),
            ("CPU", False, 0),
            ("GPU (CUDA)", True, 0),
            ("GPU (CUDA)", False, 4),
            ("Auto", True, 1),
            ("Auto", True, 0),
            ("Auto", False, 1),
        ],
    )
    def test_always_returns_a_device_and_a_compute_type(self, app, monkeypatch, label,
                                                        libraries_ready, device_count):
        install_library_probe(monkeypatch, available=libraries_ready)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=device_count))
        app.device_var.set(label)

        device, compute_type = app._resolve_device()

        assert device in ("cpu", "cuda")
        assert compute_type in ("int8", "float16")

    @pytest.mark.parametrize(
        ("label", "libraries_ready", "device_count"),
        [
            ("CPU", True, 4),
            ("GPU (CUDA)", False, 4),
            ("Auto", False, 1),
            ("Auto", True, 0),
        ],
    )
    def test_cpu_is_always_paired_with_int8(self, app, monkeypatch, label, libraries_ready, device_count):
        install_library_probe(monkeypatch, available=libraries_ready)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=device_count))
        app.device_var.set(label)

        assert app._resolve_device() == ("cpu", "int8")

    @pytest.mark.parametrize("label", ["GPU (CUDA)", "Auto"])
    def test_cuda_is_always_paired_with_float16(self, app, monkeypatch, label):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))
        app.device_var.set(label)

        assert app._resolve_device() == ("cuda", "float16")

    def test_reads_the_choice_from_the_device_variable_on_every_call(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_cuda_module(monkeypatch, StubCTranslate2(device_count=1))

        app.device_var.set("CPU")
        first = app._resolve_device()
        app.device_var.set("GPU (CUDA)")
        second = app._resolve_device()

        assert first == ("cpu", "int8")
        assert second == ("cuda", "float16")

    def test_re_probes_the_libraries_on_every_call(self, app, monkeypatch):
        probe = install_library_probe(monkeypatch, available=True)
        app.device_var.set("GPU (CUDA)")

        app._resolve_device()
        probe.available = False
        second = app._resolve_device()

        assert second == ("cpu", "int8")
        assert probe.call_count == 2
