"""Unit tests for the argument parsing and startup helpers in src/main.py."""

import os
import sys
from types import SimpleNamespace

import pytest

BASELINE_PATH_ENTRY = r"C:\Windows\system32"
VENV_NVIDIA_SEGMENTS = (".venv", "Lib", "site-packages", "nvidia")


class RecordingDpiApi:
    def __init__(self, error=None):
        self.error = error
        self.awareness_values = []

    def SetProcessDpiAwareness(self, awareness_value):
        self.awareness_values.append(awareness_value)
        if self.error is not None:
            raise self.error
        return 0


def make_ctypes_double(shcore_api):
    return SimpleNamespace(windll=SimpleNamespace(shcore=shcore_api))


class RecordingDllDirectoryRegistry:
    def __init__(self, error=None):
        self.error = error
        self.directories = []
        self.path_when_called = []

    def __call__(self, directory):
        self.directories.append(directory)
        self.path_when_called.append(os.environ.get("PATH", ""))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(close=lambda: None)


class CudaRegistrationHarness:
    def __init__(self, main_module, monkeypatch, registry, cuda_directory, resource_directory):
        self.main_module = main_module
        self.monkeypatch = monkeypatch
        self.registry = registry
        self.cuda_directory = cuda_directory
        self.resource_directory = resource_directory

    @property
    def cublas_directory(self):
        return self.resource_directory.joinpath(*VENV_NVIDIA_SEGMENTS, "cublas", "bin")

    @property
    def cudnn_directory(self):
        return self.resource_directory.joinpath(*VENV_NVIDIA_SEGMENTS, "cudnn", "bin")

    def create(self, *directories):
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def run_from_source(self):
        self.monkeypatch.setattr(self.main_module, "IS_FROZEN", False)

    def fail_registration_with(self, error):
        self.registry.error = error

    def register(self):
        return self.main_module._register_cuda_directories()

    @property
    def path_value(self):
        return os.environ.get("PATH", "")

    @property
    def path_entries(self):
        return self.path_value.split(os.pathsep)


@pytest.fixture
def cuda_registration(tmp_path, monkeypatch, main_module):
    monkeypatch.setenv("PATH", BASELINE_PATH_ENTRY)

    registry = RecordingDllDirectoryRegistry()
    monkeypatch.setattr(os, "add_dll_directory", registry)

    cuda_directory = tmp_path / "cuda"
    resource_directory = tmp_path / "resources"
    monkeypatch.setattr(main_module, "CUDA_DIR", str(cuda_directory))
    monkeypatch.setattr(main_module, "RESOURCE_DIR", str(resource_directory))
    monkeypatch.setattr(main_module, "IS_FROZEN", True)

    return CudaRegistrationHarness(
        main_module, monkeypatch, registry, cuda_directory, resource_directory)


class TestModuleContract:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("EXIT_OK", 0),
            ("EXIT_FAILED", 1),
            ("EXIT_BAD_USAGE", 2),
        ],
    )
    def test_exit_code_constant_has_the_documented_value(self, main_module, name, value):
        assert getattr(main_module, name) == value

    def test_exit_codes_are_distinct(self, main_module):
        codes = [main_module.EXIT_OK, main_module.EXIT_FAILED, main_module.EXIT_BAD_USAGE]

        assert len(set(codes)) == len(codes)

    def test_success_is_the_only_zero_exit_code(self, main_module):
        assert main_module.EXIT_OK == 0
        assert main_module.EXIT_FAILED != 0
        assert main_module.EXIT_BAD_USAGE != 0

    def test_transcribe_flag_spelling_is_pinned(self, main_module):
        assert main_module.TRANSCRIBE_FLAG == "--transcribe"

    def test_selftest_flag_spelling_is_pinned(self, main_module):
        assert main_module.SELFTEST_FLAG == "--selftest"

    def test_the_two_flags_are_different(self, main_module):
        assert main_module.TRANSCRIBE_FLAG != main_module.SELFTEST_FLAG

    def test_flags_use_the_double_dash_convention(self, main_module):
        assert main_module.TRANSCRIBE_FLAG.startswith("--")
        assert main_module.SELFTEST_FLAG.startswith("--")

    def test_system_dpi_awareness_constant_is_one(self, main_module):
        assert main_module.DPI_AWARENESS_SYSTEM == 1


class TestArgumentValue:
    def test_returns_the_token_after_the_flag(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--model", "large-v3-turbo"])

        assert main_module._argument_value("--model") == "large-v3-turbo"

    def test_returns_the_default_when_the_flag_is_absent(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--device", "CPU"])

        assert main_module._argument_value("--model", "large-v3-turbo") == "large-v3-turbo"

    def test_default_is_none_when_left_unspecified(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox"])

        assert main_module._argument_value("--model") is None

    def test_returns_the_default_when_the_flag_is_the_last_token(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--device", "CPU", "--model"])

        assert main_module._argument_value("--model", "fallback") == "fallback"

    def test_returns_none_when_the_flag_is_the_last_token_and_no_default_is_given(
            self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--model"])

        assert main_module._argument_value("--model") is None

    @pytest.mark.parametrize(
        "argv",
        [
            ["sotvox", "--model", "tiny", "--device", "CPU", "--language", "Spanish"],
            ["sotvox", "--device", "CPU", "--model", "tiny", "--language", "Spanish"],
            ["sotvox", "--device", "CPU", "--language", "Spanish", "--model", "tiny"],
        ],
        ids=["first", "middle", "last"],
    )
    def test_finds_the_flag_at_any_position(self, main_module, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", argv)

        assert main_module._argument_value("--model") == "tiny"

    def test_the_first_occurrence_wins_when_a_flag_is_repeated(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--model", "tiny", "--model", "large-v3"])

        assert main_module._argument_value("--model") == "tiny"

    def test_a_value_that_looks_like_a_flag_is_still_returned(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--model", "--device"])

        assert main_module._argument_value("--model") == "--device"

    def test_an_empty_string_value_is_returned_rather_than_the_default(
            self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--output", "", "--model", "tiny"])

        assert main_module._argument_value("--output", "C:\\transcripts") == ""

    def test_matches_the_flag_exactly_and_not_by_prefix(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--models", "tiny"])

        assert main_module._argument_value("--model", "fallback") == "fallback"

    def test_matches_the_flag_case_sensitively(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sotvox", "--MODEL", "tiny"])

        assert main_module._argument_value("--model", "fallback") == "fallback"

    def test_returns_the_default_for_an_empty_command_line(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", [])

        assert main_module._argument_value("--model", "fallback") == "fallback"

    def test_matches_a_flag_that_occupies_the_program_name_slot(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["--model", "tiny"])

        assert main_module._argument_value("--model") == "tiny"

    def test_leaves_the_command_line_untouched(self, main_module, monkeypatch):
        argv = ["sotvox", "--model", "tiny"]
        monkeypatch.setattr(sys, "argv", argv)

        main_module._argument_value("--model")

        assert argv == ["sotvox", "--model", "tiny"]

    @pytest.mark.parametrize("default", [None, "", "large-v3-turbo", 0])
    def test_any_default_object_is_passed_straight_back(self, main_module, monkeypatch, default):
        monkeypatch.setattr(sys, "argv", ["sotvox"])

        assert main_module._argument_value("--model", default) is default

    def test_reads_the_batch_flags_the_way_the_batch_runner_does(self, main_module, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "sotvox", "--transcribe", "C:\\inbox", "--output", "C:\\out",
            "--model", "small", "--language", "Spanish", "--device", "CPU",
        ])

        assert main_module._argument_value(main_module.TRANSCRIBE_FLAG) == "C:\\inbox"
        assert main_module._argument_value("--output", "C:\\default") == "C:\\out"
        assert main_module._argument_value("--model", "large-v3-turbo") == "small"
        assert main_module._argument_value("--language", "Auto-detect") == "Spanish"
        assert main_module._argument_value("--device", "Auto") == "CPU"


class TestEnableDpiAwareness:
    def test_requests_system_dpi_awareness(self, main_module, monkeypatch):
        api = RecordingDpiApi()
        monkeypatch.setattr(main_module, "ctypes", make_ctypes_double(api))

        main_module._enable_dpi_awareness()

        assert api.awareness_values == [1]

    def test_uses_the_named_awareness_constant(self, main_module, monkeypatch):
        api = RecordingDpiApi()
        monkeypatch.setattr(main_module, "ctypes", make_ctypes_double(api))

        main_module._enable_dpi_awareness()

        assert api.awareness_values == [main_module.DPI_AWARENESS_SYSTEM]

    def test_calls_the_shell_api_exactly_once(self, main_module, monkeypatch):
        api = RecordingDpiApi()
        monkeypatch.setattr(main_module, "ctypes", make_ctypes_double(api))

        main_module._enable_dpi_awareness()

        assert len(api.awareness_values) == 1

    def test_returns_nothing(self, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "ctypes", make_ctypes_double(RecordingDpiApi()))

        assert main_module._enable_dpi_awareness() is None

    @pytest.mark.parametrize(
        "error",
        [
            OSError("shcore.dll is unavailable"),
            AttributeError("SetProcessDpiAwareness"),
            RuntimeError("per-monitor awareness already set"),
            ValueError("bad awareness value"),
        ],
        ids=["os-error", "attribute-error", "runtime-error", "value-error"],
    )
    def test_swallows_any_failure_from_the_shell_api(self, main_module, monkeypatch, error):
        api = RecordingDpiApi(error=error)
        monkeypatch.setattr(main_module, "ctypes", make_ctypes_double(api))

        assert main_module._enable_dpi_awareness() is None
        assert api.awareness_values == [1]

    def test_swallows_a_missing_shcore_library(self, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "ctypes", SimpleNamespace(windll=SimpleNamespace()))

        assert main_module._enable_dpi_awareness() is None

    def test_swallows_a_missing_windll_loader_on_non_windows_hosts(self, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "ctypes", SimpleNamespace())

        assert main_module._enable_dpi_awareness() is None

    def test_does_not_swallow_a_keyboard_interrupt(self, main_module, monkeypatch):
        api = RecordingDpiApi(error=KeyboardInterrupt())
        monkeypatch.setattr(main_module, "ctypes", make_ctypes_double(api))

        with pytest.raises(KeyboardInterrupt):
            main_module._enable_dpi_awareness()


class TestRegisterCudaDirectories:
    def test_registers_the_cuda_directory_when_it_exists(self, cuda_registration):
        cuda_registration.create(cuda_registration.cuda_directory)

        cuda_registration.register()

        assert cuda_registration.registry.directories == [str(cuda_registration.cuda_directory)]

    def test_prepends_the_cuda_directory_to_path(self, cuda_registration):
        cuda_registration.create(cuda_registration.cuda_directory)

        cuda_registration.register()

        assert cuda_registration.path_entries == [
            str(cuda_registration.cuda_directory), BASELINE_PATH_ENTRY]

    def test_ignores_the_cuda_directory_when_it_does_not_exist(self, cuda_registration):
        cuda_registration.register()

        assert cuda_registration.registry.directories == []
        assert cuda_registration.path_value == BASELINE_PATH_ENTRY

    def test_ignores_a_candidate_that_is_a_file_rather_than_a_directory(self, cuda_registration):
        cuda_registration.cuda_directory.write_text("not a directory", encoding="utf-8")

        cuda_registration.register()

        assert cuda_registration.registry.directories == []
        assert cuda_registration.path_value == BASELINE_PATH_ENTRY

    def test_returns_nothing(self, cuda_registration):
        cuda_registration.create(cuda_registration.cuda_directory)

        assert cuda_registration.register() is None

    def test_registers_the_dll_directory_before_touching_path(self, cuda_registration):
        cuda_registration.create(cuda_registration.cuda_directory)

        cuda_registration.register()

        assert cuda_registration.registry.path_when_called == [BASELINE_PATH_ENTRY]

    def test_skips_the_venv_libraries_when_frozen(self, cuda_registration):
        cuda_registration.create(
            cuda_registration.cuda_directory,
            cuda_registration.cublas_directory,
            cuda_registration.cudnn_directory,
        )

        cuda_registration.register()

        assert cuda_registration.registry.directories == [str(cuda_registration.cuda_directory)]
        assert str(cuda_registration.cublas_directory) not in cuda_registration.path_value
        assert str(cuda_registration.cudnn_directory) not in cuda_registration.path_value

    def test_registers_the_venv_libraries_when_running_from_source(self, cuda_registration):
        cuda_registration.run_from_source()
        cuda_registration.create(
            cuda_registration.cuda_directory,
            cuda_registration.cublas_directory,
            cuda_registration.cudnn_directory,
        )

        cuda_registration.register()

        assert cuda_registration.registry.directories == [
            str(cuda_registration.cuda_directory),
            str(cuda_registration.cublas_directory),
            str(cuda_registration.cudnn_directory),
        ]

    def test_each_existing_directory_lands_ahead_of_the_previous_one(self, cuda_registration):
        cuda_registration.run_from_source()
        cuda_registration.create(
            cuda_registration.cuda_directory,
            cuda_registration.cublas_directory,
            cuda_registration.cudnn_directory,
        )

        cuda_registration.register()

        assert cuda_registration.path_entries == [
            str(cuda_registration.cudnn_directory),
            str(cuda_registration.cublas_directory),
            str(cuda_registration.cuda_directory),
            BASELINE_PATH_ENTRY,
        ]

    def test_only_the_venv_directories_that_exist_are_registered(self, cuda_registration):
        cuda_registration.run_from_source()
        cuda_registration.create(cuda_registration.cudnn_directory)

        cuda_registration.register()

        assert cuda_registration.registry.directories == [str(cuda_registration.cudnn_directory)]

    def test_venv_directories_are_registered_even_when_the_cuda_directory_is_absent(
            self, cuda_registration):
        cuda_registration.run_from_source()
        cuda_registration.create(
            cuda_registration.cublas_directory, cuda_registration.cudnn_directory)

        cuda_registration.register()

        assert cuda_registration.registry.directories == [
            str(cuda_registration.cublas_directory),
            str(cuda_registration.cudnn_directory),
        ]

    def test_looks_for_the_venv_libraries_under_the_resource_directory(self, cuda_registration):
        cuda_registration.run_from_source()
        cuda_registration.create(cuda_registration.cublas_directory)

        cuda_registration.register()

        expected = os.path.join(
            str(cuda_registration.resource_directory),
            ".venv", "Lib", "site-packages", "nvidia", "cublas", "bin")
        assert cuda_registration.registry.directories == [expected]

    def test_does_not_add_a_directory_that_is_already_on_path(self, cuda_registration, monkeypatch):
        cuda_registration.create(cuda_registration.cuda_directory)
        already_present = f"{cuda_registration.cuda_directory}{os.pathsep}{BASELINE_PATH_ENTRY}"
        monkeypatch.setenv("PATH", already_present)

        cuda_registration.register()

        assert cuda_registration.path_value == already_present
        assert cuda_registration.path_entries.count(str(cuda_registration.cuda_directory)) == 1

    def test_still_registers_a_directory_that_is_already_on_path_with_the_loader(
            self, cuda_registration, monkeypatch):
        cuda_registration.create(cuda_registration.cuda_directory)
        monkeypatch.setenv(
            "PATH", f"{cuda_registration.cuda_directory}{os.pathsep}{BASELINE_PATH_ENTRY}")

        cuda_registration.register()

        assert cuda_registration.registry.directories == [str(cuda_registration.cuda_directory)]

    def test_a_second_call_does_not_duplicate_the_path_entry(self, cuda_registration):
        cuda_registration.create(cuda_registration.cuda_directory)

        cuda_registration.register()
        cuda_registration.register()

        assert cuda_registration.path_entries.count(str(cuda_registration.cuda_directory)) == 1
        assert len(cuda_registration.registry.directories) == 2

    def test_skips_a_directory_that_is_only_a_substring_of_an_existing_path_entry(
            self, cuda_registration, monkeypatch, tmp_path):
        cuda_registration.create(cuda_registration.cuda_directory)
        longer_entry = str(tmp_path / "cuda_extra")
        monkeypatch.setenv("PATH", longer_entry)

        cuda_registration.register()

        assert cuda_registration.path_value == longer_entry

    def test_creates_path_when_the_environment_has_no_path_variable(
            self, cuda_registration, monkeypatch):
        cuda_registration.create(cuda_registration.cuda_directory)
        monkeypatch.delenv("PATH", raising=False)

        cuda_registration.register()

        assert cuda_registration.path_value == str(cuda_registration.cuda_directory) + os.pathsep

    def test_prepends_onto_an_empty_path(self, cuda_registration, monkeypatch):
        cuda_registration.create(cuda_registration.cuda_directory)
        monkeypatch.setenv("PATH", "")

        cuda_registration.register()

        assert cuda_registration.path_value == str(cuda_registration.cuda_directory) + os.pathsep

    @pytest.mark.parametrize(
        "error",
        [OSError("access denied"), AttributeError("add_dll_directory")],
        ids=["os-error", "attribute-error"],
    )
    def test_swallows_a_loader_failure_and_still_updates_path(self, cuda_registration, error):
        cuda_registration.create(cuda_registration.cuda_directory)
        cuda_registration.fail_registration_with(error)

        cuda_registration.register()

        assert cuda_registration.path_entries == [
            str(cuda_registration.cuda_directory), BASELINE_PATH_ENTRY]

    def test_a_loader_failure_does_not_stop_the_remaining_candidates(self, cuda_registration):
        cuda_registration.run_from_source()
        cuda_registration.create(
            cuda_registration.cuda_directory,
            cuda_registration.cublas_directory,
            cuda_registration.cudnn_directory,
        )
        cuda_registration.fail_registration_with(OSError("access denied"))

        cuda_registration.register()

        assert len(cuda_registration.registry.directories) == 3
        assert len(cuda_registration.path_entries) == 4

    def test_propagates_a_loader_failure_that_is_not_an_os_or_attribute_error(
            self, cuda_registration):
        cuda_registration.create(cuda_registration.cuda_directory)
        cuda_registration.fail_registration_with(ValueError("bad directory"))

        with pytest.raises(ValueError, match="bad directory"):
            cuda_registration.register()

        assert cuda_registration.path_value == BASELINE_PATH_ENTRY
