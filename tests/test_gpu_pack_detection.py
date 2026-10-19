"""Unit tests for the GPU pack detection helpers in src/gpu_pack.py."""

import subprocess

import pytest

import gpu_pack

CUBLAS_LIBRARY = "cublas64_12.dll"
CUDNN_LIBRARY = "cudnn64_9.dll"
NVIDIA_SMI_COMMAND = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
NVIDIA_SMI_TIMEOUT_SECONDS = 15


class RecordedLibraryLoader:
    def __init__(self, failures=None):
        self.failures = dict(failures or {})
        self.requested_libraries = []

    def __call__(self, library_name):
        self.requested_libraries.append(library_name)
        failure = self.failures.get(library_name)
        if failure is not None:
            raise failure
        return object()


class RecordedSubprocessRun:
    def __init__(self, completed_process=None, error=None):
        self.completed_process = completed_process
        self.error = error
        self.calls = []

    def __call__(self, command, **keyword_arguments):
        self.calls.append({"command": command, **keyword_arguments})
        if self.error is not None:
            raise self.error
        return self.completed_process

    @property
    def last_call(self):
        return self.calls[-1]


def write_libraries(directory, *library_names):
    for library_name in library_names:
        (directory / library_name).write_bytes(b"MZ")
    return directory


def install_library_loader(monkeypatch, failures=None):
    loader = RecordedLibraryLoader(failures)
    monkeypatch.setattr(gpu_pack.ctypes, "WinDLL", loader)
    return loader


def install_subprocess_run(monkeypatch, completed_process=None, error=None):
    recorder = RecordedSubprocessRun(completed_process, error)
    monkeypatch.setattr(gpu_pack.subprocess, "run", recorder)
    return recorder


def make_completed_process(returncode=0, stdout=""):
    return subprocess.CompletedProcess(
        args=NVIDIA_SMI_COMMAND, returncode=returncode, stdout=stdout, stderr="")


class TestParseVersion:
    @pytest.mark.parametrize(
        ("version_text", "expected"),
        [
            ("12.9.1.4", (12, 9, 1, 4)),
            ("9.20.0.48", (9, 20, 0, 48)),
            ("12", (12,)),
            ("0", (0,)),
            ("0.0.0", (0, 0, 0)),
            ("1.2.3", (1, 2, 3)),
            ("2024.11", (2024, 11)),
        ],
    )
    def test_reads_a_plain_dotted_version_as_a_tuple_of_integers(self, version_text, expected):
        assert gpu_pack._parse_version(version_text) == expected

    @pytest.mark.parametrize(
        ("version_text", "expected"),
        [
            ("1-2", (1, 2)),
            ("1+2", (1, 2)),
            ("1.2-3+4", (1, 2, 3, 4)),
            ("12-9-1-4", (12, 9, 1, 4)),
            ("12+9+1+4", (12, 9, 1, 4)),
        ],
    )
    def test_splits_on_dots_hyphens_and_pluses_alike(self, version_text, expected):
        assert gpu_pack._parse_version(version_text) == expected

    @pytest.mark.parametrize(
        ("version_text", "expected"),
        [
            ("1.2.rc1.4", (1, 2)),
            ("12.9.1.4.post1", (12, 9, 1, 4)),
            ("12.9.1.4rc1", (12, 9, 1)),
            ("1.2.3dev", (1, 2)),
            ("9.20.0.48+cuda12", (9, 20, 0, 48)),
            ("1.2.3.", (1, 2, 3)),
            ("1..2", (1,)),
        ],
    )
    def test_stops_at_the_first_non_numeric_part(self, version_text, expected):
        assert gpu_pack._parse_version(version_text) == expected

    @pytest.mark.parametrize(
        "version_text",
        ["rc1", "", "v1.2.3", "alpha", " 12", "-1.2", "+1", ".", "1_2"],
    )
    def test_yields_the_empty_tuple_when_the_first_part_is_not_numeric(self, version_text):
        assert gpu_pack._parse_version(version_text) == ()

    def test_a_negative_looking_version_loses_its_sign_because_the_hyphen_is_a_separator(self):
        assert gpu_pack._parse_version("1.2-3") == (1, 2, 3)

    def test_drops_leading_zeros_because_parts_become_integers(self):
        assert gpu_pack._parse_version("1.02.003") == (1, 2, 3)

    def test_every_element_is_an_integer_rather_than_a_string(self):
        parsed = gpu_pack._parse_version("12.9.1.4")

        assert all(isinstance(number, int) for number in parsed)

    def test_returns_a_tuple(self):
        assert isinstance(gpu_pack._parse_version("12.9"), tuple)

    def test_accepts_non_ascii_decimal_digits(self):
        assert gpu_pack._parse_version("١٢.3") == (12, 3)

    def test_a_superscript_digit_passes_the_digit_check_and_crashes_the_integer_conversion(self):
        with pytest.raises(ValueError):
            gpu_pack._parse_version("1.²")


class TestParsedVersionOrdering:
    @pytest.mark.parametrize(
        ("lower_version", "higher_version"),
        [
            ("12.9", "12.10"),
            ("9.3.0.75", "9.20.0.48"),
            ("2", "10"),
            ("1.2", "1.2.1"),
            ("12", "12.0.1"),
            ("12.9.1.4", "12.9.2.0"),
            ("9.1.0.70", "9.1.0.71"),
        ],
    )
    def test_tuple_comparison_orders_versions_by_numeric_component(self, lower_version, higher_version):
        assert gpu_pack._parse_version(lower_version) < gpu_pack._parse_version(higher_version)

    def test_text_comparison_would_order_a_two_digit_minor_wrongly(self):
        assert "12.9" > "12.10"
        assert gpu_pack._parse_version("12.9") < gpu_pack._parse_version("12.10")

    def test_identical_versions_compare_equal(self):
        assert gpu_pack._parse_version("12.9.1.4") == gpu_pack._parse_version("12.9.1.4")

    def test_an_unparsable_version_sorts_below_every_parsed_version(self):
        assert gpu_pack._parse_version("rc1") < gpu_pack._parse_version("0")

    def test_the_newest_release_wins_when_taking_the_maximum_of_a_release_list(self):
        releases = ["12.9.1.4", "12.10.0.0", "9.20.0.48", "12.9.2.1"]

        newest = max(gpu_pack._parse_version(release) for release in releases)

        assert newest == (12, 10, 0, 0)


class TestIsInstalled:
    def test_returns_true_when_both_required_libraries_exist(self, cuda_dir):
        write_libraries(cuda_dir, CUBLAS_LIBRARY, CUDNN_LIBRARY)

        assert gpu_pack.is_installed() is True

    @pytest.mark.parametrize("present_library", [CUBLAS_LIBRARY, CUDNN_LIBRARY])
    def test_returns_false_when_only_one_required_library_exists(self, cuda_dir, present_library):
        write_libraries(cuda_dir, present_library)

        assert gpu_pack.is_installed() is False

    def test_returns_false_when_the_cuda_directory_is_empty(self, cuda_dir):
        assert list(cuda_dir.iterdir()) == []

        assert gpu_pack.is_installed() is False

    def test_returns_false_when_the_directory_holds_only_unrelated_libraries(self, cuda_dir):
        write_libraries(cuda_dir, "cublas64_11.dll", "cudnn64_8.dll", "cudart64_12.dll")

        assert gpu_pack.is_installed() is False

    def test_returns_false_when_the_cuda_directory_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu_pack, "CUDA_DIR", str(tmp_path / "never-created"))

        assert gpu_pack.is_installed() is False

    def test_a_directory_named_like_a_required_library_counts_as_installed(self, cuda_dir):
        (cuda_dir / CUBLAS_LIBRARY).mkdir()
        write_libraries(cuda_dir, CUDNN_LIBRARY)

        assert gpu_pack.is_installed() is True

    def test_an_empty_file_counts_as_an_installed_library(self, cuda_dir):
        (cuda_dir / CUBLAS_LIBRARY).write_bytes(b"")
        (cuda_dir / CUDNN_LIBRARY).write_bytes(b"")

        assert gpu_pack.is_installed() is True

    def test_reads_the_cuda_directory_on_every_call_rather_than_caching(self, cuda_dir, tmp_path, monkeypatch):
        write_libraries(cuda_dir, CUBLAS_LIBRARY, CUDNN_LIBRARY)
        assert gpu_pack.is_installed() is True

        other_directory = tmp_path / "other-cuda"
        other_directory.mkdir()
        monkeypatch.setattr(gpu_pack, "CUDA_DIR", str(other_directory))

        assert gpu_pack.is_installed() is False

    def test_notices_a_library_deleted_between_calls(self, cuda_dir):
        write_libraries(cuda_dir, CUBLAS_LIBRARY, CUDNN_LIBRARY)
        assert gpu_pack.is_installed() is True

        (cuda_dir / CUDNN_LIBRARY).unlink()

        assert gpu_pack.is_installed() is False

    def test_required_libraries_are_the_cublas_and_cudnn_pair(self):
        assert gpu_pack.REQUIRED_LIBRARIES == (CUBLAS_LIBRARY, CUDNN_LIBRARY)


class TestLibrariesAvailable:
    def test_returns_true_without_loading_anything_when_the_pack_is_installed(self, cuda_dir, monkeypatch):
        write_libraries(cuda_dir, CUBLAS_LIBRARY, CUDNN_LIBRARY)
        loader = install_library_loader(monkeypatch)

        assert gpu_pack.libraries_available() is True
        assert loader.requested_libraries == []

    def test_does_not_load_anything_even_when_every_load_would_fail(self, cuda_dir, monkeypatch):
        write_libraries(cuda_dir, CUBLAS_LIBRARY, CUDNN_LIBRARY)
        loader = install_library_loader(
            monkeypatch,
            {CUBLAS_LIBRARY: OSError("boom"), CUDNN_LIBRARY: OSError("boom")},
        )

        assert gpu_pack.libraries_available() is True
        assert loader.requested_libraries == []

    def test_returns_true_when_every_library_loads_from_the_system(self, cuda_dir, monkeypatch):
        loader = install_library_loader(monkeypatch)

        assert gpu_pack.libraries_available() is True
        assert loader.requested_libraries == [CUBLAS_LIBRARY, CUDNN_LIBRARY]

    def test_loads_the_libraries_in_the_declared_order(self, cuda_dir, monkeypatch):
        loader = install_library_loader(monkeypatch)

        gpu_pack.libraries_available()

        assert loader.requested_libraries == list(gpu_pack.REQUIRED_LIBRARIES)

    def test_stops_at_the_first_library_that_fails_to_load(self, cuda_dir, monkeypatch):
        loader = install_library_loader(monkeypatch, {CUBLAS_LIBRARY: OSError("not found")})

        assert gpu_pack.libraries_available() is False
        assert loader.requested_libraries == [CUBLAS_LIBRARY]

    def test_reports_unavailable_when_the_second_library_fails_to_load(self, cuda_dir, monkeypatch):
        loader = install_library_loader(monkeypatch, {CUDNN_LIBRARY: OSError("not found")})

        assert gpu_pack.libraries_available() is False
        assert loader.requested_libraries == [CUBLAS_LIBRARY, CUDNN_LIBRARY]

    @pytest.mark.parametrize(
        "failure",
        [
            OSError("[WinError 126] The specified module could not be found"),
            FileNotFoundError(CUDNN_LIBRARY),
            PermissionError("access denied"),
        ],
        ids=["oserror", "filenotfounderror", "permissionerror"],
    )
    def test_swallows_every_oserror_flavour_and_reports_unavailable(self, cuda_dir, monkeypatch, failure):
        install_library_loader(monkeypatch, {CUBLAS_LIBRARY: failure})

        assert gpu_pack.libraries_available() is False

    def test_a_failure_that_is_not_an_oserror_is_not_swallowed(self, cuda_dir, monkeypatch):
        install_library_loader(monkeypatch, {CUBLAS_LIBRARY: RuntimeError("unexpected")})

        with pytest.raises(RuntimeError):
            gpu_pack.libraries_available()

    @pytest.mark.parametrize("present_library", [CUBLAS_LIBRARY, CUDNN_LIBRARY])
    def test_falls_back_to_loading_when_the_pack_is_only_half_installed(self, cuda_dir, monkeypatch, present_library):
        write_libraries(cuda_dir, present_library)
        loader = install_library_loader(monkeypatch)

        assert gpu_pack.libraries_available() is True
        assert loader.requested_libraries == [CUBLAS_LIBRARY, CUDNN_LIBRARY]

    def test_a_half_installed_pack_is_unavailable_when_the_system_copy_is_missing(self, cuda_dir, monkeypatch):
        write_libraries(cuda_dir, CUBLAS_LIBRARY)
        install_library_loader(monkeypatch, {CUDNN_LIBRARY: OSError("not found")})

        assert gpu_pack.libraries_available() is False

    def test_installing_the_pack_between_calls_flips_the_answer(self, cuda_dir, monkeypatch):
        install_library_loader(monkeypatch, {CUBLAS_LIBRARY: OSError("not found")})
        assert gpu_pack.libraries_available() is False

        write_libraries(cuda_dir, CUBLAS_LIBRARY, CUDNN_LIBRARY)

        assert gpu_pack.libraries_available() is True


class TestDetectNvidiaGpu:
    def test_returns_the_reported_gpu_name(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "NVIDIA GeForce RTX 4090\n"))

        assert gpu_pack.detect_nvidia_gpu() == "NVIDIA GeForce RTX 4090"

    def test_returns_only_the_first_gpu_when_several_are_reported(self, monkeypatch):
        stdout = "NVIDIA A100-SXM4-40GB\nNVIDIA A100-SXM4-40GB\nTesla T4\n"
        install_subprocess_run(monkeypatch, make_completed_process(0, stdout))

        assert gpu_pack.detect_nvidia_gpu() == "NVIDIA A100-SXM4-40GB"

    def test_strips_the_whitespace_around_the_reported_name(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "   NVIDIA T1000 8GB   \n"))

        assert gpu_pack.detect_nvidia_gpu() == "NVIDIA T1000 8GB"

    def test_strips_the_padding_of_the_first_line_when_more_lines_follow(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "  Tesla T4  \n  Tesla V100  \n"))

        assert gpu_pack.detect_nvidia_gpu() == "Tesla T4"

    def test_ignores_blank_lines_before_the_first_gpu(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "\n \n\nQuadro P2000\n"))

        assert gpu_pack.detect_nvidia_gpu() == "Quadro P2000"

    def test_handles_windows_line_endings(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "NVIDIA RTX A2000\r\nTesla T4\r\n"))

        assert gpu_pack.detect_nvidia_gpu() == "NVIDIA RTX A2000"

    def test_keeps_the_inner_spacing_of_a_gpu_name(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "NVIDIA GeForce GTX 1080 Ti\n"))

        assert gpu_pack.detect_nvidia_gpu() == "NVIDIA GeForce GTX 1080 Ti"

    def test_returns_the_name_when_the_output_has_no_trailing_newline(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4"))

        assert gpu_pack.detect_nvidia_gpu() == "Tesla T4"

    @pytest.mark.parametrize("returncode", [1, 2, 9, 127, -1])
    def test_returns_none_when_nvidia_smi_exits_with_a_failure_code(self, monkeypatch, returncode):
        install_subprocess_run(monkeypatch, make_completed_process(returncode, "NVIDIA GeForce RTX 4090\n"))

        assert gpu_pack.detect_nvidia_gpu() is None

    @pytest.mark.parametrize(
        "stdout",
        ["", " ", "\n", "\r\n", "\n\n\n", "   \t  \n  \n"],
        ids=["empty", "space", "newline", "crlf", "blank-lines", "mixed-whitespace"],
    )
    def test_returns_none_when_the_output_is_empty_or_whitespace_only(self, monkeypatch, stdout):
        install_subprocess_run(monkeypatch, make_completed_process(0, stdout))

        assert gpu_pack.detect_nvidia_gpu() is None

    def test_returns_none_when_nvidia_smi_is_not_installed(self, monkeypatch):
        install_subprocess_run(monkeypatch, error=FileNotFoundError(2, "The system cannot find the file specified"))

        assert gpu_pack.detect_nvidia_gpu() is None

    def test_returns_none_when_nvidia_smi_times_out(self, monkeypatch):
        timeout_error = subprocess.TimeoutExpired(NVIDIA_SMI_COMMAND, NVIDIA_SMI_TIMEOUT_SECONDS)
        install_subprocess_run(monkeypatch, error=timeout_error)

        assert gpu_pack.detect_nvidia_gpu() is None

    @pytest.mark.parametrize(
        "error",
        [
            OSError("[WinError 5] Access is denied"),
            PermissionError("access denied"),
            MemoryError(),
            RuntimeError("something unexpected"),
            ValueError("bad argument"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ],
        ids=["oserror", "permissionerror", "memoryerror", "runtimeerror", "valueerror", "unicodedecodeerror"],
    )
    def test_the_broad_except_clause_swallows_every_ordinary_exception(self, monkeypatch, error):
        install_subprocess_run(monkeypatch, error=error)

        assert gpu_pack.detect_nvidia_gpu() is None

    def test_an_interrupt_is_not_swallowed_because_it_is_not_an_exception(self, monkeypatch):
        install_subprocess_run(monkeypatch, error=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            gpu_pack.detect_nvidia_gpu()

    def test_returns_none_when_the_result_carries_no_captured_output(self, monkeypatch):
        install_subprocess_run(monkeypatch, make_completed_process(0, None))

        assert gpu_pack.detect_nvidia_gpu() is None

    def test_queries_nvidia_smi_for_the_gpu_name_without_a_csv_header(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()

        assert recorder.last_call["command"] == NVIDIA_SMI_COMMAND

    def test_gives_nvidia_smi_a_fifteen_second_timeout(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()

        assert recorder.last_call["timeout"] == NVIDIA_SMI_TIMEOUT_SECONDS

    def test_captures_the_output_as_decoded_text(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()

        assert recorder.last_call["capture_output"] is True
        assert recorder.last_call["text"] is True

    def test_hides_the_console_window_of_the_child_process(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()

        assert recorder.last_call["creationflags"] == subprocess.CREATE_NO_WINDOW

    def test_never_passes_a_shell_or_a_check_flag(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()

        assert "shell" not in recorder.last_call
        assert "check" not in recorder.last_call

    def test_runs_nvidia_smi_exactly_once_per_detection(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()

        assert len(recorder.calls) == 1

    def test_does_not_retry_after_a_failed_detection(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, error=FileNotFoundError("nvidia-smi"))

        assert gpu_pack.detect_nvidia_gpu() is None
        assert len(recorder.calls) == 1

    def test_detection_is_repeated_on_every_call(self, monkeypatch):
        recorder = install_subprocess_run(monkeypatch, make_completed_process(0, "Tesla T4\n"))

        gpu_pack.detect_nvidia_gpu()
        gpu_pack.detect_nvidia_gpu()

        assert len(recorder.calls) == 2
