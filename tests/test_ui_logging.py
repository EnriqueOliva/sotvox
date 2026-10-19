"""Unit tests for the SotvoxApp session log, log-file bootstrap, flushing and shutdown."""

import datetime as datetime_module
import pathlib
import platform
import re
import subprocess
import sys

import pytest
from doubles import FakeRoot, FakeText

import ui

FROZEN_MOMENT = datetime_module.datetime(2026, 7, 28, 14, 5, 9)
FROZEN_STAMP = "[28-07-2026] - [14-05-09]"
FROZEN_TIME_PREFIX = "[14:05:09]"


class FrozenDateTime:
    @classmethod
    def now(cls):
        return FROZEN_MOMENT


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


class RecordingSubprocessRun:
    def __init__(self, *results):
        self.results = list(results)
        self.commands = []
        self.keyword_arguments = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.keyword_arguments.append(kwargs)
        index = len(self.commands) - 1
        outcome = self.results[index] if index < len(self.results) else self.results[-1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class OrderRecordingText(FakeText):
    def __init__(self, **initial):
        super().__init__(**initial)
        self.operations = []

    def configure(self, **kwargs):
        super().configure(**kwargs)
        self.operations.append(("configure", kwargs.get("state")))

    config = configure

    def insert(self, index, text):
        super().insert(index, text)
        self.operations.append(("insert", text))

    def see(self, index):
        super().see(index)
        self.operations.append(("see", index))


def install_subprocess_results(monkeypatch, *results):
    runner = RecordingSubprocessRun(*results)
    monkeypatch.setattr(ui.subprocess, "run", runner)
    return runner


def header_text(app):
    return pathlib.Path(app._log_path).read_text(encoding="utf-8")


def header_lines(app):
    return header_text(app).splitlines()


@pytest.fixture
def log_directory(tmp_path, monkeypatch):
    directory = tmp_path / "appdata" / "logs"
    monkeypatch.setattr(ui, "LOG_DIR", str(directory))
    return directory


@pytest.fixture
def frozen_now(monkeypatch):
    monkeypatch.setattr(ui, "datetime", FrozenDateTime)
    return FROZEN_MOMENT


@pytest.fixture
def recorded_flush(monkeypatch):
    calls = []
    monkeypatch.setattr(ui.SotvoxApp, "_flush_log", lambda self: calls.append(self))
    return calls


@pytest.fixture
def missing_nvidia_smi(monkeypatch):
    return install_subprocess_results(
        monkeypatch, FileNotFoundError("nvidia-smi is not on PATH"))


class TestSessionLogAppend:
    def test_appends_the_message_to_the_session_log(self, app, frozen_now):
        app._slog("Model loaded")

        assert app._session_log == [f"{FROZEN_TIME_PREFIX} Model loaded"]

    def test_prefixes_every_entry_with_an_hour_minute_second_timestamp(self, app):
        app._slog("anything")

        assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2}\] anything", app._session_log[0])

    def test_preserves_the_order_of_successive_entries(self, app, frozen_now):
        app._slog("first")
        app._slog("second")
        app._slog("third")

        assert app._session_log == [
            f"{FROZEN_TIME_PREFIX} first",
            f"{FROZEN_TIME_PREFIX} second",
            f"{FROZEN_TIME_PREFIX} third",
        ]

    def test_keeps_a_multi_line_message_as_a_single_entry(self, app, frozen_now):
        app._slog("Traceback:\nline one\nline two")

        assert app._session_log == [f"{FROZEN_TIME_PREFIX} Traceback:\nline one\nline two"]

    def test_records_an_empty_message_as_a_bare_timestamp(self, app, frozen_now):
        app._slog("")

        assert app._session_log == [f"{FROZEN_TIME_PREFIX} "]

    def test_touches_no_widget(self, app):
        app._slog("quiet entry")

        assert app.log_text.content == []
        assert app.log_text.config_history == []
        assert app.progress_canvas.shapes == []
        assert app.status_left.config_history == []

    def test_schedules_nothing_on_the_main_loop(self, app):
        app._slog("quiet entry")

        assert app.root.after_calls == []

    def test_writes_no_file_even_when_a_log_path_is_configured(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")

        app._slog("quiet entry")

        assert list(tmp_path.iterdir()) == []

    def test_does_not_advance_the_flush_index(self, app):
        app._slog("quiet entry")

        assert app._log_flush_idx == 0


class TestUiLog:
    def test_records_the_line_in_the_session_log(self, app, frozen_now):
        app._log("Processing: clip.mp3")

        assert app._session_log == [f"{FROZEN_TIME_PREFIX} Processing: clip.mp3"]

    def test_appends_the_text_to_the_widget_with_a_trailing_newline(self, app):
        app._log("Processing: clip.mp3")

        assert app.log_text.value == "Processing: clip.mp3\n"

    def test_enables_the_widget_around_the_insert_and_disables_it_again(self, app):
        app.log_text = OrderRecordingText()

        app._log("Done")

        assert app.log_text.operations == [
            ("configure", "normal"),
            ("insert", "Done\n"),
            ("see", "end"),
            ("configure", "disabled"),
        ]

    def test_leaves_the_widget_disabled_afterwards(self, app):
        app._log("Done")

        assert app.log_text.cget("state") == "disabled"

    def test_scrolls_the_widget_to_the_end(self, app):
        app._log("Done")

        assert app.log_text.see_calls == ["end"]

    def test_schedules_the_widget_update_on_the_main_thread(self, app):
        app._log("Done")

        assert [delay for delay, _callback in app.root.after_calls] == [0]

    def test_records_the_session_line_before_the_widget_update_runs(self, app, frozen_now):
        app.root = FakeRoot(run_immediate=False)

        app._log("Deferred")

        assert app._session_log == [f"{FROZEN_TIME_PREFIX} Deferred"]
        assert app.log_text.value == ""

    def test_widget_is_updated_once_the_scheduled_callback_runs(self, app):
        app.root = FakeRoot(run_immediate=False)
        app._log("Deferred")

        app.root.run_pending()

        assert app.log_text.value == "Deferred\n"

    def test_successive_calls_accumulate_lines_in_order(self, app):
        app._log("first")
        app._log("second")

        assert app.log_text.value == "first\nsecond\n"

    def test_writes_nothing_to_disk(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")

        app._log("Done")

        assert list(tmp_path.iterdir()) == []

    def test_an_empty_message_still_appends_a_newline(self, app):
        app._log("")

        assert app.log_text.value == "\n"


class TestInitLogFileCreation:
    def test_creates_the_log_directory_including_missing_parents(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert log_directory.is_dir()

    def test_tolerates_the_log_directory_already_existing(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        log_directory.mkdir(parents=True)

        app._init_log()

        assert pathlib.Path(app._log_path).is_file()

    def test_names_the_file_with_the_day_month_year_and_hour_minute_second_stamp(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert pathlib.Path(app._log_path).name == f"{FROZEN_STAMP}.txt"

    def test_places_the_file_inside_the_configured_log_directory(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert pathlib.Path(app._log_path).parent == log_directory

    def test_the_stamp_shape_is_stable_without_a_frozen_clock(
            self, app, log_directory, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert re.fullmatch(r"\[\d{2}-\d{2}-\d{4}\] - \[\d{2}-\d{2}-\d{2}\]\.txt",
                            pathlib.Path(app._log_path).name)

    def test_the_file_really_exists_on_disk_afterwards(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert pathlib.Path(app._log_path).is_file()

    def test_overwrites_a_file_left_behind_with_the_same_stamp(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        log_directory.mkdir(parents=True)
        (log_directory / f"{FROZEN_STAMP}.txt").write_text("stale content", encoding="utf-8")

        app._init_log()

        assert "stale content" not in header_text(app)

    def test_writes_non_ascii_settings_as_utf8(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app.output_var.set("C:\\salidas\\transcripciones-ñ")

        app._init_log()

        assert "  Output path: C:\\salidas\\transcripciones-ñ" in header_lines(app)

    def test_the_header_ends_with_a_trailing_newline(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert header_text(app).endswith("=" * 60 + "\n")


class TestInitLogHeaderContent:
    def test_announces_the_session_start_time(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert "SESSION STARTED: 2026-07-28 14:05:09" in header_lines(app)

    def test_opens_with_a_rule_of_sixty_equals_signs(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        lines = header_lines(app)
        assert lines[0] == "=" * 60
        assert lines[2] == "=" * 60

    def test_records_the_operating_system(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert f"  OS: {platform.platform()}" in header_text(app)

    def test_records_the_python_version(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert f"  Python: {sys.version}" in header_text(app)

    def test_the_system_block_precedes_the_settings_block(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        lines = header_lines(app)
        assert lines.index("SYSTEM") < lines.index("SETTINGS")

    @pytest.mark.integration
    def test_records_the_installed_memory(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        memory_lines = [line for line in header_lines(app) if line.startswith("  RAM: ")]
        assert len(memory_lines) == 1
        assert re.fullmatch(r"  RAM: \d+\.\d GB total, \d+\.\d GB available", memory_lines[0])


class TestInitLogComponentVersions:
    @pytest.mark.integration
    def test_records_the_installed_component_versions(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        import av
        import ctranslate2
        import faster_whisper

        app._init_log()

        lines = header_lines(app)
        assert f"  PyAV: {av.__version__} (bundled FFmpeg libraries)" in lines
        assert f"  faster-whisper: {faster_whisper.__version__}" in lines
        assert f"  CTranslate2: {ctranslate2.__version__}" in lines

    @pytest.mark.parametrize(
        "unavailable_module", ["av", "ctranslate2", "faster_whisper"])
    def test_degrades_to_a_single_unavailable_line_when_an_import_fails(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi,
            monkeypatch, unavailable_module):
        monkeypatch.setitem(sys.modules, unavailable_module, None)

        app._init_log()

        lines = header_lines(app)
        degraded = [line for line in lines if line.startswith("  Component versions unavailable (")]
        assert len(degraded) == 1
        assert not any(line.startswith("  PyAV:") for line in lines)
        assert not any(line.startswith("  faster-whisper:") for line in lines)
        assert not any(line.startswith("  CTranslate2:") for line in lines)

    def test_still_writes_the_settings_block_when_component_imports_fail(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi,
            monkeypatch):
        monkeypatch.setitem(sys.modules, "av", None)

        app._init_log()

        assert "  Model: large-v3-turbo" in header_lines(app)

    def test_the_file_is_still_created_when_component_imports_fail(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi,
            monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", None)

        app._init_log()

        assert pathlib.Path(app._log_path).is_file()


class TestInitLogGpuProbe:
    def test_records_the_gpu_name_and_video_memory(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            FakeCompletedProcess(0, "no cuda banner here\n"))

        app._init_log()

        assert "  GPU: NVIDIA GeForce RTX 4070 (12.0 GB VRAM)" in header_lines(app)

    def test_records_the_nvidia_driver_version(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            FakeCompletedProcess(0, ""))

        app._init_log()

        assert "  NVIDIA driver: 552.44" in header_lines(app)

    def test_tolerates_nvidia_smi_being_absent(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert "  GPU: not detected" in header_lines(app)

    @pytest.mark.parametrize(
        "raised_error",
        [
            FileNotFoundError("nvidia-smi is not on PATH"),
            OSError("access denied"),
            subprocess.SubprocessError("spawn failed"),
        ],
    )
    def test_any_launch_failure_degrades_to_the_not_detected_line(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch, raised_error):
        install_subprocess_results(monkeypatch, raised_error)

        app._init_log()

        assert "  GPU: not detected" in header_lines(app)

    def test_omits_every_gpu_line_when_nvidia_smi_exits_with_an_error_code(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(monkeypatch, FakeCompletedProcess(9, ""))

        app._init_log()

        assert not any(line.startswith("  GPU:") for line in header_lines(app))

    def test_omits_every_gpu_line_when_nvidia_smi_reports_nothing(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(monkeypatch, FakeCompletedProcess(0, "   \n"))

        app._init_log()

        assert not any(line.startswith("  GPU:") for line in header_lines(app))

    def test_falls_back_to_not_detected_when_the_memory_field_is_missing(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch, FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070\n"))

        lines_before_call = app._init_log()

        assert lines_before_call is None
        assert "  GPU: not detected" in header_lines(app)
        assert not any(line.startswith("  GPU: NVIDIA") for line in header_lines(app))

    def test_falls_back_to_not_detected_when_the_memory_field_is_not_numeric(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch, FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, twelve gigs, 552.44\n"))

        app._init_log()

        assert "  GPU: not detected" in header_lines(app)

    def test_a_missing_driver_field_leaves_both_a_gpu_line_and_the_not_detected_line(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch, FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282\n"))

        app._init_log()

        lines = header_lines(app)
        assert "  GPU: NVIDIA GeForce RTX 4070 (12.0 GB VRAM)" in lines
        assert "  GPU: not detected" in lines
        assert not any(line.startswith("  NVIDIA driver:") for line in lines)

    def test_records_the_cuda_version_reported_in_the_nvidia_smi_banner(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        banner = ("| NVIDIA-SMI 552.44   Driver Version: 552.44   CUDA Version: 12.4     |\n"
                  "|-------------------------------+----------------------+\n")
        install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            FakeCompletedProcess(0, banner))

        app._init_log()

        assert "  CUDA: 12.4" in header_lines(app)

    def test_records_only_the_first_cuda_version_line(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        banner = ("| CUDA Version: 12.4  |\n"
                  "| CUDA Version: 11.8  |\n")
        install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            FakeCompletedProcess(0, banner))

        lines = None
        app._init_log()
        lines = [line for line in header_lines(app) if line.startswith("  CUDA: ")]

        assert lines == ["  CUDA: 12.4"]

    def test_omits_the_cuda_line_when_the_banner_never_mentions_it(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            FakeCompletedProcess(0, "| NVIDIA-SMI 552.44 |\n"))

        app._init_log()

        assert not any(line.startswith("  CUDA: ") for line in header_lines(app))

    def test_tolerates_the_cuda_banner_probe_failing_on_its_own(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            OSError("nvidia-smi died"))

        app._init_log()

        lines = header_lines(app)
        assert "  GPU: NVIDIA GeForce RTX 4070 (12.0 GB VRAM)" in lines
        assert not any(line.startswith("  CUDA: ") for line in lines)

    def test_queries_nvidia_smi_without_flashing_a_console_window(
            self, app, log_directory, frozen_now, recorded_flush, monkeypatch):
        runner = install_subprocess_results(
            monkeypatch,
            FakeCompletedProcess(0, "NVIDIA GeForce RTX 4070, 12282, 552.44\n"),
            FakeCompletedProcess(0, ""))

        app._init_log()

        assert runner.commands[0] == [
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits"]
        assert runner.commands[1] == ["nvidia-smi"]
        assert all(kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
                   for kwargs in runner.keyword_arguments)
        assert all(kwargs["capture_output"] is True for kwargs in runner.keyword_arguments)


class TestInitLogSettingsBlock:
    def test_echoes_the_five_current_option_values(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        lines = header_lines(app)
        settings_index = lines.index("SETTINGS")
        assert lines[settings_index + 1:settings_index + 6] == [
            "  Model: large-v3-turbo",
            "  Language: Spanish",
            "  Device: Auto",
            "  Multilingual: False",
            "  Output path: C:\\sotvox-out",
        ]

    def test_reflects_option_values_changed_before_the_log_was_initialised(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app.model_var.set("tiny")
        app.language_var.set("Auto-detect")
        app.device_var.set("GPU (CUDA)")
        app.multilingual_var.set(True)
        app.output_var.set("D:\\out")

        app._init_log()

        lines = header_lines(app)
        settings_index = lines.index("SETTINGS")
        assert lines[settings_index + 1:settings_index + 6] == [
            "  Model: tiny",
            "  Language: Auto-detect",
            "  Device: GPU (CUDA)",
            "  Multilingual: True",
            "  Output path: D:\\out",
        ]

    def test_the_settings_block_is_the_last_block_before_the_closing_rule(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        lines = header_lines(app)
        assert lines[-1] == "=" * 60
        assert lines[-2] == ""
        assert lines[-3] == "  Output path: C:\\sotvox-out"


class TestInitLogSessionBookkeeping:
    def test_appends_an_application_started_entry_to_the_session_log(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert app._session_log == [f"{FROZEN_TIME_PREFIX} Application started"]

    def test_the_header_itself_is_not_recorded_in_the_session_log(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert not any("SESSION STARTED" in entry for entry in app._session_log)

    def test_flushes_the_session_log_immediately(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert recorded_flush == [app]

    def test_writes_the_started_entry_through_the_real_flush(
            self, app, log_directory, frozen_now, missing_nvidia_smi):
        app._init_log()

        assert header_lines(app)[-1] == f"{FROZEN_TIME_PREFIX} Application started"

    def test_the_real_flush_schedules_the_next_pass(
            self, app, log_directory, frozen_now, missing_nvidia_smi):
        app._init_log()

        assert app.root.after_calls == [(2000, app._flush_log)]

    def test_touches_no_log_widget(
            self, app, log_directory, frozen_now, recorded_flush, missing_nvidia_smi):
        app._init_log()

        assert app.log_text.content == []


class TestFlushLog:
    def test_appends_the_pending_session_lines_to_the_log_file(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["first", "second"]

        app._flush_log()

        assert pathlib.Path(app._log_path).read_text(encoding="utf-8") == "first\nsecond\n"

    def test_creates_the_log_file_when_it_does_not_exist_yet(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["first"]

        app._flush_log()

        assert pathlib.Path(app._log_path).is_file()

    def test_preserves_content_already_present_in_the_file(self, app, tmp_path):
        log_file = tmp_path / "session.txt"
        log_file.write_text("HEADER\n", encoding="utf-8")
        app._log_path = str(log_file)
        app._session_log = ["first"]

        app._flush_log()

        assert log_file.read_text(encoding="utf-8") == "HEADER\nfirst\n"

    def test_advances_the_flush_index_to_the_end_of_the_session_log(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["first", "second", "third"]

        app._flush_log()

        assert app._log_flush_idx == 3

    def test_appends_only_the_lines_added_since_the_previous_flush(self, app, tmp_path):
        log_file = tmp_path / "session.txt"
        app._log_path = str(log_file)
        app._session_log = ["first"]
        app._flush_log()

        app._session_log.append("second")
        app._flush_log()

        assert log_file.read_text(encoding="utf-8") == "first\nsecond\n"

    def test_writes_nothing_when_called_again_with_no_new_lines(self, app, tmp_path):
        log_file = tmp_path / "session.txt"
        app._log_path = str(log_file)
        app._session_log = ["first"]
        app._flush_log()

        app._flush_log()

        assert log_file.read_text(encoding="utf-8") == "first\n"

    def test_writes_nothing_when_the_session_log_is_empty(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")

        app._flush_log()

        assert not pathlib.Path(app._log_path).exists()

    def test_writes_nothing_when_no_log_path_is_configured(self, app, tmp_path):
        app._log_path = None
        app._session_log = ["orphan line"]

        app._flush_log()

        assert list(tmp_path.iterdir()) == []

    def test_leaves_the_flush_index_alone_when_no_log_path_is_configured(self, app):
        app._log_path = None
        app._session_log = ["orphan line"]

        app._flush_log()

        assert app._log_flush_idx == 0

    def test_writes_each_entry_on_its_own_line(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["[10:00:00] one", "[10:00:01] two", "[10:00:02] three"]

        app._flush_log()

        content = pathlib.Path(app._log_path).read_text(encoding="utf-8")
        assert content.splitlines() == app._session_log

    def test_writes_non_ascii_entries_as_utf8(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["Transcripción — ñandú"]

        app._flush_log()

        assert pathlib.Path(app._log_path).read_text(
            encoding="utf-8") == "Transcripción — ñandú\n"

    def test_swallows_a_write_failure(self, app, tmp_path):
        app._log_path = str(tmp_path / "missing-directory" / "session.txt")
        app._session_log = ["first"]

        app._flush_log()

        assert not (tmp_path / "missing-directory").exists()

    def test_leaves_the_flush_index_unchanged_when_the_write_fails(self, app, tmp_path):
        app._log_path = str(tmp_path / "missing-directory" / "session.txt")
        app._session_log = ["first"]

        app._flush_log()

        assert app._log_flush_idx == 0

    def test_retries_the_same_lines_after_a_failed_write(self, app, tmp_path):
        target = tmp_path / "later" / "session.txt"
        app._log_path = str(target)
        app._session_log = ["first"]
        app._flush_log()

        target.parent.mkdir()
        app._flush_log()

        assert target.read_text(encoding="utf-8") == "first\n"

    def test_always_reschedules_itself_two_seconds_later(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["first"]

        app._flush_log()

        assert app.root.after_calls == [(2000, app._flush_log)]

    def test_reschedules_itself_when_there_is_nothing_to_write(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")

        app._flush_log()

        assert app.root.after_calls == [(2000, app._flush_log)]

    def test_reschedules_itself_after_a_failed_write(self, app, tmp_path):
        app._log_path = str(tmp_path / "missing-directory" / "session.txt")
        app._session_log = ["first"]

        app._flush_log()

        assert app.root.after_calls == [(2000, app._flush_log)]

    def test_reschedules_itself_when_no_log_path_is_configured(self, app):
        app._log_path = None

        app._flush_log()

        assert app.root.after_calls == [(2000, app._flush_log)]

    def test_the_reschedule_is_not_run_inline(self, app, tmp_path):
        app._log_path = str(tmp_path / "session.txt")
        app._session_log = ["first"]

        app._flush_log()

        assert len(app.root.after_calls) == 1


class TestOnClose:
    @pytest.mark.parametrize(
        ("elapsed_seconds", "expected_duration"),
        [
            (0.0, "0h 0m 0s"),
            (0.9, "0h 0m 0s"),
            (1.0, "0h 0m 1s"),
            (59.0, "0h 0m 59s"),
            (59.9, "0h 0m 59s"),
            (60.0, "0h 1m 0s"),
            (125.0, "0h 2m 5s"),
            (3599.0, "0h 59m 59s"),
            (3600.0, "1h 0m 0s"),
            (3661.0, "1h 1m 1s"),
            (7325.4, "2h 2m 5s"),
            (90061.0, "25h 1m 1s"),
        ],
    )
    def test_logs_the_session_duration_in_hours_minutes_and_seconds(
            self, app, frozen_now, monkeypatch, elapsed_seconds, expected_duration):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + elapsed_seconds)

        app._on_close()

        assert app._session_log[0] == f"{FROZEN_TIME_PREFIX} Session duration: {expected_duration}"

    def test_measures_the_duration_from_the_recorded_session_start(
            self, app, frozen_now, monkeypatch):
        app._session_start = 5000.0
        monkeypatch.setattr(ui.time, "time", lambda: 5090.0)

        app._on_close()

        assert app._session_log[0] == f"{FROZEN_TIME_PREFIX} Session duration: 0h 1m 30s"

    def test_records_an_application_closed_entry_after_the_duration(
            self, app, frozen_now, monkeypatch):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)

        app._on_close()

        assert app._session_log == [
            f"{FROZEN_TIME_PREFIX} Session duration: 0h 0m 5s",
            f"{FROZEN_TIME_PREFIX} Application closed",
        ]

    def test_appends_the_remaining_session_lines_to_the_log_file(
            self, app, frozen_now, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        log_file = tmp_path / "session.txt"
        log_file.write_text("HEADER\n", encoding="utf-8")
        app._log_path = str(log_file)

        app._on_close()

        assert log_file.read_text(encoding="utf-8").splitlines() == [
            "HEADER",
            f"{FROZEN_TIME_PREFIX} Session duration: 0h 0m 5s",
            f"{FROZEN_TIME_PREFIX} Application closed",
        ]

    def test_appends_only_the_tail_after_the_flush_index(
            self, app, frozen_now, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        log_file = tmp_path / "session.txt"
        app._log_path = str(log_file)
        app._session_log = ["already written", "also written"]
        app._log_flush_idx = 2

        app._on_close()

        assert log_file.read_text(encoding="utf-8").splitlines() == [
            f"{FROZEN_TIME_PREFIX} Session duration: 0h 0m 5s",
            f"{FROZEN_TIME_PREFIX} Application closed",
        ]

    def test_leaves_the_flush_index_untouched(self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        app._log_path = str(tmp_path / "session.txt")

        app._on_close()

        assert app._log_flush_idx == 0

    def test_destroys_the_root_window(self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        app._log_path = str(tmp_path / "session.txt")

        app._on_close()

        assert app.root.destroyed is True

    def test_destroys_the_root_window_even_when_the_final_write_fails(
            self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        app._log_path = str(tmp_path / "missing-directory" / "session.txt")

        app._on_close()

        assert app.root.destroyed is True

    def test_still_records_the_duration_when_the_final_write_fails(
            self, app, frozen_now, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 65.0)
        app._log_path = str(tmp_path / "missing-directory" / "session.txt")

        app._on_close()

        assert app._session_log[0] == f"{FROZEN_TIME_PREFIX} Session duration: 0h 1m 5s"

    def test_writes_no_file_when_no_log_path_is_configured(
            self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        app._log_path = None

        app._on_close()

        assert list(tmp_path.iterdir()) == []

    def test_destroys_the_root_window_when_no_log_path_is_configured(
            self, app, monkeypatch):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        app._log_path = None

        app._on_close()

        assert app.root.destroyed is True

    def test_schedules_nothing_further_on_the_main_loop(self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        app._log_path = str(tmp_path / "session.txt")

        app._on_close()

        assert app.root.after_calls == []

    def test_creates_the_log_file_when_the_session_never_flushed(
            self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(ui.time, "time", lambda: app._session_start + 5.0)
        log_file = tmp_path / "session.txt"
        app._log_path = str(log_file)

        app._on_close()

        assert log_file.is_file()
