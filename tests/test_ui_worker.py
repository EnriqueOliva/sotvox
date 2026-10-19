"""Behaviour of SotvoxApp._toggle_transcription and the background transcription worker."""

import os
from pathlib import Path

import pytest
from doubles import FakeVar, FakeWhisperModel, make_info

import ui
from constants import LANG_MAP

SAMPLE_PROBE = {
    "format": {"format_long_name": "QuickTime / MOV", "duration": "5.000", "bit_rate": 128000},
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480,
         "r_frame_rate": "25.00"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": 48000, "channels": 2},
        {"codec_type": "subtitle", "codec_name": "mov_text"},
    ],
}


class RecordedThread:
    def __init__(self, target, daemon, keyword_arguments):
        self.target = target
        self.daemon = daemon
        self.keyword_arguments = keyword_arguments
        self.start_count = 0

    def start(self):
        self.start_count += 1


class ThreadingModuleDouble:
    def __init__(self):
        self.threads = []

    def Thread(self, target=None, daemon=None, **keyword_arguments):
        recorded = RecordedThread(target, daemon, keyword_arguments)
        self.threads.append(recorded)
        return recorded


class WhisperModelFactoryDouble:
    def __init__(self):
        self.calls = []
        self.failures = {}
        self.models = []

    def __call__(self, model_name, device=None, compute_type=None):
        self.calls.append(
            {"model_name": model_name, "device": device, "compute_type": compute_type})
        failure = self.failures.get(device)
        if failure is not None:
            raise failure
        model = FakeWhisperModel()
        self.models.append(model)
        return model

    @property
    def requested_devices(self):
        return [call["device"] for call in self.calls]


class TranscriptionDouble:
    def __init__(self, default_result):
        self.default_result = default_result
        self.queued_results = []
        self.calls = []
        self.side_effect = None

    def __call__(self, model, audio_path, lang_code=None, on_progress=None,
                 is_cancelled=None, on_chunk=None):
        call = {
            "model": model,
            "audio_path": audio_path,
            "lang_code": lang_code,
            "on_progress": on_progress,
            "is_cancelled": is_cancelled,
            "on_chunk": on_chunk,
        }
        self.calls.append(call)
        if self.side_effect is not None:
            self.side_effect(call)
        if len(self.calls) <= len(self.queued_results):
            outcome = self.queued_results[len(self.calls) - 1]
        else:
            outcome = self.default_result
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def audio_paths(self):
        return [call["audio_path"] for call in self.calls]


class SaveTranscriptDouble:
    def __init__(self):
        self.calls = []
        self.error = None
        self.writes_the_file = True
        self.side_effect = None

    def __call__(self, filepath, full_text, output_dir):
        call = {"filepath": filepath, "full_text": full_text, "output_dir": output_dir}
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        saved_name = os.path.splitext(os.path.basename(filepath))[0] + ".txt"
        if self.writes_the_file:
            Path(output_dir, saved_name).write_text(full_text, encoding="utf-8")
        if self.side_effect is not None:
            self.side_effect(call)
        return saved_name

    @property
    def saved_texts(self):
        return [call["full_text"] for call in self.calls]


class ProbeFileDouble:
    def __init__(self, result):
        self.result = result
        self.error = None
        self.calls = []

    def __call__(self, filepath):
        self.calls.append(filepath)
        if self.error is not None:
            raise self.error
        return self.result


class HasAudioStreamDouble:
    def __init__(self):
        self.default_answer = True
        self.answers_by_name = {}
        self.calls = []

    def __call__(self, filepath):
        self.calls.append(filepath)
        return self.answers_by_name.get(os.path.basename(filepath), self.default_answer)


class GpuPackDouble:
    def __init__(self):
        self.libraries_ready = False
        self.error = None
        self.checks = 0

    def libraries_available(self):
        self.checks += 1
        if self.error is not None:
            raise self.error
        return self.libraries_ready


class WinsoundDouble:
    def __init__(self):
        self.beeps = []

    def MessageBeep(self, sound):
        self.beeps.append(sound)


class WorkerHarness:
    def __init__(self, app, media_directory, output_directory, monkeypatch):
        self.app = app
        self.media_directory = media_directory
        self.output_directory = output_directory
        self.model_factory = WhisperModelFactoryDouble()
        self.transcribe = TranscriptionDouble((["hello world"], make_info(), "ok"))
        self.transcribe_multilingual = TranscriptionDouble(
            (["hola world"], make_info(language="es+en"), "ok"))
        self.save_transcript = SaveTranscriptDouble()
        self.probe_file = ProbeFileDouble(SAMPLE_PROBE)
        self.has_audio_stream = HasAudioStreamDouble()
        self.gpu_pack = GpuPackDouble()
        self.winsound = WinsoundDouble()
        self.threading = ThreadingModuleDouble()

        app.files = []
        app.output_var = FakeVar(str(output_directory))
        app.model_var = FakeVar("small")
        app.device_var = FakeVar("CPU")
        app.language_var = FakeVar("Spanish")
        app.multilingual_var = FakeVar(False)

        monkeypatch.setattr(ui, "WhisperModel", self.model_factory)
        monkeypatch.setattr(ui, "transcribe_audio", self.transcribe)
        monkeypatch.setattr(ui, "transcribe_audio_multilingual", self.transcribe_multilingual)
        monkeypatch.setattr(ui, "save_transcript", self.save_transcript)
        monkeypatch.setattr(ui, "probe_file", self.probe_file)
        monkeypatch.setattr(ui, "has_audio_stream", self.has_audio_stream)
        monkeypatch.setattr(ui, "gpu_pack", self.gpu_pack)
        monkeypatch.setattr(ui, "winsound", self.winsound)
        monkeypatch.setattr(ui, "threading", self.threading)

    def queue_file(self, name, contents=b"media bytes"):
        path = self.media_directory / name
        path.write_bytes(contents)
        self.app.files.append(str(path))
        return str(path)

    def queue_absent_file(self, name):
        path = self.media_directory / name
        self.app.files.append(str(path))
        return str(path)

    def select_gpu_device(self, libraries_ready=True):
        self.app.device_var = FakeVar("GPU (CUDA)")
        self.gpu_pack.libraries_ready = libraries_ready

    def enable_multilingual_mode(self):
        self.app.multilingual_var = FakeVar(True)

    def run(self):
        self.app._transcribe_worker()

    @property
    def log(self):
        return self.app.log_text.value

    @property
    def logged_lines(self):
        return self.log.splitlines()

    @property
    def session_log(self):
        return [entry.split("] ", 1)[1] for entry in self.app._session_log]

    @property
    def status_texts(self):
        return [entry["text"] for entry in self.app.progress_label.config_history]

    @property
    def progress_percent_texts(self):
        return [entry["text"] for entry in self.app.progress_pct.config_history]

    @property
    def transcript_files(self):
        if self.output_directory.is_dir():
            return sorted(entry.name for entry in self.output_directory.iterdir())
        return []

    def transcript_text(self, name):
        return (self.output_directory / name).read_text(encoding="utf-8")

    def session_log_starting_with(self, prefix):
        return [entry for entry in self.session_log if entry.startswith(prefix)]


@pytest.fixture
def threading_module(monkeypatch):
    double = ThreadingModuleDouble()
    monkeypatch.setattr(ui, "threading", double)
    return double


@pytest.fixture
def worker(app, tmp_path, monkeypatch):
    media_directory = tmp_path / "media"
    media_directory.mkdir()
    return WorkerHarness(app, media_directory, tmp_path / "transcripts", monkeypatch)


class TestToggleTranscriptionWithAnEmptyQueue:
    def test_logs_a_hint_when_there_is_nothing_to_transcribe(self, app, threading_module):
        app.files = []

        app._toggle_transcription()

        assert "No files to transcribe. Add some files first." in app.log_text.value.splitlines()

    def test_starts_no_worker_thread_when_there_is_nothing_to_transcribe(
            self, app, threading_module):
        app.files = []

        app._toggle_transcription()

        assert threading_module.threads == []

    def test_leaves_the_transcribing_flag_untouched_when_there_is_nothing_to_transcribe(
            self, app, threading_module):
        app.files = []

        app._toggle_transcription()

        assert app.is_transcribing is False

    def test_leaves_the_button_untouched_when_there_is_nothing_to_transcribe(
            self, app, threading_module):
        app.files = []

        app._toggle_transcription()

        assert app.transcribe_btn.texts == []
        assert app.transcribe_btn.enabled is True


class TestToggleTranscriptionStart:
    def test_marks_the_app_as_transcribing(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.is_transcribing is True

    def test_clears_a_stale_cancellation_flag(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]
        app.cancel_requested = True

        app._toggle_transcription()

        assert app.cancel_requested is False

    def test_turns_the_button_into_a_cancel_button(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.transcribe_btn.texts == ["Cancel"]
        assert app.transcribe_btn.cget("text") == "Cancel"

    def test_leaves_the_button_enabled_while_running(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.transcribe_btn.enabled is True
        assert app.transcribe_btn.enabled_history == []

    def test_starts_exactly_one_thread(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert len(threading_module.threads) == 1
        assert threading_module.threads[0].start_count == 1

    def test_the_started_thread_targets_the_worker_and_is_a_daemon(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        started = threading_module.threads[0]
        assert started.target == app._transcribe_worker
        assert started.daemon is True

    def test_does_not_write_the_empty_queue_hint(self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert "No files to transcribe" not in app.log_text.value

    def test_records_the_run_settings_in_the_session_log(self, app, threading_module):
        app.files = [r"C:\media\a.mp3", r"C:\media\b.mp4"]
        app.model_var = FakeVar("medium")
        app.language_var = FakeVar("English")
        app.multilingual_var = FakeVar(True)

        app._toggle_transcription()

        assert any(
            entry.endswith(
                "Transcription started: 2 file(s), model=medium, lang=English, multilingual=True")
            for entry in app._session_log
        )

    def test_the_worker_body_has_not_run_yet_because_the_thread_is_only_started(
            self, app, threading_module):
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.model is None
        assert app.progress_label.config_history == []


class TestToggleTranscriptionCancel:
    def test_requests_cancellation_while_transcribing(self, app, threading_module):
        app.is_transcribing = True
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.cancel_requested is True

    def test_turns_the_button_into_a_disabled_cancelling_button(self, app, threading_module):
        app.is_transcribing = True
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.transcribe_btn.texts == ["Cancelling..."]
        assert app.transcribe_btn.enabled is False
        assert app.transcribe_btn.enabled_history == [False]

    def test_does_not_start_a_second_thread(self, app, threading_module):
        app.is_transcribing = True
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert threading_module.threads == []

    def test_leaves_the_transcribing_flag_set_for_the_worker_to_clear(
            self, app, threading_module):
        app.is_transcribing = True
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert app.is_transcribing is True

    def test_cancels_even_when_the_queue_is_empty(self, app, threading_module):
        app.is_transcribing = True
        app.files = []

        app._toggle_transcription()

        assert app.cancel_requested is True
        assert "No files to transcribe" not in app.log_text.value

    def test_records_the_cancellation_in_the_session_log(self, app, threading_module):
        app.is_transcribing = True
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()

        assert any(
            entry.endswith("Cancellation requested by user") for entry in app._session_log)

    def test_a_second_cancellation_request_is_idempotent(self, app, threading_module):
        app.is_transcribing = True
        app.files = [r"C:\media\clip.mp3"]

        app._toggle_transcription()
        app._toggle_transcription()

        assert app.cancel_requested is True
        assert app.transcribe_btn.texts == ["Cancelling...", "Cancelling..."]
        assert threading_module.threads == []


class TestWorkerOutputDirectory:
    def test_creates_the_output_directory_when_it_is_missing(self, worker):
        worker.queue_file("clip.mp3")
        assert not worker.output_directory.exists()

        worker.run()

        assert worker.output_directory.is_dir()

    def test_reuses_an_existing_output_directory_and_keeps_its_contents(self, worker):
        worker.output_directory.mkdir()
        (worker.output_directory / "old.txt").write_text("kept", encoding="utf-8")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcript_text("old.txt") == "kept"
        assert worker.transcript_files == ["clip.txt", "old.txt"]

    def test_records_the_output_directory_and_the_queue_in_the_session_log(self, worker):
        first = worker.queue_file("a.mp3")
        second = worker.queue_file("b.mp3")

        worker.run()

        assert f"Output directory: {worker.output_directory}" in worker.session_log
        assert "Files in queue (2):" in worker.session_log
        assert f"  - {first}" in worker.session_log
        assert f"  - {second}" in worker.session_log

    def test_a_failure_creating_the_output_directory_escapes_before_the_finally_block(
            self, worker, tmp_path):
        blocking_file = tmp_path / "blocking-output"
        blocking_file.write_text("not a directory", encoding="utf-8")
        worker.app.output_var = FakeVar(str(blocking_file))
        worker.app.is_transcribing = True
        worker.queue_file("clip.mp3")

        with pytest.raises(FileExistsError):
            worker.run()

        assert worker.app.is_transcribing is True
        assert worker.app.transcribe_btn.texts == []


class TestWorkerModelLoading:
    def test_loads_the_model_when_none_is_loaded_yet(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.model_factory.calls == [
            {"model_name": "small", "device": "cpu", "compute_type": "int8"}]
        assert worker.app.model is worker.model_factory.models[0]

    def test_remembers_the_loaded_model_name_and_device(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app._loaded_model_name == "small"
        assert worker.app._loaded_device == "cpu"

    def test_reuses_the_already_loaded_model_when_the_name_and_device_match(self, worker):
        already_loaded = FakeWhisperModel()
        worker.app.model = already_loaded
        worker.app._loaded_model_name = "small"
        worker.app._loaded_device = "cpu"
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.model_factory.calls == []
        assert worker.app.model is already_loaded
        assert "Model already loaded: small on cpu" in worker.session_log

    def test_hands_the_loaded_model_to_the_transcriber(self, worker):
        already_loaded = FakeWhisperModel()
        worker.app.model = already_loaded
        worker.app._loaded_model_name = "small"
        worker.app._loaded_device = "cpu"
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcribe.calls[0]["model"] is already_loaded

    def test_reloads_the_model_when_the_selected_name_differs(self, worker):
        stale_model = FakeWhisperModel()
        worker.app.model = stale_model
        worker.app._loaded_model_name = "tiny"
        worker.app._loaded_device = "cpu"
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.model_factory.calls[0]["model_name"] == "small"
        assert worker.app.model is not stale_model
        assert worker.app._loaded_model_name == "small"

    def test_reloads_the_model_when_the_resolved_device_differs(self, worker):
        stale_model = FakeWhisperModel()
        worker.app.model = stale_model
        worker.app._loaded_model_name = "small"
        worker.app._loaded_device = "cuda"
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.model_factory.requested_devices == ["cpu"]
        assert worker.app.model is not stale_model
        assert worker.app._loaded_device == "cpu"

    def test_reloads_the_model_when_the_instance_is_missing_although_the_labels_match(
            self, worker):
        worker.app.model = None
        worker.app._loaded_model_name = "small"
        worker.app._loaded_device = "cpu"
        worker.queue_file("clip.mp3")

        worker.run()

        assert len(worker.model_factory.calls) == 1
        assert worker.app.model is worker.model_factory.models[0]

    def test_records_the_model_load_duration_in_the_session_log(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.session_log_starting_with("Model loaded in ")

    def test_announces_the_cpu_device_label(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Model 'small' loaded on CPU" in worker.logged_lines

    def test_announces_the_gpu_device_label(self, worker):
        worker.select_gpu_device()
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Model 'small' loaded on GPU (CUDA)" in worker.logged_lines
        assert worker.model_factory.calls == [
            {"model_name": "small", "device": "cuda", "compute_type": "float16"}]

    def test_a_gpu_request_without_installed_libraries_loads_on_the_cpu(self, worker):
        worker.select_gpu_device(libraries_ready=False)
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.model_factory.requested_devices == ["cpu"]
        assert "Model 'small' loaded on CPU" in worker.logged_lines

    def test_updates_the_status_while_the_model_is_loading(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[0] == "Loading model 'small'..."

    def test_resets_the_progress_bar_before_loading(self, worker):
        worker.app._progress_value = 42
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.progress_percent_texts[0] == ""

    def test_records_the_requested_load_parameters_in_the_session_log(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Loading model: name=small, device=cpu, compute_type=int8" in worker.session_log


class TestWorkerCudaFallback:
    def test_falls_back_to_a_cpu_model_when_the_cuda_load_fails(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.model_factory.requested_devices == ["cuda", "cpu"]
        assert worker.model_factory.calls[1]["compute_type"] == "int8"

    def test_logs_the_cuda_failure_and_the_fallback(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "CUDA load failed (cudnn missing), falling back to CPU" in worker.logged_lines

    def test_records_the_cuda_traceback_in_the_session_log(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.queue_file("clip.mp3")

        worker.run()

        traceback_entries = worker.session_log_starting_with("CUDA load traceback:")
        assert traceback_entries
        assert "RuntimeError: cudnn missing" in traceback_entries[0]

    def test_announces_the_cpu_label_after_a_fallback(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Model 'small' loaded on CPU" in worker.logged_lines
        assert "Model 'small' loaded on GPU (CUDA)" not in worker.logged_lines

    def test_transcription_still_runs_after_a_fallback(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcript_files == ["clip.txt"]
        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_the_fallback_device_is_remembered_so_the_next_run_reuses_the_model(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.queue_file("clip.mp3")
        worker.run()

        worker.app.device_var = FakeVar("CPU")
        worker.run()

        assert worker.model_factory.requested_devices == ["cuda", "cpu"]
        assert worker.session_log.count("Model already loaded: small on cpu") == 1

    def test_a_cpu_load_failure_reaches_the_fatal_handler(self, worker):
        worker.model_factory.failures["cpu"] = RuntimeError("model file corrupt")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Fatal error: model file corrupt" in worker.logged_lines
        assert worker.app.model is None
        assert worker.transcribe.calls == []

    def test_a_cpu_load_failure_is_not_described_as_a_cuda_fallback(self, worker):
        worker.model_factory.failures["cpu"] = RuntimeError("model file corrupt")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "falling back to CPU" not in worker.log
        assert worker.model_factory.requested_devices == ["cpu"]

    def test_a_failing_fallback_load_reaches_the_fatal_handler(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.model_factory.failures["cpu"] = RuntimeError("no cpu backend either")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "CUDA load failed (cudnn missing), falling back to CPU" in worker.logged_lines
        assert "Fatal error: no cpu backend either" in worker.logged_lines
        assert worker.winsound.beeps == [ui.SOUND_FAILURE]


class TestWorkerSpeedHint:
    SPEED_HINT = ("Note: large-v3 on CPU is slow. Try large-v3-turbo — much faster, "
                  "almost as accurate.")

    def test_logs_the_speed_hint_for_large_v3_on_the_cpu(self, worker):
        worker.app.model_var = FakeVar("large-v3")
        worker.queue_file("clip.mp3")

        worker.run()

        assert self.SPEED_HINT in worker.logged_lines

    @pytest.mark.parametrize("model_name", ["large-v3-turbo", "large-v2", "medium", "small"])
    def test_does_not_log_the_speed_hint_for_other_cpu_models(self, worker, model_name):
        worker.app.model_var = FakeVar(model_name)
        worker.queue_file("clip.mp3")

        worker.run()

        assert self.SPEED_HINT not in worker.logged_lines

    def test_does_not_log_the_speed_hint_for_large_v3_on_the_gpu(self, worker):
        worker.select_gpu_device()
        worker.app.model_var = FakeVar("large-v3")
        worker.queue_file("clip.mp3")

        worker.run()

        assert self.SPEED_HINT not in worker.logged_lines

    def test_logs_the_speed_hint_for_large_v3_after_a_cuda_fallback(self, worker):
        worker.select_gpu_device()
        worker.model_factory.failures["cuda"] = RuntimeError("cudnn missing")
        worker.app.model_var = FakeVar("large-v3")
        worker.queue_file("clip.mp3")

        worker.run()

        assert self.SPEED_HINT in worker.logged_lines


class TestWorkerLanguageSelection:
    def test_auto_detect_maps_to_no_language_code(self, worker):
        worker.app.language_var = FakeVar("Auto-detect")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcribe.calls[0]["lang_code"] is None

    @pytest.mark.parametrize("language_name", ["Spanish", "English", "Japanese", "Portuguese"])
    def test_a_named_language_maps_to_its_code(self, worker, language_name):
        worker.app.language_var = FakeVar(language_name)
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcribe.calls[0]["lang_code"] == LANG_MAP[language_name]

    def test_an_unknown_language_name_maps_to_no_language_code(self, worker):
        worker.app.language_var = FakeVar("Klingon")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcribe.calls[0]["lang_code"] is None

    def test_records_the_language_code_in_the_session_log(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Transcribing audio (lang_code=es)..." in worker.session_log

    def test_reports_the_selected_language_in_the_done_line(self, worker):
        worker.transcribe.default_result = (["hola"], make_info(language="en"), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "· Spanish · clip.txt" in worker.log

    def test_reports_the_detected_language_in_the_done_line_for_auto_detect(self, worker):
        worker.app.language_var = FakeVar("Auto-detect")
        worker.transcribe.default_result = (["hola"], make_info(language="pt"), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "· pt · clip.txt" in worker.log

    def test_does_not_announce_multilingual_mode_when_it_is_off(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Multilingual mode active — language selection will be ignored" \
            not in worker.session_log


class TestWorkerMultilingualMode:
    def test_routes_to_the_multilingual_transcriber(self, worker):
        worker.enable_multilingual_mode()
        source = worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcribe_multilingual.audio_paths == [source]
        assert worker.transcribe.calls == []

    def test_ignores_the_selected_language(self, worker):
        worker.enable_multilingual_mode()
        worker.app.language_var = FakeVar("Spanish")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcribe_multilingual.calls[0]["lang_code"] is None

    def test_announces_that_the_language_selection_is_ignored(self, worker):
        worker.enable_multilingual_mode()
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Multilingual mode active — language selection will be ignored" \
            in worker.session_log

    def test_records_the_chunk_mode_in_the_session_log(self, worker):
        worker.enable_multilingual_mode()
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Transcribing audio (multilingual mode, 30s chunks)..." in worker.session_log

    def test_the_chunk_callback_records_each_detected_language(self, worker):
        worker.enable_multilingual_mode()

        def report_two_chunks(call):
            call["on_chunk"](1, 2, 0.0, 30.0, "es", 0.9912)
            call["on_chunk"](2, 2, 30.0, 45.5, "en", 0.5)

        worker.transcribe_multilingual.side_effect = report_two_chunks
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Chunk [1/2] 0.0s-30.0s: language=es (probability=0.9912)" in worker.session_log
        assert "  Chunk [2/2] 30.0s-45.5s: language=en (probability=0.5000)" in worker.session_log

    def test_reports_the_detected_languages_in_the_done_line(self, worker):
        worker.enable_multilingual_mode()
        worker.app.language_var = FakeVar("Spanish")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "· es+en · clip.txt" in worker.log

    def test_writes_the_multilingual_text(self, worker):
        worker.enable_multilingual_mode()
        worker.transcribe_multilingual.default_result = (
            ["hola", "hello"], make_info(language="es+en"), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcript_text("clip.txt") == "hola\nhello"


class TestWorkerVideoProbing:
    @pytest.mark.parametrize("extension", [".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv"])
    def test_video_files_are_probed_for_an_audio_track(self, worker, extension):
        source = worker.queue_file(f"clip{extension}")

        worker.run()

        assert worker.probe_file.calls == [source]
        assert worker.has_audio_stream.calls == [source]

    @pytest.mark.parametrize("extension", [".mp3", ".wav", ".m4a", ".flac"])
    def test_audio_files_are_not_probed_at_all(self, worker, extension):
        worker.queue_file(f"clip{extension}")

        worker.run()

        assert worker.probe_file.calls == []
        assert worker.has_audio_stream.calls == []

    def test_a_video_without_an_audio_track_is_counted_as_failed(self, worker):
        worker.has_audio_stream.default_answer = False
        worker.queue_file("silent.mp4")

        worker.run()

        assert "  Error: Video has no audio track — nothing to transcribe" in worker.logged_lines
        assert worker.status_texts[-1] == "Failed — 1 file(s) had errors"

    def test_a_video_without_an_audio_track_is_never_transcribed_or_written(self, worker):
        worker.has_audio_stream.default_answer = False
        worker.queue_file("silent.mp4")

        worker.run()

        assert worker.transcribe.calls == []
        assert worker.save_transcript.calls == []
        assert worker.transcript_files == []

    def test_a_video_without_an_audio_track_does_not_stop_the_remaining_files(self, worker):
        worker.has_audio_stream.answers_by_name = {"silent.mp4": False}
        worker.queue_file("silent.mp4")
        worker.queue_file("good.mp3")

        worker.run()

        assert worker.transcript_files == ["good.txt"]
        assert worker.status_texts[-1] == "Done — 1 transcribed, 1 failed"

    def test_the_probe_metadata_is_written_to_the_session_log(self, worker):
        worker.queue_file("clip.mp4")

        worker.run()

        assert "  Probe format: QuickTime / MOV, duration=5.000s, bitrate=128000 bps" \
            in worker.session_log
        assert "  Probe stream [video]: h264 640x480 @ 25.00 fps" in worker.session_log
        assert "  Probe stream [audio]: aac 48000 Hz, 2 ch" in worker.session_log
        assert "  Probe stream [subtitle]: mov_text" in worker.session_log

    def test_missing_probe_fields_are_reported_as_placeholders(self, worker):
        worker.probe_file.result = {"format": {}, "streams": [{}]}
        worker.queue_file("clip.mp4")

        worker.run()

        assert "  Probe format: unknown, duration=?s, bitrate=? bps" in worker.session_log
        assert "  Probe stream [?]: ?" in worker.session_log

    def test_a_probe_returning_nothing_reports_a_metadata_failure(self, worker):
        worker.probe_file.result = None
        worker.queue_file("clip.mp4")

        worker.run()

        assert "  Probe: failed to read file metadata" in worker.session_log

    def test_a_probe_returning_nothing_still_transcribes_the_video(self, worker):
        worker.probe_file.result = None
        worker.queue_file("clip.mp4")

        worker.run()

        assert worker.transcript_files == ["clip.txt"]
        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_a_probe_raising_is_swallowed_as_non_fatal(self, worker):
        worker.probe_file.error = RuntimeError("pyav blew up")
        worker.queue_file("clip.mp4")

        worker.run()

        assert "  Probe error (non-fatal): pyav blew up" in worker.session_log
        assert "  Error: pyav blew up" not in worker.logged_lines

    def test_transcription_proceeds_after_a_raising_probe(self, worker):
        worker.probe_file.error = RuntimeError("pyav blew up")
        worker.queue_file("clip.mp4")

        worker.run()

        assert worker.has_audio_stream.calls
        assert worker.transcript_files == ["clip.txt"]
        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_an_uppercase_video_extension_is_still_treated_as_a_video(self, worker):
        source = worker.queue_file("clip.MP4")

        worker.run()

        assert worker.probe_file.calls == [source]


class TestWorkerTranscriptionStatuses:
    def test_a_timed_out_file_is_skipped_without_writing(self, worker):
        worker.transcribe.default_result = (["partial text"], make_info(), "timed_out")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Timed out, skipping" in worker.logged_lines
        assert worker.save_transcript.calls == []
        assert worker.transcript_files == []

    def test_a_timed_out_file_counts_as_neither_a_success_nor_a_failure(self, worker):
        worker.transcribe.default_result = (["partial text"], make_info(), "timed_out")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Complete — 0 file(s) transcribed"

    def test_a_retry_timed_out_file_is_logged_but_still_written(self, worker):
        worker.transcribe.default_result = (["partial text"], make_info(), "retry_timed_out")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Retry timed out, skipping" in worker.logged_lines
        assert worker.transcript_text("clip.txt") == "partial text"

    def test_a_retry_timed_out_file_counts_as_a_success(self, worker):
        worker.transcribe.default_result = (["partial text"], make_info(), "retry_timed_out")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_only_the_retry_timeout_of_the_two_timeout_statuses_writes_a_transcript(self, worker):
        worker.queue_file("gave-up.mp3")
        worker.queue_file("partial.mp3")
        worker.transcribe.queued_results = [
            (["lost text"], make_info(), "timed_out"),
            (["kept text"], make_info(), "retry_timed_out"),
        ]

        worker.run()

        assert worker.transcript_files == ["partial.txt"]
        assert worker.save_transcript.saved_texts == ["kept text"]

    def test_an_empty_result_is_skipped_without_writing(self, worker):
        worker.transcribe.default_result = ([], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  No speech detected, skipping" in worker.logged_lines
        assert worker.save_transcript.calls == []
        assert worker.transcript_files == []

    @pytest.mark.parametrize("segments", [[""], ["   "], ["\n", "\t"], ["", " "]])
    def test_whitespace_only_text_is_skipped_without_writing(self, worker, segments):
        worker.transcribe.default_result = (segments, make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  No speech detected, skipping" in worker.logged_lines
        assert worker.transcript_files == []

    def test_a_skipped_empty_file_counts_as_neither_a_success_nor_a_failure(self, worker):
        worker.transcribe.default_result = ([], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Complete — 0 file(s) transcribed"

    def test_text_surrounded_by_whitespace_is_written_verbatim(self, worker):
        worker.transcribe.default_result = (["  spoken  "], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcript_text("clip.txt") == "  spoken  "

    def test_records_the_result_status_and_segment_count_in_the_session_log(self, worker):
        worker.transcribe.default_result = (["one", "two"], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Transcription result: status=ok, segments=2" in worker.session_log

    def test_records_the_detected_language_probability_in_the_session_log(self, worker):
        worker.transcribe.default_result = (
            ["one"], make_info(duration=42.0, language="pt", language_probability=0.8125), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Audio duration: 42.0s" in worker.session_log
        assert "  Detected language: pt (probability: 0.8125)" in worker.session_log

    def test_records_the_duration_after_vad_when_it_is_reported(self, worker):
        worker.transcribe.default_result = (
            ["one"], make_info(duration_after_vad=8.25), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Duration after VAD: 8.2s" in worker.session_log

    def test_omits_the_duration_after_vad_when_it_is_zero(self, worker):
        worker.transcribe.default_result = (["one"], make_info(duration_after_vad=0.0), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.session_log_starting_with("  Duration after VAD:") == []

    def test_omits_the_duration_after_vad_when_the_attribute_is_absent(self, worker):
        worker.transcribe.default_result = (["one"], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.session_log_starting_with("  Duration after VAD:") == []


class TestWorkerSuccessfulFile:
    def test_announces_every_queued_file(self, worker):
        worker.queue_file("first.mp3")
        worker.queue_file("second.mp3")

        worker.run()

        assert "Processing: first.mp3" in worker.logged_lines
        assert "Processing: second.mp3" in worker.logged_lines

    def test_numbers_each_file_in_the_status_line(self, worker):
        worker.queue_file("first.mp3")
        worker.queue_file("second.mp3")

        worker.run()

        assert "[1/2] first.mp3" in worker.status_texts
        assert "[2/2] second.mp3" in worker.status_texts

    def test_processes_the_queued_files_in_order(self, worker):
        first = worker.queue_file("first.mp3")
        second = worker.queue_file("second.mp3")
        third = worker.queue_file("third.mp3")

        worker.run()

        assert worker.transcribe.audio_paths == [first, second, third]

    def test_joins_the_segments_with_newlines_when_writing(self, worker):
        worker.transcribe.default_result = (["one", "two", "three"], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcript_text("clip.txt") == "one\ntwo\nthree"

    def test_saves_into_the_configured_output_directory(self, worker):
        source = worker.queue_file("clip.mp3")

        worker.run()

        assert worker.save_transcript.calls == [
            {"filepath": source, "full_text": "hello world",
             "output_dir": str(worker.output_directory)}]

    def test_counts_the_written_file_as_a_success(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_reports_the_saved_name_in_the_done_line(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert any(
            line.startswith("  Done in ") and line.endswith("· Spanish · clip.txt")
            for line in worker.logged_lines
        )

    def test_records_the_written_text_length_in_the_session_log(self, worker):
        worker.transcribe.default_result = (["12345"], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Text length: 5 chars" in worker.session_log

    def test_records_the_written_output_file_size_in_the_session_log(self, worker):
        worker.transcribe.default_result = (["12345"], make_info(), "ok")
        worker.queue_file("clip.mp3")

        worker.run()

        expected = f"  Output file: {worker.output_directory / 'clip.txt'} (5.0 B)"
        assert expected in worker.session_log

    def test_a_missing_output_file_only_costs_the_size_line(self, worker):
        worker.save_transcript.writes_the_file = False
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.session_log_starting_with("  Output file:") == []
        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_records_the_source_file_size_in_the_session_log(self, worker):
        worker.queue_file("clip.mp3", contents=b"0123456789")

        worker.run()

        assert "  File size: 10.0 B" in worker.session_log

    def test_a_missing_source_file_only_costs_the_size_line(self, worker):
        worker.queue_absent_file("ghost.mp3")

        worker.run()

        assert worker.session_log_starting_with("  File size:") == []
        assert worker.status_texts[-1] == "Complete — 1 file(s) transcribed"

    def test_records_the_full_source_path_in_the_session_log(self, worker):
        source = worker.queue_file("clip.mp3")

        worker.run()

        assert f"  Full path: {source}" in worker.session_log


class TestWorkerFileFailures:
    def test_a_transcription_error_is_caught_and_logged(self, worker):
        worker.transcribe.default_result = RuntimeError("decoder exploded")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Error: decoder exploded" in worker.logged_lines
        assert worker.session_log_starting_with("  Traceback:")

    def test_a_transcription_error_is_counted_as_a_failure(self, worker):
        worker.transcribe.default_result = RuntimeError("decoder exploded")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Failed — 1 file(s) had errors"
        assert worker.transcript_files == []

    def test_a_transcription_error_is_not_treated_as_fatal(self, worker):
        worker.transcribe.default_result = RuntimeError("decoder exploded")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Fatal error" not in worker.log
        assert "All done." in worker.log

    def test_the_loop_continues_after_a_failing_file(self, worker):
        worker.queue_file("broken.mp3")
        worker.queue_file("fine.mp3")
        worker.transcribe.queued_results = [
            RuntimeError("decoder exploded"), (["good"], make_info(), "ok")]

        worker.run()

        assert len(worker.transcribe.calls) == 2
        assert worker.transcript_files == ["fine.txt"]

    def test_a_mixed_run_reports_both_counts(self, worker):
        worker.queue_file("broken.mp3")
        worker.queue_file("fine.mp3")
        worker.transcribe.queued_results = [
            RuntimeError("decoder exploded"), (["good"], make_info(), "ok")]

        worker.run()

        assert worker.status_texts[-1] == "Done — 1 transcribed, 1 failed"

    def test_a_failure_in_the_last_file_does_not_hide_earlier_successes(self, worker):
        worker.queue_file("fine.mp3")
        worker.queue_file("broken.mp3")
        worker.transcribe.queued_results = [
            (["good"], make_info(), "ok"), RuntimeError("decoder exploded")]

        worker.run()

        assert worker.transcript_files == ["fine.txt"]
        assert worker.status_texts[-1] == "Done — 1 transcribed, 1 failed"

    def test_a_save_failure_is_counted_as_a_file_failure(self, worker):
        worker.save_transcript.error = OSError("disk full")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "  Error: disk full" in worker.logged_lines
        assert worker.status_texts[-1] == "Failed — 1 file(s) had errors"

    def test_every_file_failing_reports_the_failed_status(self, worker):
        worker.transcribe.default_result = RuntimeError("decoder exploded")
        worker.queue_file("one.mp3")
        worker.queue_file("two.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Failed — 2 file(s) had errors"
        assert worker.winsound.beeps == [ui.SOUND_FAILURE]


class TestWorkerCancellation:
    def test_cancelling_before_the_first_file_breaks_the_loop(self, worker):
        worker.app.cancel_requested = True
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Transcription cancelled." in worker.logged_lines
        assert worker.transcribe.calls == []

    def test_cancelling_between_files_stops_the_remaining_work(self, worker):
        worker.queue_file("first.mp3")
        worker.queue_file("second.mp3")
        worker.save_transcript.side_effect = lambda call: setattr(
            worker.app, "cancel_requested", True)

        worker.run()

        assert worker.transcribe.audio_paths == [worker.app.files[0]]
        assert worker.transcript_files == ["first.txt"]
        assert "Transcription cancelled." in worker.logged_lines

    def test_cancelling_during_transcription_skips_the_write(self, worker):
        worker.queue_file("clip.mp3")
        worker.transcribe.side_effect = lambda call: setattr(
            worker.app, "cancel_requested", True)

        worker.run()

        assert worker.save_transcript.calls == []
        assert worker.transcript_files == []

    def test_cancelling_during_transcription_does_not_log_the_loop_cancellation_line(
            self, worker):
        worker.queue_file("clip.mp3")
        worker.transcribe.side_effect = lambda call: setattr(
            worker.app, "cancel_requested", True)

        worker.run()

        assert "Transcription cancelled." not in worker.logged_lines

    def test_a_timed_out_file_bypasses_the_mid_loop_cancellation_check(self, worker):
        worker.queue_file("first.mp3")
        worker.queue_file("second.mp3")
        worker.transcribe.default_result = (["partial"], make_info(), "timed_out")
        worker.transcribe.side_effect = lambda call: setattr(
            worker.app, "cancel_requested", True)

        worker.run()

        assert "  Timed out, skipping" in worker.logged_lines
        assert "Transcription cancelled." in worker.logged_lines
        assert len(worker.transcribe.calls) == 1

    def test_nothing_is_announced_when_the_run_is_cancelled(self, worker):
        worker.app.cancel_requested = True
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts == ["Loading model 'small'..."]
        assert "All done." not in worker.log

    def test_no_sound_plays_when_the_run_is_cancelled(self, worker):
        worker.app.cancel_requested = True
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.winsound.beeps == []

    def test_the_cancellation_callback_reflects_the_live_flag(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        is_cancelled = worker.transcribe.calls[0]["is_cancelled"]
        assert is_cancelled() is False
        worker.app.cancel_requested = True
        assert is_cancelled() is True

    def test_the_multilingual_transcriber_also_receives_a_cancellation_callback(self, worker):
        worker.enable_multilingual_mode()
        worker.queue_file("clip.mp3")

        worker.run()

        is_cancelled = worker.transcribe_multilingual.calls[0]["is_cancelled"]
        worker.app.cancel_requested = True
        assert is_cancelled() is True


class TestWorkerProgress:
    def test_progress_reaches_one_hundred_after_a_complete_run(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app._progress_value == 100
        assert worker.progress_percent_texts[-1] == "100%"

    def test_progress_never_reaches_one_hundred_when_cancelled(self, worker):
        worker.app.cancel_requested = True
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app._progress_value == 0
        assert "100%" not in worker.progress_percent_texts

    def test_progress_steps_once_per_file_before_completing(self, worker):
        worker.queue_file("first.mp3")
        worker.queue_file("second.mp3")

        worker.run()

        assert worker.progress_percent_texts == ["", "", "50%", "100%"]

    def test_the_segment_callback_clamps_to_ninety_nine_at_the_end_of_a_file(self, worker):
        worker.queue_file("clip.mp3")
        worker.transcribe.side_effect = lambda call: call["on_progress"](10.0, 10.0)

        worker.run()

        assert "99%" in worker.progress_percent_texts
        assert worker.progress_percent_texts.index("99%") \
            < worker.progress_percent_texts.index("100%")

    def test_the_segment_callback_clamps_an_overshooting_segment_end(self, worker):
        worker.queue_file("clip.mp3")
        observed = []
        worker.transcribe.side_effect = lambda call: (
            call["on_progress"](40.0, 10.0), observed.append(worker.app._progress_value))

        worker.run()

        assert observed == [99]

    def test_the_segment_callback_scales_by_the_position_in_the_queue(self, worker):
        worker.queue_file("first.mp3")
        worker.queue_file("second.mp3")
        observed = []
        worker.transcribe.side_effect = lambda call: (
            call["on_progress"](5.0, 10.0), observed.append(worker.app._progress_value))

        worker.run()

        assert observed == [25.0, 75.0]

    def test_the_multilingual_transcriber_receives_the_same_progress_callback(self, worker):
        worker.enable_multilingual_mode()
        worker.queue_file("clip.mp3")
        worker.transcribe_multilingual.side_effect = lambda call: call["on_progress"](5.0, 10.0)

        worker.run()

        assert "50%" in worker.progress_percent_texts


class TestWorkerCompletion:
    def test_a_fully_successful_run_reports_complete_and_plays_the_success_sound(self, worker):
        worker.queue_file("one.mp3")
        worker.queue_file("two.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Complete — 2 file(s) transcribed"
        assert worker.winsound.beeps == [ui.SOUND_SUCCESS]

    def test_a_fully_failed_run_reports_failed_and_plays_the_failure_sound(self, worker):
        worker.transcribe.default_result = RuntimeError("decoder exploded")
        worker.queue_file("one.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Failed — 1 file(s) had errors"
        assert worker.winsound.beeps == [ui.SOUND_FAILURE]

    def test_a_mixed_run_reports_a_summary_and_plays_the_success_sound(self, worker):
        worker.queue_file("broken.mp3")
        worker.queue_file("fine.mp3")
        worker.transcribe.queued_results = [
            RuntimeError("decoder exploded"), (["good"], make_info(), "ok")]

        worker.run()

        assert worker.status_texts[-1] == "Done — 1 transcribed, 1 failed"
        assert worker.winsound.beeps == [ui.SOUND_SUCCESS]

    def test_the_closing_line_names_the_output_directory(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert f"All done. Output: {worker.output_directory}" in worker.logged_lines

    def test_the_final_status_is_mirrored_in_both_status_labels(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.progress_label.cget("text") == "Complete — 1 file(s) transcribed"
        assert worker.app.status_left.cget("text") == "Complete — 1 file(s) transcribed"

    def test_an_empty_queue_still_loads_the_model_and_reports_completion(self, worker):
        worker.run()

        assert len(worker.model_factory.calls) == 1
        assert worker.status_texts[-1] == "Complete — 0 file(s) transcribed"
        assert worker.winsound.beeps == [ui.SOUND_SUCCESS]


class TestWorkerFatalErrors:
    def test_a_failure_resolving_the_device_is_reported_as_fatal(self, worker):
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "Fatal error: gpu pack unreadable" in worker.logged_lines
        assert worker.model_factory.calls == []

    def test_a_fatal_error_sets_the_error_status_and_plays_the_failure_sound(self, worker):
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.status_texts[-1] == "Error occurred"
        assert worker.winsound.beeps == [ui.SOUND_FAILURE]

    def test_a_fatal_error_records_the_traceback_in_the_session_log(self, worker):
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        traceback_entries = worker.session_log_starting_with("Fatal traceback:")
        assert traceback_entries
        assert "RuntimeError: gpu pack unreadable" in traceback_entries[0]

    def test_a_fatal_error_suppresses_the_completion_announcement(self, worker):
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert "All done." not in worker.log
        assert "Complete" not in worker.log

    def test_a_fatal_error_leaves_the_progress_bar_short_of_one_hundred(self, worker):
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app._progress_value == 0
        assert "100%" not in worker.progress_percent_texts

    def test_a_fatal_error_does_not_escape_the_worker(self, worker):
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.transcript_files == []


class TestWorkerFinallyBlock:
    def test_clears_the_transcribing_flag_after_a_clean_run(self, worker):
        worker.app.is_transcribing = True
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.is_transcribing is False

    def test_clears_the_cancellation_flag_after_a_clean_run(self, worker):
        worker.app.is_transcribing = True
        worker.queue_file("clip.mp3")
        worker.transcribe.side_effect = lambda call: setattr(
            worker.app, "cancel_requested", True)

        worker.run()

        assert worker.app.cancel_requested is False

    def test_restores_the_button_after_a_clean_run(self, worker):
        worker.app.transcribe_btn.set_text("Cancel")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.transcribe_btn.texts[-1] == "Transcribe"
        assert worker.app.transcribe_btn.enabled is True
        assert worker.app.transcribe_btn.enabled_history[-1] is True

    def test_clears_the_flags_after_a_fatal_error(self, worker):
        worker.app.is_transcribing = True
        worker.app.cancel_requested = True
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.is_transcribing is False
        assert worker.app.cancel_requested is False

    def test_restores_the_button_after_a_fatal_error(self, worker):
        worker.app.transcribe_btn.set_text("Cancelling...")
        worker.app.transcribe_btn.set_enabled(False)
        worker.gpu_pack.error = RuntimeError("gpu pack unreadable")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.transcribe_btn.texts[-1] == "Transcribe"
        assert worker.app.transcribe_btn.enabled is True

    def test_restores_the_button_after_a_cancelled_run(self, worker):
        worker.app.is_transcribing = True
        worker.app.cancel_requested = True
        worker.app.transcribe_btn.set_text("Cancelling...")
        worker.app.transcribe_btn.set_enabled(False)
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.transcribe_btn.texts[-1] == "Transcribe"
        assert worker.app.transcribe_btn.enabled is True

    def test_restores_the_button_after_a_run_where_every_file_failed(self, worker):
        worker.transcribe.default_result = RuntimeError("decoder exploded")
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.transcribe_btn.texts[-1] == "Transcribe"
        assert worker.app.transcribe_btn.enabled is True

    def test_schedules_the_button_restoration_on_the_ui_thread(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()

        assert worker.app.root.after_calls[-1][0] == 0
        assert worker.app.transcribe_btn.texts[-1] == "Transcribe"

    def test_a_second_run_is_possible_after_the_flags_are_reset(self, worker):
        worker.queue_file("clip.mp3")

        worker.run()
        worker.run()

        assert len(worker.transcribe.calls) == 2
        assert worker.app.is_transcribing is False
