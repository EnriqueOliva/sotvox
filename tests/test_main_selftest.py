"""Unit tests for the diagnostic self-test report writer in src/main.py."""

import av
import ctranslate2
import faster_whisper
import pytest
from doubles import FakeSegment, FakeWhisperModel, make_info, write_silent_wav

import constants
import engine
import gpu_pack

DEFAULT_MEDIA_PATH = r"C:\media\sample.mp4"
REPORT_FILENAME = "selftest.txt"
FAILURE_MESSAGE = "self-test step exploded"
TRACEBACK_HEADER = "Traceback (most recent call last):"
PROBE_RESULT = {
    "format": {"format_long_name": "QuickTime / MOV", "duration": "12.500", "bit_rate": 128000},
    "streams": [],
}


def raise_error(error):
    def raiser(*args, **kwargs):
        raise error

    return raiser


def return_value(value):
    def returner(*args, **kwargs):
        return value

    return returner


class RecordingModelFactory:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.constructor_calls = []

    def __call__(self, model_name, device=None, compute_type=None):
        self.constructor_calls.append(
            {"model_name": model_name, "device": device, "compute_type": compute_type})
        index = len(self.constructor_calls) - 1
        if index >= len(self.outcomes):
            raise AssertionError(f"unexpected WhisperModel construction #{index + 1}")
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def devices(self):
        return [call["device"] for call in self.constructor_calls]


class CountingCudaDeviceProbe:
    def __init__(self, device_count=0, error=None):
        self.device_count = device_count
        self.error = error
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.device_count


class CountingLibraryProbe:
    def __init__(self, available=False, error=None):
        self.available = available
        self.error = error
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.available


class SelftestHarness:
    def __init__(self, main_module, monkeypatch, log_directory, factory, cpu_model,
                 device_probe, library_probe, real_media_probes):
        self.main_module = main_module
        self.monkeypatch = monkeypatch
        self.log_directory = log_directory
        self.factory = factory
        self.cpu_model = cpu_model
        self.device_probe = device_probe
        self.library_probe = library_probe
        self.real_media_probes = real_media_probes

    def run(self, media_path=DEFAULT_MEDIA_PATH):
        return self.main_module._run_selftest(media_path)

    @property
    def report_path(self):
        return self.log_directory / REPORT_FILENAME

    def read_report(self):
        return self.report_path.read_text(encoding="utf-8")

    def report_lines(self):
        return self.read_report().splitlines()

    def line_starting_with(self, prefix):
        matches = [line for line in self.report_lines() if line.startswith(prefix)]
        assert len(matches) == 1, f"expected exactly one {prefix!r} line, found {matches}"
        return matches[0]

    def set_models(self, *outcomes):
        self.factory.outcomes = list(outcomes)

    def add_model(self, outcome):
        self.factory.outcomes.append(outcome)

    def enable_gpu(self, device_count=1):
        self.library_probe.available = True
        self.device_probe.device_count = device_count

    def use_real_media_probes(self):
        for name, function in self.real_media_probes.items():
            self.monkeypatch.setattr(engine, name, function)


@pytest.fixture
def selftest(tmp_path, monkeypatch, main_module):
    log_directory = tmp_path / "logs"
    monkeypatch.setattr(constants, "LOG_DIR", str(log_directory))

    real_media_probes = {
        "has_audio_stream": engine.has_audio_stream,
        "probe_file": engine.probe_file,
    }
    monkeypatch.setattr(engine, "has_audio_stream", return_value(True))
    monkeypatch.setattr(engine, "probe_file", return_value(PROBE_RESULT))

    cpu_model = FakeWhisperModel([
        ([FakeSegment("  Hello there "), FakeSegment("world.  ")],
         make_info(duration=12.5, language="en")),
    ])
    factory = RecordingModelFactory([cpu_model])
    monkeypatch.setattr(faster_whisper, "WhisperModel", factory)

    device_probe = CountingCudaDeviceProbe(device_count=0)
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", device_probe)

    library_probe = CountingLibraryProbe(available=False)
    monkeypatch.setattr(gpu_pack, "libraries_available", library_probe)

    return SelftestHarness(main_module, monkeypatch, log_directory, factory, cpu_model,
                           device_probe, library_probe, real_media_probes)


class TestSelftestReportFile:
    def test_returns_the_path_of_the_report_it_wrote(self, selftest):
        returned_path = selftest.run()

        assert returned_path == str(selftest.report_path)

    def test_writes_the_report_into_the_log_directory(self, selftest):
        selftest.run()

        assert selftest.report_path.is_file()

    def test_creates_the_log_directory_tree_when_it_is_missing(self, selftest, tmp_path):
        nested = tmp_path / "logs" / "diagnostics" / "runs"
        selftest.monkeypatch.setattr(constants, "LOG_DIR", str(nested))
        selftest.log_directory = nested

        selftest.run()

        assert selftest.report_path.is_file()

    def test_overwrites_a_report_left_by_a_previous_run(self, selftest):
        selftest.log_directory.mkdir(parents=True)
        selftest.report_path.write_text("stale content from an older run\n", encoding="utf-8")

        selftest.run()

        assert "stale content" not in selftest.read_report()

    def test_report_ends_with_a_trailing_newline(self, selftest):
        selftest.run()

        assert selftest.read_report().endswith("\n")

    def test_writes_the_report_as_utf8(self, selftest):
        selftest.set_models(FakeWhisperModel([
            ([FakeSegment("Hola señor")], make_info(duration=1.0, language="es")),
        ]))

        selftest.run(r"C:\media\señor niño.mp4")

        assert "señor".encode() in selftest.report_path.read_bytes()
        assert "Hola señor" in selftest.read_report()

    def test_the_report_is_a_plain_text_line_sequence(self, selftest):
        selftest.run()

        assert all(line == line.rstrip("\r") for line in selftest.report_lines())


class TestSelftestReportContent:
    def test_first_line_names_the_application_self_test(self, selftest):
        selftest.run()

        assert selftest.report_lines()[0] == "Sotvox self-test"

    def test_records_the_media_path_under_test(self, selftest):
        selftest.run(r"D:\clips\interview.wav")

        assert selftest.line_starting_with("Media:") == r"Media: D:\clips\interview.wav"

    def test_records_an_empty_media_path_without_failing(self, selftest):
        selftest.run("")

        assert selftest.line_starting_with("Media:") == "Media: "
        assert selftest.report_lines()[-1] == "RESULT: PASS"

    def test_leaves_a_blank_line_between_the_header_and_the_findings(self, selftest):
        selftest.run()

        assert selftest.report_lines()[2] == ""

    def test_records_the_pyav_version(self, selftest):
        selftest.run()

        assert selftest.line_starting_with("PyAV:") == f"PyAV: {av.__version__}"

    def test_records_the_faster_whisper_version(self, selftest):
        selftest.run()

        expected = f"faster-whisper: {faster_whisper.__version__}"
        assert selftest.line_starting_with("faster-whisper:") == expected

    def test_records_the_ctranslate2_version(self, selftest):
        selftest.run()

        assert selftest.line_starting_with("CTranslate2:") == f"CTranslate2: {ctranslate2.__version__}"

    @pytest.mark.parametrize("device_count", [0, 1, 4])
    def test_records_the_cuda_device_count(self, selftest, device_count):
        selftest.device_probe.device_count = device_count
        if device_count:
            selftest.library_probe.available = False

        selftest.run()

        assert selftest.line_starting_with("CUDA devices:") == f"CUDA devices: {device_count}"

    @pytest.mark.parametrize("has_audio", [True, False])
    def test_records_whether_the_media_has_an_audio_stream(self, selftest, has_audio):
        selftest.monkeypatch.setattr(engine, "has_audio_stream", return_value(has_audio))

        selftest.run()

        assert selftest.line_starting_with("has_audio_stream:") == f"has_audio_stream: {has_audio}"

    def test_records_the_container_format_section_of_the_probe(self, selftest):
        selftest.run()

        assert selftest.line_starting_with("probe:") == f"probe: {PROBE_RESULT['format']}"

    def test_records_a_none_probe_when_the_container_cannot_be_read(self, selftest):
        selftest.monkeypatch.setattr(engine, "probe_file", return_value(None))

        selftest.run()

        assert selftest.line_starting_with("probe:") == "probe: None"

    def test_records_the_detected_language(self, selftest):
        selftest.set_models(FakeWhisperModel([
            ([FakeSegment("hola")], make_info(duration=3.0, language="es")),
        ]))

        selftest.run()

        assert selftest.line_starting_with("detected language:") == "detected language: es"

    @pytest.mark.parametrize(
        ("duration", "expected"),
        [(0.0, "duration: 0.00s"), (3.14159, "duration: 3.14s"), (12.5, "duration: 12.50s")],
    )
    def test_formats_the_duration_with_two_decimals(self, selftest, duration, expected):
        selftest.set_models(FakeWhisperModel([
            ([FakeSegment("hello")], make_info(duration=duration)),
        ]))

        selftest.run()

        assert selftest.line_starting_with("duration:") == expected

    def test_joins_the_stripped_segment_texts_into_the_transcript_line(self, selftest):
        selftest.run()

        assert selftest.line_starting_with("transcript:") == "transcript: Hello there world."

    def test_transcript_line_is_empty_when_the_model_returns_no_segments(self, selftest):
        selftest.set_models(FakeWhisperModel([([], make_info(duration=1.0))]))

        selftest.run()

        assert selftest.line_starting_with("transcript:") == "transcript: "

    def test_ends_with_an_overall_pass_verdict(self, selftest):
        selftest.run()

        assert selftest.report_lines()[-1] == "RESULT: PASS"

    def test_reports_the_cpu_stage_as_passing(self, selftest):
        selftest.run()

        assert "CPU RESULT: PASS" in selftest.report_lines()


class TestCpuStage:
    def test_loads_the_tiny_model_on_the_cpu_with_int8_precision(self, selftest):
        selftest.run()

        assert selftest.factory.constructor_calls == [
            {"model_name": "tiny", "device": "cpu", "compute_type": "int8"}]

    def test_transcribes_the_media_path_with_vad_and_a_beam_size_of_five(self, selftest):
        selftest.run(r"E:\clips\meeting.mkv")

        assert selftest.cpu_model.last_call == {
            "audio": r"E:\clips\meeting.mkv", "beam_size": 5, "vad_filter": True}

    def test_consumes_every_segment_the_model_yields(self, selftest):
        selftest.run()

        assert selftest.cpu_model.iterators[0].consumed == 2

    def test_transcribes_exactly_once_on_the_cpu(self, selftest):
        selftest.run()

        assert len(selftest.cpu_model.calls) == 1


class TestGpuStage:
    @pytest.mark.parametrize(
        ("libraries_available", "device_count"),
        [(False, 0), (False, 1), (True, 0)],
        ids=["no-libraries-no-device", "no-libraries-with-device", "libraries-without-device"],
    )
    def test_skips_the_gpu_stage_when_libraries_or_devices_are_missing(
            self, selftest, libraries_available, device_count):
        selftest.library_probe.available = libraries_available
        selftest.device_probe.device_count = device_count

        selftest.run()

        assert "GPU RESULT: SKIPPED (libraries not installed)" in selftest.report_lines()

    def test_a_skipped_gpu_stage_still_yields_an_overall_pass(self, selftest):
        selftest.run()

        assert selftest.report_lines()[-1] == "RESULT: PASS"

    def test_does_not_build_a_gpu_model_when_the_stage_is_skipped(self, selftest):
        selftest.run()

        assert selftest.factory.devices == ["cpu"]

    def test_does_not_query_the_device_count_again_when_the_libraries_are_missing(self, selftest):
        selftest.device_probe.device_count = 2

        selftest.run()

        assert selftest.device_probe.call_count == 1

    def test_queries_the_device_count_a_second_time_when_the_libraries_are_present(self, selftest):
        selftest.enable_gpu()
        selftest.add_model(FakeWhisperModel([([FakeSegment("gpu")], make_info())]))

        selftest.run()

        assert selftest.device_probe.call_count == 2

    def test_asks_the_gpu_pack_whether_the_libraries_are_available(self, selftest):
        selftest.run()

        assert selftest.library_probe.call_count == 1

    def test_loads_the_tiny_model_on_cuda_with_float16_precision(self, selftest):
        selftest.enable_gpu()
        selftest.add_model(FakeWhisperModel([([FakeSegment("gpu")], make_info())]))

        selftest.run()

        assert selftest.factory.constructor_calls[1] == {
            "model_name": "tiny", "device": "cuda", "compute_type": "float16"}

    def test_records_the_gpu_transcript_from_the_gpu_model(self, selftest):
        gpu_model = FakeWhisperModel([
            ([FakeSegment("  from the "), FakeSegment("graphics card ")], make_info()),
        ])
        selftest.enable_gpu()
        selftest.add_model(gpu_model)

        selftest.run()

        assert selftest.line_starting_with("gpu transcript:") == "gpu transcript: from the graphics card"

    def test_reports_the_gpu_stage_as_passing_when_both_prerequisites_hold(self, selftest):
        selftest.enable_gpu()
        selftest.add_model(FakeWhisperModel([([FakeSegment("gpu")], make_info())]))

        selftest.run()

        lines = selftest.report_lines()
        assert "GPU RESULT: PASS" in lines
        assert "GPU RESULT: SKIPPED (libraries not installed)" not in lines
        assert lines[-1] == "RESULT: PASS"

    def test_transcribes_the_same_media_on_the_gpu_with_the_same_options(self, selftest):
        gpu_model = FakeWhisperModel([([FakeSegment("gpu")], make_info())])
        selftest.enable_gpu()
        selftest.add_model(gpu_model)

        selftest.run(r"F:\clips\lecture.mp3")

        assert gpu_model.last_call == {
            "audio": r"F:\clips\lecture.mp3", "beam_size": 5, "vad_filter": True}


def break_audio_stream_probe(harness):
    harness.monkeypatch.setattr(
        engine, "has_audio_stream", raise_error(RuntimeError(FAILURE_MESSAGE)))


def break_container_probe(harness):
    harness.monkeypatch.setattr(engine, "probe_file", raise_error(RuntimeError(FAILURE_MESSAGE)))


def break_device_count_query(harness):
    harness.device_probe.error = RuntimeError(FAILURE_MESSAGE)


def break_cpu_model_construction(harness):
    harness.set_models(RuntimeError(FAILURE_MESSAGE))


def break_cpu_transcription(harness):
    harness.set_models(FakeWhisperModel([RuntimeError(FAILURE_MESSAGE)]))


def break_library_probe(harness):
    harness.library_probe.error = RuntimeError(FAILURE_MESSAGE)


def break_gpu_model_construction(harness):
    harness.enable_gpu()
    harness.add_model(RuntimeError(FAILURE_MESSAGE))


def break_gpu_transcription(harness):
    harness.enable_gpu()
    harness.add_model(FakeWhisperModel([RuntimeError(FAILURE_MESSAGE)]))


FAILING_STEPS = [
    pytest.param(break_audio_stream_probe, id="audio-stream-probe"),
    pytest.param(break_container_probe, id="container-probe"),
    pytest.param(break_device_count_query, id="device-count-query"),
    pytest.param(break_cpu_model_construction, id="cpu-model-construction"),
    pytest.param(break_cpu_transcription, id="cpu-transcription"),
    pytest.param(break_library_probe, id="library-probe"),
    pytest.param(break_gpu_model_construction, id="gpu-model-construction"),
    pytest.param(break_gpu_transcription, id="gpu-transcription"),
]


class TestSelftestFailures:
    @pytest.mark.parametrize("break_step", FAILING_STEPS)
    def test_a_failing_step_does_not_propagate(self, selftest, break_step):
        break_step(selftest)

        assert selftest.run() == str(selftest.report_path)

    @pytest.mark.parametrize("break_step", FAILING_STEPS)
    def test_a_failing_step_ends_the_report_with_a_fail_verdict(self, selftest, break_step):
        break_step(selftest)

        selftest.run()

        assert selftest.report_lines()[-1] == "RESULT: FAIL"

    @pytest.mark.parametrize("break_step", FAILING_STEPS)
    def test_a_failing_step_records_a_traceback(self, selftest, break_step):
        break_step(selftest)

        selftest.run()

        report = selftest.read_report()
        assert TRACEBACK_HEADER in report
        assert "RuntimeError" in report
        assert FAILURE_MESSAGE in report

    @pytest.mark.parametrize("break_step", FAILING_STEPS)
    def test_a_failing_step_never_records_an_overall_pass(self, selftest, break_step):
        break_step(selftest)

        selftest.run()

        assert "RESULT: PASS" not in selftest.report_lines()

    @pytest.mark.parametrize(
        "break_step",
        [break_audio_stream_probe, break_container_probe, break_device_count_query,
         break_cpu_model_construction, break_cpu_transcription],
        ids=["audio-stream-probe", "container-probe", "device-count-query",
             "cpu-model-construction", "cpu-transcription"],
    )
    def test_a_failure_before_the_cpu_verdict_leaves_no_cpu_pass_line(self, selftest, break_step):
        break_step(selftest)

        selftest.run()

        assert "CPU RESULT: PASS" not in selftest.report_lines()

    @pytest.mark.parametrize(
        "break_step",
        [break_library_probe, break_gpu_model_construction, break_gpu_transcription],
        ids=["library-probe", "gpu-model-construction", "gpu-transcription"],
    )
    def test_a_failure_in_the_gpu_stage_keeps_the_cpu_findings(self, selftest, break_step):
        break_step(selftest)

        selftest.run()

        lines = selftest.report_lines()
        assert "CPU RESULT: PASS" in lines
        assert selftest.line_starting_with("transcript:") == "transcript: Hello there world."
        assert lines[-1] == "RESULT: FAIL"

    def test_a_failure_in_the_gpu_stage_records_no_gpu_verdict(self, selftest):
        break_gpu_model_construction(selftest)

        selftest.run()

        lines = selftest.report_lines()
        assert "GPU RESULT: PASS" not in lines
        assert "GPU RESULT: SKIPPED (libraries not installed)" not in lines

    def test_an_early_failure_still_records_the_header_lines(self, selftest):
        break_audio_stream_probe(selftest)

        selftest.run(r"G:\clips\broken.mp4")

        lines = selftest.report_lines()
        assert lines[0] == "Sotvox self-test"
        assert lines[1] == r"Media: G:\clips\broken.mp4"

    def test_an_early_failure_keeps_the_library_version_lines(self, selftest):
        break_audio_stream_probe(selftest)

        selftest.run()

        assert selftest.line_starting_with("PyAV:") == f"PyAV: {av.__version__}"

    def test_a_failure_while_reading_versions_keeps_only_the_header(self, selftest):
        break_device_count_query(selftest)

        selftest.run()

        lines = selftest.report_lines()
        assert not [line for line in lines if line.startswith("has_audio_stream:")]
        assert lines[-1] == "RESULT: FAIL"

    def test_the_report_is_written_even_though_the_run_failed(self, selftest):
        break_cpu_model_construction(selftest)

        selftest.run()

        assert selftest.report_path.is_file()

    def test_a_keyboard_interrupt_is_not_swallowed(self, selftest):
        selftest.monkeypatch.setattr(
            engine, "has_audio_stream", raise_error(KeyboardInterrupt()))

        with pytest.raises(KeyboardInterrupt):
            selftest.run()

    def test_no_report_survives_a_keyboard_interrupt(self, selftest):
        selftest.monkeypatch.setattr(
            engine, "has_audio_stream", raise_error(KeyboardInterrupt()))

        with pytest.raises(KeyboardInterrupt):
            selftest.run()

        assert not selftest.report_path.exists()


class TestSelftestAgainstRealMedia:
    @pytest.mark.integration
    def test_detects_the_audio_stream_of_a_real_wav_file(self, selftest, tmp_path):
        media_path = write_silent_wav(tmp_path / "silence.wav")
        selftest.use_real_media_probes()

        selftest.run(str(media_path))

        assert selftest.line_starting_with("has_audio_stream:") == "has_audio_stream: True"

    @pytest.mark.integration
    def test_probes_the_container_format_of_a_real_wav_file(self, selftest, tmp_path):
        media_path = write_silent_wav(tmp_path / "silence.wav")
        selftest.use_real_media_probes()

        selftest.run(str(media_path))

        assert "format_long_name" in selftest.line_starting_with("probe:")

    @pytest.mark.integration
    def test_reports_a_missing_file_as_having_no_audio_stream(self, selftest, tmp_path):
        selftest.use_real_media_probes()

        selftest.run(str(tmp_path / "absent.wav"))

        assert selftest.line_starting_with("has_audio_stream:") == "has_audio_stream: False"
        assert selftest.line_starting_with("probe:") == "probe: None"
        assert selftest.report_lines()[-1] == "RESULT: PASS"
