"""Behaviour of main._run_batch_transcribe, the headless automation entry point."""

import os
import re
import sys
from types import SimpleNamespace

import ctranslate2
import faster_whisper
import pytest
from doubles import FakeClock, make_info

import constants
import engine

TIMESTAMP_PATTERN = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")
DEFAULT_SEGMENT_TEXTS = ["hello there"]
DEFAULT_DURATION_SECONDS = 12.0
DEFAULT_LANGUAGE = "en"
CLOCK_STEP_SECONDS = 3.0


class LoadedWhisperModel:
    def __init__(self, model_name, device, compute_type):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type


class RecordingWhisperModelFactory:
    def __init__(self):
        self.constructions = []
        self.failures_by_device = {}

    def __call__(self, model_name, device=None, compute_type=None):
        self.constructions.append(
            {"model_name": model_name, "device": device, "compute_type": compute_type})
        failure = self.failures_by_device.get(device)
        if failure is not None:
            raise failure
        return LoadedWhisperModel(model_name, device, compute_type)

    @property
    def devices(self):
        return [construction["device"] for construction in self.constructions]

    @property
    def compute_types(self):
        return [construction["compute_type"] for construction in self.constructions]


class RecordingTranscriber:
    def __init__(self):
        self.calls = []
        self.failures = {}
        self.segment_texts = {}
        self.infos = {}
        self.statuses = {}

    def __call__(self, model, audio_path, language_code):
        name = os.path.basename(audio_path)
        self.calls.append({
            "model": model,
            "audio_path": audio_path,
            "language_code": language_code,
            "name": name,
        })
        failure = self.failures.get(name)
        if failure is not None:
            raise failure
        text_parts = list(self.segment_texts.get(name, DEFAULT_SEGMENT_TEXTS))
        info = self.infos.get(
            name, make_info(duration=DEFAULT_DURATION_SECONDS, language=DEFAULT_LANGUAGE))
        return text_parts, info, self.statuses.get(name, "ok")

    @property
    def transcribed_names(self):
        return [call["name"] for call in self.calls]

    @property
    def language_codes(self):
        return [call["language_code"] for call in self.calls]


class RecordingTranscriptSaver:
    def __init__(self):
        self.calls = []
        self.failures = {}

    def __call__(self, source_path, full_text, output_dir):
        name = os.path.basename(source_path)
        self.calls.append({
            "source_path": source_path,
            "full_text": full_text,
            "output_dir": output_dir,
            "name": name,
        })
        failure = self.failures.get(name)
        if failure is not None:
            raise failure
        target_name = os.path.splitext(name)[0] + ".txt"
        with open(os.path.join(output_dir, target_name), "w", encoding="utf-8") as handle:
            handle.write(full_text)
        return target_name

    @property
    def saved_names(self):
        return [call["name"] for call in self.calls]


class CudaDeviceProbe:
    def __init__(self, device_count=0):
        self.device_count = device_count
        self.error = None
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.device_count


class BatchHarness:
    def __init__(self, main_module, sandboxed_dirs, tmp_path, monkeypatch):
        self.main_module = main_module
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.log_dir = sandboxed_dirs["log"]
        self.default_output_dir = sandboxed_dirs["output"]
        self.input_dir = tmp_path / "incoming"
        self.input_dir.mkdir()
        self.transcriber = RecordingTranscriber()
        self.saver = RecordingTranscriptSaver()
        self.model_factory = RecordingWhisperModelFactory()
        self.cuda_probe = CudaDeviceProbe()
        self.clock = FakeClock(start=1000.0, step=CLOCK_STEP_SECONDS)

        monkeypatch.setattr(engine, "transcribe_audio", self.transcriber)
        monkeypatch.setattr(engine, "save_transcript", self.saver)
        monkeypatch.setattr(faster_whisper, "WhisperModel", self.model_factory)
        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", self.cuda_probe)
        monkeypatch.setattr(main_module, "time", SimpleNamespace(time=self.clock))

    def use_log_directory(self, directory):
        self.monkeypatch.setattr(constants, "LOG_DIR", str(directory))
        self.log_dir = directory

    def add_media(self, name, content=b""):
        path = self.input_dir / name
        path.write_bytes(content)
        return path

    def add_subdirectory(self, name):
        path = self.input_dir / name
        path.mkdir()
        return path

    def add_existing_transcript(self, name, text="already done", directory=None):
        target_dir = self.default_output_dir if directory is None else directory
        path = target_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def make_file_outside_the_input_folder(self, name):
        path = self.tmp_path / name
        path.write_text("not a folder", encoding="utf-8")
        return path

    def run(self, *extra_arguments):
        return self.run_raw("--transcribe", str(self.input_dir), *extra_arguments)

    def run_raw(self, *arguments):
        self.monkeypatch.setattr(sys, "argv", ["sotvox", *arguments])
        return self.main_module._run_batch_transcribe()

    @property
    def report_path(self):
        return self.log_dir / "batch.txt"

    @property
    def report_text(self):
        return self.report_path.read_text(encoding="utf-8")

    @property
    def report_lines(self):
        return self.report_text.splitlines()

    @property
    def report_messages(self):
        return [
            TIMESTAMP_PATTERN.sub("", line)
            for line in self.report_lines
            if TIMESTAMP_PATTERN.match(line)
        ]

    def messages_containing(self, fragment):
        return [message for message in self.report_messages if fragment in message]

    def transcripts_in(self, directory=None):
        target_dir = self.default_output_dir if directory is None else directory
        if not target_dir.is_dir():
            return []
        return sorted(entry.name for entry in target_dir.iterdir())

    def read_transcript(self, name, directory=None):
        target_dir = self.default_output_dir if directory is None else directory
        return (target_dir / name).read_text(encoding="utf-8")


@pytest.fixture
def harness(main_module, sandboxed_dirs, tmp_path, monkeypatch):
    return BatchHarness(main_module, sandboxed_dirs, tmp_path, monkeypatch)


def arguments_without_the_transcribe_flag(harness):
    return ()


def arguments_with_a_valueless_transcribe_flag(harness):
    return ("--transcribe",)


def arguments_with_a_missing_folder(harness):
    return ("--transcribe", str(harness.tmp_path / "no-such-folder"))


def arguments_with_a_file_instead_of_a_folder(harness):
    return ("--transcribe", str(harness.make_file_outside_the_input_folder("notes.txt")))


def arguments_with_an_empty_folder_name(harness):
    return ("--transcribe", "")


INVALID_INPUT_BUILDERS = [
    arguments_without_the_transcribe_flag,
    arguments_with_a_valueless_transcribe_flag,
    arguments_with_a_missing_folder,
    arguments_with_a_file_instead_of_a_folder,
    arguments_with_an_empty_folder_name,
]
INVALID_INPUT_IDS = [
    "flag absent",
    "flag without a value",
    "folder does not exist",
    "path is a file",
    "folder name is empty",
]


def scenario_invalid_input_folder(harness):
    return harness.run_raw("--transcribe", str(harness.tmp_path / "no-such-folder"))


def scenario_nothing_pending(harness):
    return harness.run()


def scenario_successful_batch(harness):
    harness.add_media("clip.mp3")
    return harness.run()


def scenario_batch_with_a_failing_file(harness):
    harness.add_media("clip.mp3")
    harness.transcriber.failures["clip.mp3"] = RuntimeError("decode error")
    return harness.run()


def scenario_fatal_model_failure(harness):
    harness.add_media("clip.mp3")
    harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")
    return harness.run()


RETURN_PATH_SCENARIOS = [
    scenario_invalid_input_folder,
    scenario_nothing_pending,
    scenario_successful_batch,
    scenario_batch_with_a_failing_file,
    scenario_fatal_model_failure,
]
RETURN_PATH_IDS = [
    "invalid input folder",
    "nothing pending",
    "successful batch",
    "batch with a failing file",
    "fatal model failure",
]


class TestInputFolderValidation:
    @pytest.mark.parametrize("build_arguments", INVALID_INPUT_BUILDERS, ids=INVALID_INPUT_IDS)
    def test_returns_bad_usage_for_an_unusable_input_folder(self, harness, main_module, build_arguments):
        assert harness.run_raw(*build_arguments(harness)) == main_module.EXIT_BAD_USAGE

    @pytest.mark.parametrize("build_arguments", INVALID_INPUT_BUILDERS, ids=INVALID_INPUT_IDS)
    def test_logs_an_input_folder_error_for_an_unusable_input_folder(self, harness, build_arguments):
        harness.run_raw(*build_arguments(harness))

        assert harness.messages_containing("ERROR: input folder not found")

    @pytest.mark.parametrize("build_arguments", INVALID_INPUT_BUILDERS, ids=INVALID_INPUT_IDS)
    def test_loads_no_model_for_an_unusable_input_folder(self, harness, build_arguments):
        harness.run_raw(*build_arguments(harness))

        assert harness.model_factory.constructions == []

    @pytest.mark.parametrize("build_arguments", INVALID_INPUT_BUILDERS, ids=INVALID_INPUT_IDS)
    def test_transcribes_nothing_for_an_unusable_input_folder(self, harness, build_arguments):
        harness.run_raw(*build_arguments(harness))

        assert harness.transcriber.calls == []

    def test_error_line_names_the_rejected_folder(self, harness):
        missing_folder = harness.tmp_path / "no-such-folder"

        harness.run_raw("--transcribe", str(missing_folder))

        assert f"ERROR: input folder not found: {missing_folder}" in harness.report_messages

    def test_error_line_reports_none_when_the_flag_is_absent(self, harness):
        harness.run_raw()

        assert "ERROR: input folder not found: None" in harness.report_messages

    def test_start_line_is_logged_before_the_folder_is_rejected(self, harness):
        harness.run_raw("--transcribe", str(harness.tmp_path / "no-such-folder"))

        assert harness.report_messages[0].startswith("Batch started: input=")

    def test_output_directory_is_not_created_when_the_input_folder_is_rejected(self, harness):
        planned_output = harness.tmp_path / "never-created"

        harness.run_raw("--transcribe", str(harness.tmp_path / "gone"), "--output", str(planned_output))

        assert not planned_output.exists()

    def test_a_folder_that_exists_is_accepted(self, harness, main_module):
        harness.add_media("clip.mp3")

        assert harness.run() == main_module.EXIT_OK


class TestNothingToTranscribe:
    def test_returns_ok_when_the_input_folder_is_empty(self, harness, main_module):
        assert harness.run() == main_module.EXIT_OK

    def test_logs_nothing_new_when_the_input_folder_is_empty(self, harness):
        harness.run()

        assert "Nothing new to transcribe" in harness.report_messages

    def test_logs_nothing_new_when_only_unsupported_files_are_present(self, harness):
        harness.add_media("notes.txt")
        harness.add_media("manual.pdf")
        harness.add_media("README")

        harness.run()

        assert "Nothing new to transcribe" in harness.report_messages

    def test_logs_nothing_new_when_every_file_already_has_a_transcript(self, harness):
        harness.add_media("first.mp3")
        harness.add_media("second.mp4")
        harness.add_existing_transcript("first.txt")
        harness.add_existing_transcript("second.txt")

        harness.run()

        assert "Nothing new to transcribe" in harness.report_messages

    def test_returns_ok_when_every_file_already_has_a_transcript(self, harness, main_module):
        harness.add_media("first.mp3")
        harness.add_existing_transcript("first.txt")

        assert harness.run() == main_module.EXIT_OK

    def test_loads_no_model_when_nothing_is_pending(self, harness):
        harness.add_media("first.mp3")
        harness.add_existing_transcript("first.txt")

        harness.run()

        assert harness.model_factory.constructions == []

    def test_transcribes_nothing_when_nothing_is_pending(self, harness):
        harness.run()

        assert harness.transcriber.calls == []

    def test_no_pending_count_line_is_logged_when_nothing_is_pending(self, harness):
        harness.run()

        assert harness.messages_containing("new file(s) to transcribe") == []

    def test_no_final_tally_is_logged_when_nothing_is_pending(self, harness):
        harness.run()

        assert harness.messages_containing("Batch finished") == []

    def test_output_directory_is_still_created_when_nothing_is_pending(self, harness):
        planned_output = harness.tmp_path / "created-anyway"

        harness.run("--output", str(planned_output))

        assert planned_output.is_dir()

    def test_existing_transcripts_are_left_untouched_when_nothing_is_pending(self, harness):
        harness.add_media("first.mp3")
        harness.add_existing_transcript("first.txt", "the original text")

        harness.run()

        assert harness.read_transcript("first.txt") == "the original text"


class TestPendingFileSelection:
    def test_only_files_without_a_transcript_are_transcribed(self, harness):
        for name in ("alpha.mp3", "beta.mp4", "gamma.wav", "delta.flac"):
            harness.add_media(name)
        harness.add_existing_transcript("beta.txt")
        harness.add_existing_transcript("delta.txt")

        harness.run()

        assert harness.transcriber.transcribed_names == ["alpha.mp3", "gamma.wav"]

    def test_pending_count_line_reports_the_number_selected(self, harness):
        harness.add_media("alpha.mp3")
        harness.add_media("beta.mp3")
        harness.add_existing_transcript("beta.txt")

        harness.run()

        assert "1 new file(s) to transcribe" in harness.report_messages

    def test_a_transcript_named_after_the_full_filename_does_not_count(self, harness):
        harness.add_media("clip.mp3")
        harness.add_existing_transcript("clip.mp3.txt")

        harness.run()

        assert harness.transcriber.transcribed_names == ["clip.mp3"]

    def test_a_transcript_sitting_in_the_input_folder_does_not_count(self, harness):
        harness.add_media("clip.mp3")
        harness.add_existing_transcript("clip.txt", directory=harness.input_dir)

        harness.run()

        assert harness.transcriber.transcribed_names == ["clip.mp3"]

    def test_a_transcript_in_a_different_output_folder_does_not_count(self, harness):
        harness.add_media("clip.mp3")
        harness.add_existing_transcript("clip.txt")
        other_output = harness.tmp_path / "elsewhere"
        other_output.mkdir()

        harness.run("--output", str(other_output))

        assert harness.transcriber.transcribed_names == ["clip.mp3"]

    def test_a_directory_named_like_the_transcript_blocks_the_file(self, harness):
        harness.add_media("clip.mp3")
        (harness.default_output_dir / "clip.txt").mkdir()

        harness.run()

        assert harness.transcriber.calls == []

    def test_directories_named_like_media_files_are_ignored(self, harness):
        harness.add_subdirectory("clip.mp4")
        harness.add_media("real.mp3")

        harness.run()

        assert harness.transcriber.transcribed_names == ["real.mp3"]

    def test_files_nested_in_subdirectories_are_not_discovered(self, harness):
        nested = harness.add_subdirectory("archive")
        (nested / "buried.mp3").write_bytes(b"")

        harness.run()

        assert harness.transcriber.calls == []

    @pytest.mark.parametrize("name", ["notes.txt", "manual.pdf", "data.json", "README", "archive.zip"])
    def test_unsupported_extensions_are_ignored(self, harness, name):
        harness.add_media(name)
        harness.add_media("real.mp3")

        harness.run()

        assert harness.transcriber.transcribed_names == ["real.mp3"]

    def test_a_dotfile_named_like_an_extension_is_ignored(self, harness):
        harness.add_media(".mp3")

        harness.run()

        assert harness.transcriber.calls == []

    @pytest.mark.parametrize("name", ["CLIP.MP4", "Talk.Mp3", "Song.WAV"])
    def test_extension_matching_is_case_insensitive(self, harness, name):
        harness.add_media(name)

        harness.run()

        assert harness.transcriber.transcribed_names == [name]

    def test_uppercase_extension_still_maps_to_a_lowercase_transcript_name(self, harness):
        harness.add_media("CLIP.MP4")

        harness.run()

        assert harness.transcripts_in() == ["CLIP.txt"]

    @pytest.mark.parametrize("extension", sorted(constants.SUPPORTED_EXTENSIONS))
    def test_every_supported_extension_is_picked_up(self, harness, extension):
        harness.add_media(f"sample{extension}")

        harness.run()

        assert harness.transcriber.transcribed_names == [f"sample{extension}"]

    def test_files_sharing_a_stem_are_both_queued(self, harness):
        harness.add_media("talk.mp3")
        harness.add_media("talk.wav")

        harness.run()

        assert harness.transcriber.transcribed_names == ["talk.mp3", "talk.wav"]

    def test_transcribe_receives_the_full_source_path(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.transcriber.calls[0]["audio_path"] == str(harness.input_dir / "clip.mp3")


class TestProcessingOrder:
    def test_files_are_transcribed_in_sorted_order(self, harness):
        for name in ("charlie.mp3", "alpha.mp3", "bravo.mp3"):
            harness.add_media(name)

        harness.run()

        assert harness.transcriber.transcribed_names == ["alpha.mp3", "bravo.mp3", "charlie.mp3"]

    def test_sorting_is_by_raw_name_so_uppercase_comes_first(self, harness):
        harness.add_media("apple.mp3")
        harness.add_media("Banana.mp3")

        harness.run()

        assert harness.transcriber.transcribed_names == ["Banana.mp3", "apple.mp3"]

    def test_sorting_is_lexicographic_not_numeric(self, harness):
        harness.add_media("clip2.mp3")
        harness.add_media("clip10.mp3")

        harness.run()

        assert harness.transcriber.transcribed_names == ["clip10.mp3", "clip2.mp3"]

    def test_per_file_log_lines_follow_the_processing_order(self, harness):
        for name in ("charlie.mp3", "alpha.mp3", "bravo.mp3"):
            harness.add_media(name)

        harness.run()

        processed = [message.strip().split(":")[0] for message in harness.messages_containing("audio in")]
        assert processed == ["alpha.mp3", "bravo.mp3", "charlie.mp3"]


class TestOutputDirectory:
    def test_defaults_to_the_configured_output_directory(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.saver.calls[0]["output_dir"] == str(harness.default_output_dir)

    def test_output_flag_overrides_the_default(self, harness):
        harness.add_media("clip.mp3")
        chosen_output = harness.tmp_path / "chosen"
        chosen_output.mkdir()

        harness.run("--output", str(chosen_output))

        assert harness.transcripts_in(chosen_output) == ["clip.txt"]

    def test_nothing_is_written_to_the_default_directory_when_output_is_overridden(self, harness):
        harness.add_media("clip.mp3")
        chosen_output = harness.tmp_path / "chosen"

        harness.run("--output", str(chosen_output))

        assert harness.transcripts_in() == []

    def test_output_directory_is_created_when_missing(self, harness):
        harness.add_media("clip.mp3")
        chosen_output = harness.tmp_path / "not-there-yet"

        harness.run("--output", str(chosen_output))

        assert chosen_output.is_dir()

    def test_nested_output_directory_is_created(self, harness):
        harness.add_media("clip.mp3")
        chosen_output = harness.tmp_path / "deep" / "deeper" / "deepest"

        harness.run("--output", str(chosen_output))

        assert harness.transcripts_in(chosen_output) == ["clip.txt"]

    def test_existing_output_directory_contents_are_preserved(self, harness):
        harness.add_media("clip.mp3")
        harness.add_existing_transcript("unrelated.txt", "keep me")

        harness.run()

        assert harness.read_transcript("unrelated.txt") == "keep me"

    def test_output_can_be_the_input_folder(self, harness):
        harness.add_media("clip.mp3")

        harness.run("--output", str(harness.input_dir))

        assert (harness.input_dir / "clip.txt").read_text(encoding="utf-8") == "hello there"

    def test_start_line_records_the_resolved_output_directory(self, harness):
        chosen_output = harness.tmp_path / "chosen"

        harness.run("--output", str(chosen_output))

        assert f"output={chosen_output} " in harness.report_messages[0]


class TestModelSelection:
    def test_defaults_to_large_v3_turbo(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.model_factory.constructions[0]["model_name"] == "large-v3-turbo"

    @pytest.mark.parametrize("model_name", ["tiny", "small", "medium", "large-v3"])
    def test_model_flag_overrides_the_default(self, harness, model_name):
        harness.add_media("clip.mp3")

        harness.run("--model", model_name)

        assert harness.model_factory.constructions[0]["model_name"] == model_name

    def test_start_line_records_the_resolved_model(self, harness):
        harness.run("--model", "tiny")

        assert harness.report_messages[0].endswith("model=tiny")

    def test_model_is_constructed_once_for_the_whole_batch(self, harness):
        for name in ("one.mp3", "two.mp3", "three.mp3"):
            harness.add_media(name)

        harness.run()

        assert len(harness.model_factory.constructions) == 1

    def test_the_same_model_instance_is_passed_to_every_transcription(self, harness):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")

        harness.run()

        models = {id(call["model"]) for call in harness.transcriber.calls}
        assert len(models) == 1

    def test_the_transcriber_receives_the_constructed_model(self, harness):
        harness.add_media("clip.mp3")

        harness.run("--model", "small")

        assert harness.transcriber.calls[0]["model"].model_name == "small"

    def test_loading_line_names_the_model_and_the_device(self, harness):
        harness.add_media("clip.mp3")

        harness.run("--model", "medium", "--device", "CPU")

        assert "Loading model medium on cpu" in harness.report_messages


class TestLanguageResolution:
    def test_defaults_to_auto_detect_which_sends_no_language_code(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.transcriber.language_codes == [None]

    def test_explicit_auto_detect_sends_no_language_code(self, harness):
        harness.add_media("clip.mp3")

        harness.run("--language", "Auto-detect")

        assert harness.transcriber.language_codes == [None]

    @pytest.mark.parametrize(
        ("language_name", "language_code"),
        [
            ("Spanish", "es"),
            ("English", "en"),
            ("Portuguese", "pt"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Japanese", "ja"),
            ("Chinese", "zh"),
            ("Korean", "ko"),
        ],
    )
    def test_a_known_language_name_reaches_the_transcriber_as_its_code(
            self, harness, language_name, language_code):
        harness.add_media("clip.mp3")

        harness.run("--language", language_name)

        assert harness.transcriber.language_codes == [language_code]

    @pytest.mark.parametrize("language_name", ["Klingon", "spanish", "ES", "", "auto-detect"])
    def test_an_unknown_language_name_resolves_to_no_language_code(self, harness, language_name):
        harness.add_media("clip.mp3")

        harness.run("--language", language_name)

        assert harness.transcriber.language_codes == [None]

    def test_the_same_language_code_is_used_for_every_file(self, harness):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")

        harness.run("--language", "Spanish")

        assert harness.transcriber.language_codes == ["es", "es"]

    def test_the_language_flag_does_not_change_the_reported_detected_language(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.infos["clip.mp3"] = make_info(duration=8.0, language="pt")

        harness.run("--language", "Spanish")

        assert "  clip.mp3: pt, 8s audio in 3s -> clip.txt" in harness.report_messages


class TestDeviceSelection:
    def test_cpu_is_forced_by_the_device_flag(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 4

        harness.run("--device", "CPU")

        assert harness.model_factory.constructions[0]["device"] == "cpu"
        assert harness.model_factory.constructions[0]["compute_type"] == "int8"

    def test_forcing_cpu_never_consults_ctranslate2(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 4

        harness.run("--device", "CPU")

        assert harness.cuda_probe.call_count == 0

    def test_auto_uses_cuda_when_a_device_is_reported(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1

        harness.run("--device", "Auto")

        assert harness.model_factory.constructions[0]["device"] == "cuda"
        assert harness.model_factory.constructions[0]["compute_type"] == "float16"

    def test_auto_is_the_default_device_choice(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 2

        harness.run()

        assert harness.model_factory.constructions[0]["device"] == "cuda"

    def test_auto_consults_ctranslate2_exactly_once(self, harness):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")
        harness.cuda_probe.device_count = 1

        harness.run()

        assert harness.cuda_probe.call_count == 1

    @pytest.mark.parametrize("device_count", [0, -1])
    def test_auto_falls_back_to_cpu_when_no_cuda_device_is_reported(self, harness, device_count):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = device_count

        harness.run()

        assert harness.model_factory.constructions[0]["device"] == "cpu"
        assert harness.model_factory.constructions[0]["compute_type"] == "int8"

    def test_auto_falls_back_to_cpu_when_ctranslate2_cannot_be_imported(self, harness, monkeypatch):
        harness.add_media("clip.mp3")
        monkeypatch.setitem(sys.modules, "ctranslate2", None)

        harness.run()

        assert harness.model_factory.constructions[0]["device"] == "cpu"

    def test_a_ctranslate2_import_failure_is_not_fatal(self, harness, main_module, monkeypatch):
        harness.add_media("clip.mp3")
        monkeypatch.setitem(sys.modules, "ctranslate2", None)

        assert harness.run() == main_module.EXIT_OK

    def test_auto_falls_back_to_cpu_when_the_device_query_raises(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 8
        harness.cuda_probe.error = RuntimeError("cuda driver missing")

        harness.run()

        assert harness.model_factory.constructions[0]["device"] == "cpu"

    def test_a_device_query_failure_is_not_logged_as_an_error(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.error = RuntimeError("cuda driver missing")

        harness.run()

        assert harness.messages_containing("FATAL") == []

    @pytest.mark.parametrize("device_choice", ["Auto", "GPU", "cpu", "cuda", ""])
    def test_any_choice_other_than_the_exact_word_cpu_allows_cuda(self, harness, device_choice):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1

        harness.run("--device", device_choice)

        assert harness.model_factory.constructions[0]["device"] == "cuda"


class TestCudaFallback:
    def test_a_cuda_load_failure_constructs_a_second_model_on_cpu(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")

        harness.run()

        assert harness.model_factory.devices == ["cuda", "cpu"]
        assert harness.model_factory.compute_types == ["float16", "int8"]

    def test_a_cuda_load_failure_is_logged(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")

        harness.run()

        assert "CUDA failed, falling back to CPU" in harness.report_messages

    def test_a_cuda_load_failure_still_returns_ok(self, harness, main_module):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")

        assert harness.run() == main_module.EXIT_OK

    def test_a_cuda_load_failure_still_transcribes_every_file(self, harness):
        for name in ("one.mp3", "two.mp3", "three.mp3"):
            harness.add_media(name)
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")

        harness.run()

        assert harness.transcripts_in() == ["one.txt", "three.txt", "two.txt"]

    def test_the_cpu_replacement_model_is_the_one_used(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")

        harness.run()

        assert harness.transcriber.calls[0]["model"].device == "cpu"

    def test_the_cuda_failure_is_not_reported_as_fatal(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")

        harness.run()

        assert harness.messages_containing("FATAL") == []

    def test_a_cpu_load_failure_is_fatal(self, harness, main_module):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        assert harness.run() == main_module.EXIT_FAILED

    def test_a_cpu_load_failure_logs_the_error_as_fatal(self, harness):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        assert "FATAL: no cpu backend" in harness.report_messages

    def test_a_cpu_load_failure_is_not_retried(self, harness):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        assert harness.model_factory.devices == ["cpu"]

    def test_a_cpu_load_failure_does_not_claim_a_cuda_fallback(self, harness):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        assert harness.messages_containing("CUDA failed") == []

    def test_failing_on_both_devices_is_fatal(self, harness, main_module):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_FAILED
        assert harness.model_factory.devices == ["cuda", "cpu"]

    def test_failing_on_both_devices_logs_the_fallback_then_the_fatal_error(self, harness):
        harness.add_media("clip.mp3")
        harness.cuda_probe.device_count = 1
        harness.model_factory.failures_by_device["cuda"] = RuntimeError("cublas missing")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        messages = harness.report_messages
        assert messages.index("CUDA failed, falling back to CPU") < messages.index("FATAL: no cpu backend")

    def test_no_file_is_transcribed_when_model_loading_is_fatal(self, harness):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        assert harness.transcriber.calls == []
        assert harness.transcripts_in() == []

    def test_no_final_tally_is_logged_when_model_loading_is_fatal(self, harness):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        assert harness.messages_containing("Batch finished") == []

    def test_a_faster_whisper_import_failure_is_fatal(self, harness, main_module, monkeypatch):
        harness.add_media("clip.mp3")
        monkeypatch.setitem(sys.modules, "faster_whisper", None)

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_FAILED
        assert harness.messages_containing("FATAL: import of faster_whisper halted")

    def test_a_fatal_failure_logs_a_traceback(self, harness):
        harness.add_media("clip.mp3")
        harness.model_factory.failures_by_device["cpu"] = RuntimeError("no cpu backend")

        harness.run()

        assert "RuntimeError: no cpu backend" in harness.report_text


class TestSuccessfulRun:
    def test_returns_ok_and_writes_one_transcript_per_file(self, harness, main_module):
        for name in ("one.mp3", "two.mp4", "three.wav"):
            harness.add_media(name)

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_OK
        assert harness.transcripts_in() == ["one.txt", "three.txt", "two.txt"]

    def test_transcript_content_is_the_segments_joined_with_newlines(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = ["first line", "second line", "third line"]

        harness.run()

        assert harness.read_transcript("clip.txt") == "first line\nsecond line\nthird line"

    def test_a_single_segment_is_written_verbatim(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = ["only one"]

        harness.run()

        assert harness.read_transcript("clip.txt") == "only one"

    def test_per_file_line_reports_language_duration_elapsed_and_output_name(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.infos["clip.mp3"] = make_info(duration=42.0, language="es")

        harness.run()

        assert "  clip.mp3: es, 42s audio in 3s -> clip.txt" in harness.report_messages

    def test_a_fractional_duration_is_rounded_in_the_report(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.infos["clip.mp3"] = make_info(duration=17.2, language="en")

        harness.run()

        assert "  clip.mp3: en, 17s audio in 3s -> clip.txt" in harness.report_messages

    def test_start_line_records_every_resolved_argument(self, harness):
        chosen_output = harness.tmp_path / "chosen"

        harness.run("--output", str(chosen_output), "--model", "tiny")

        expected = f"Batch started: input={harness.input_dir} output={chosen_output} model=tiny"
        assert expected in harness.report_messages

    def test_a_non_ok_transcription_status_is_still_treated_as_a_success(self, harness, main_module):
        harness.add_media("clip.mp3")
        harness.transcriber.statuses["clip.mp3"] = "timed_out"

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_OK
        assert "Batch finished: 1 transcribed, 0 failed" in harness.report_messages

    def test_the_saver_receives_the_joined_text(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = ["alpha", "beta"]

        harness.run()

        assert harness.saver.calls[0]["full_text"] == "alpha\nbeta"

    def test_the_saver_receives_the_source_path(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.saver.calls[0]["source_path"] == str(harness.input_dir / "clip.mp3")


class TestSilentFiles:
    @pytest.mark.parametrize(
        "segment_texts",
        [[], [""], ["", ""], ["   "], ["\n", "\t"], [" \r\n "]],
        ids=["no segments", "one empty", "two empty", "spaces", "newline and tab", "carriage return"],
    )
    def test_a_transcript_without_text_is_skipped(self, harness, segment_texts):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = segment_texts

        harness.run()

        assert "  clip.mp3: no speech detected, skipped" in harness.report_messages

    def test_no_file_is_written_for_a_silent_file(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = []

        harness.run()

        assert harness.transcripts_in() == []
        assert harness.saver.calls == []

    def test_a_silent_file_counts_as_neither_success_nor_failure(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = ["   "]

        harness.run()

        assert "Batch finished: 0 transcribed, 0 failed" in harness.report_messages

    def test_a_silent_file_leaves_the_exit_code_at_ok(self, harness, main_module):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = []

        assert harness.run() == main_module.EXIT_OK

    def test_a_silent_file_does_not_stop_the_remaining_files(self, harness):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")
        harness.add_media("three.mp3")
        harness.transcriber.segment_texts["three.mp3"] = []

        harness.run()

        assert harness.transcripts_in() == ["one.txt", "two.txt"]

    def test_a_silent_file_is_excluded_from_the_tally_of_its_siblings(self, harness):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")
        harness.transcriber.segment_texts["two.mp3"] = []

        harness.run()

        assert "Batch finished: 1 transcribed, 0 failed" in harness.report_messages

    def test_a_silent_file_logs_no_success_line(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = []

        harness.run()

        assert harness.messages_containing("audio in") == []

    def test_whitespace_only_text_is_never_handed_to_the_saver(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.segment_texts["clip.mp3"] = ["\t", " "]

        harness.run()

        assert harness.saver.calls == []


class TestPerFileFailureIsolation:
    def test_a_failing_file_does_not_stop_the_batch(self, harness):
        for name in ("one.mp3", "two.mp3", "three.mp3"):
            harness.add_media(name)
        harness.transcriber.failures["one.mp3"] = RuntimeError("decode error")

        harness.run()

        assert harness.transcriber.transcribed_names == ["one.mp3", "three.mp3", "two.mp3"]

    def test_successful_transcripts_are_still_written_when_another_file_fails(self, harness):
        for name in ("one.mp3", "two.mp3", "three.mp3"):
            harness.add_media(name)
        harness.transcriber.failures["three.mp3"] = RuntimeError("decode error")

        harness.run()

        assert harness.transcripts_in() == ["one.txt", "two.txt"]

    def test_the_failure_is_logged_with_the_error_message(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.failures["clip.mp3"] = RuntimeError("decode error")

        harness.run()

        assert "  clip.mp3: FAILED decode error" in harness.report_messages

    def test_the_failure_logs_a_traceback(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.failures["clip.mp3"] = ValueError("bad frame")

        harness.run()

        assert "ValueError: bad frame" in harness.report_text
        assert "Traceback (most recent call last):" in harness.report_text

    def test_the_exit_code_is_failed_when_any_file_fails(self, harness, main_module):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")
        harness.transcriber.failures["two.mp3"] = RuntimeError("decode error")

        assert harness.run() == main_module.EXIT_FAILED

    def test_a_save_failure_is_counted_as_a_file_failure(self, harness, main_module):
        harness.add_media("clip.mp3")
        harness.saver.failures["clip.mp3"] = OSError("disk full")

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_FAILED
        assert "  clip.mp3: FAILED disk full" in harness.report_messages

    def test_a_save_failure_leaves_no_transcript_behind(self, harness):
        harness.add_media("clip.mp3")
        harness.saver.failures["clip.mp3"] = OSError("disk full")

        harness.run()

        assert harness.transcripts_in() == []

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("boom"), ValueError("boom"), OSError("boom"), KeyError("boom"), ZeroDivisionError("boom")],
        ids=["RuntimeError", "ValueError", "OSError", "KeyError", "ZeroDivisionError"],
    )
    def test_any_exception_type_is_swallowed_per_file(self, harness, main_module, error):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")
        harness.transcriber.failures["one.mp3"] = error

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_FAILED
        assert harness.transcripts_in() == ["two.txt"]

    def test_every_file_may_fail(self, harness, main_module):
        for name in ("one.mp3", "two.mp3", "three.mp3"):
            harness.add_media(name)
            harness.transcriber.failures[name] = RuntimeError("decode error")

        exit_code = harness.run()

        assert exit_code == main_module.EXIT_FAILED
        assert "Batch finished: 0 transcribed, 3 failed" in harness.report_messages

    def test_a_failing_file_produces_no_success_line(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.failures["clip.mp3"] = RuntimeError("decode error")

        harness.run()

        assert harness.messages_containing("audio in") == []


class TestFinalTally:
    def test_the_final_line_counts_successes_and_failures(self, harness):
        for name in ("one.mp3", "two.mp3", "three.mp3", "four.mp3"):
            harness.add_media(name)
        harness.transcriber.failures["four.mp3"] = RuntimeError("decode error")
        harness.transcriber.segment_texts["three.mp3"] = []

        harness.run()

        assert "Batch finished: 2 transcribed, 1 failed" in harness.report_messages

    def test_the_final_line_is_the_last_message(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.report_messages[-1] == "Batch finished: 1 transcribed, 0 failed"

    @pytest.mark.parametrize(
        ("failing_names", "expected_tally"),
        [
            ((), "Batch finished: 3 transcribed, 0 failed"),
            (("one.mp3",), "Batch finished: 2 transcribed, 1 failed"),
            (("one.mp3", "two.mp3"), "Batch finished: 1 transcribed, 2 failed"),
            (("one.mp3", "two.mp3", "three.mp3"), "Batch finished: 0 transcribed, 3 failed"),
        ],
        ids=["none fail", "one fails", "two fail", "all fail"],
    )
    def test_the_tally_tracks_the_number_of_failures(self, harness, failing_names, expected_tally):
        for name in ("one.mp3", "two.mp3", "three.mp3"):
            harness.add_media(name)
        for name in failing_names:
            harness.transcriber.failures[name] = RuntimeError("decode error")

        harness.run()

        assert expected_tally in harness.report_messages

    @pytest.mark.parametrize("failure_count", [0, 1, 3])
    def test_the_exit_code_is_ok_only_when_nothing_failed(self, harness, main_module, failure_count):
        names = ["one.mp3", "two.mp3", "three.mp3"]
        for name in names:
            harness.add_media(name)
        for name in names[:failure_count]:
            harness.transcriber.failures[name] = RuntimeError("decode error")

        exit_code = harness.run()

        expected_exit = main_module.EXIT_OK if failure_count == 0 else main_module.EXIT_FAILED
        assert exit_code == expected_exit

    @pytest.mark.parametrize(
        ("run_scenario", "expected_exit_name"),
        list(zip(
            RETURN_PATH_SCENARIOS,
            ["EXIT_BAD_USAGE", "EXIT_OK", "EXIT_OK", "EXIT_FAILED", "EXIT_FAILED"],
            strict=True,
        )),
        ids=RETURN_PATH_IDS,
    )
    def test_every_return_path_yields_its_documented_exit_code(
            self, harness, main_module, run_scenario, expected_exit_name):
        assert run_scenario(harness) == getattr(main_module, expected_exit_name)


class TestReportFile:
    def test_the_report_is_batch_txt_inside_the_log_directory(self, harness):
        harness.run()

        assert harness.report_path.is_file()

    def test_the_log_directory_is_created_when_missing(self, harness):
        harness.use_log_directory(harness.tmp_path / "fresh" / "logs")
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.report_path.is_file()

    def test_the_report_is_appended_to_across_runs(self, harness):
        harness.add_media("first.mp3")
        harness.run()
        first_run_text = harness.report_text

        harness.add_media("second.mp3")
        harness.run()

        assert harness.report_text.startswith(first_run_text)
        assert "  second.mp3: en, 12s audio in 3s -> second.txt" in harness.report_messages

    def test_a_second_run_does_not_erase_the_first_runs_lines(self, harness):
        harness.add_media("first.mp3")
        harness.run()

        harness.add_media("second.mp3")
        harness.run()

        assert len(harness.messages_containing("Batch started")) == 2
        assert len(harness.messages_containing("Batch finished")) == 2

    def test_a_second_run_skips_what_the_first_run_transcribed(self, harness):
        harness.add_media("first.mp3")
        harness.run()

        harness.add_media("second.mp3")
        harness.run()

        assert harness.transcriber.transcribed_names == ["first.mp3", "second.mp3"]

    def test_a_repeated_run_over_the_same_folder_finds_nothing_new(self, harness):
        harness.add_media("clip.mp3")
        harness.run()

        harness.run()

        assert "Nothing new to transcribe" in harness.report_messages

    def test_every_line_of_a_clean_run_is_timestamped(self, harness):
        harness.add_media("one.mp3")
        harness.add_media("two.mp3")

        harness.run()

        assert all(TIMESTAMP_PATTERN.match(line) for line in harness.report_lines)

    def test_the_report_ends_with_a_newline(self, harness):
        harness.add_media("clip.mp3")

        harness.run()

        assert harness.report_text.endswith("\n")

    def test_traceback_continuation_lines_carry_no_timestamp(self, harness):
        harness.add_media("clip.mp3")
        harness.transcriber.failures["clip.mp3"] = ValueError("bad frame")

        harness.run()

        untimestamped = [line for line in harness.report_lines if line and not TIMESTAMP_PATTERN.match(line)]
        assert "ValueError: bad frame" in untimestamped

    @pytest.mark.parametrize("run_scenario", RETURN_PATH_SCENARIOS, ids=RETURN_PATH_IDS)
    def test_the_report_is_complete_and_readable_after_every_return_path(self, harness, run_scenario):
        run_scenario(harness)

        assert harness.report_messages[0].startswith("Batch started: input=")
        assert harness.report_text.endswith("\n")

    @pytest.mark.parametrize("run_scenario", RETURN_PATH_SCENARIOS, ids=RETURN_PATH_IDS)
    def test_the_report_can_be_reopened_and_appended_after_every_return_path(self, harness, run_scenario):
        run_scenario(harness)
        text_written_by_the_batch = harness.report_text

        with open(harness.report_path, "a", encoding="utf-8") as handle:
            handle.write("appended by the test\n")

        assert harness.report_text == text_written_by_the_batch + "appended by the test\n"

    @pytest.mark.parametrize("run_scenario", RETURN_PATH_SCENARIOS, ids=RETURN_PATH_IDS)
    def test_the_report_handle_is_released_on_every_return_path(self, harness, run_scenario):
        run_scenario(harness)

        harness.report_path.unlink()

        assert not harness.report_path.exists()
