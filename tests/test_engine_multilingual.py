"""Unit tests for engine.MultilingualInfo and engine.transcribe_audio_multilingual."""

import numpy
import pytest
from doubles import FakeClock, FakeSegment, FakeWhisperModel, make_info

import engine

SAMPLES_PER_SECOND = 16000
SECONDS_PER_CHUNK = 30
SAMPLES_PER_CHUNK = SECONDS_PER_CHUNK * SAMPLES_PER_SECOND
AUDIO_PATH = "C:\\media\\interview.mp4"


class CallRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *arguments):
        self.calls.append(arguments)

    @property
    def call_count(self):
        return len(self.calls)


class NeverCancelled:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return False


class CancelledFromCall:
    def __init__(self, first_cancelled_call):
        self.first_cancelled_call = first_cancelled_call
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls >= self.first_cancelled_call


class DecodedAudioStub:
    def __init__(self, total_samples):
        self.array = numpy.arange(total_samples, dtype=numpy.float32)
        self.calls = []

    def __call__(self, audio_path, sampling_rate=None):
        self.calls.append((audio_path, sampling_rate))
        return self.array


def chunk_response(texts=(), language="en", language_probability=0.9, duration_after_vad=None):
    segments = [FakeSegment(text) for text in texts]
    info = make_info(language=language, language_probability=language_probability,
                     duration_after_vad=duration_after_vad)
    return segments, info


def build_model(*responses, repeat_last=False):
    return FakeWhisperModel(list(responses), repeat_last=repeat_last)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch, fake_clock):
    monkeypatch.setattr(engine.time, "time", fake_clock)
    return fake_clock


@pytest.fixture
def install_clock(monkeypatch):
    def installer(step):
        clock = FakeClock(step=step)
        monkeypatch.setattr(engine.time, "time", clock)
        return clock

    return installer


@pytest.fixture
def install_decoded_audio(monkeypatch):
    def installer(total_samples):
        stub = DecodedAudioStub(total_samples)
        monkeypatch.setattr(engine, "decode_audio", stub)
        return stub

    return installer


class TestChunkingConstants:
    def test_sampling_rate_is_sixteen_kilohertz(self):
        assert engine.SAMPLING_RATE == SAMPLES_PER_SECOND

    def test_chunk_length_is_thirty_seconds(self):
        assert engine.CHUNK_SECONDS == SECONDS_PER_CHUNK


class TestMultilingualInfo:
    @pytest.mark.parametrize(
        ("languages", "expected_language"),
        [
            ({"en": 1}, "en"),
            ({"ja": 5}, "ja"),
            ({"en": 2, "es": 1}, "en+es"),
            ({"es": 1, "en": 2}, "en+es"),
            ({"pt": 1, "de": 3, "ar": 2}, "ar+de+pt"),
            ({}, "unknown"),
            (None, "unknown"),
        ],
    )
    def test_combines_language_keys_into_a_sorted_plus_joined_label(self, languages, expected_language):
        info = engine.MultilingualInfo(45.0, 40.0, languages, 0.8)

        assert info.language == expected_language

    def test_a_single_language_is_reported_without_a_separator(self):
        info = engine.MultilingualInfo(30.0, 28.0, {"fr": 3}, 0.77)

        assert info.language == "fr"
        assert "+" not in info.language

    def test_an_empty_languages_mapping_reports_unknown(self):
        info = engine.MultilingualInfo(30.0, 28.0, {}, 0.0)

        assert info.language == "unknown"

    def test_stores_duration_verbatim(self):
        info = engine.MultilingualInfo(123.456, 0.0, {"en": 1}, 0.5)

        assert info.duration == 123.456

    def test_stores_duration_after_vad_verbatim(self):
        info = engine.MultilingualInfo(123.456, 98.75, {"en": 1}, 0.5)

        assert info.duration_after_vad == 98.75

    def test_keeps_a_zero_duration_after_vad_rather_than_replacing_it(self):
        info = engine.MultilingualInfo(10.0, 0.0, {"en": 1}, 0.5)

        assert info.duration_after_vad == 0.0

    def test_language_probability_is_the_average_it_was_given(self):
        info = engine.MultilingualInfo(10.0, 9.0, {"en": 1, "es": 1}, 0.6543)

        assert info.language_probability == 0.6543

    def test_a_zero_average_probability_is_preserved(self):
        info = engine.MultilingualInfo(10.0, 9.0, {}, 0.0)

        assert info.language_probability == 0.0

    def test_keeps_the_language_tally_mapping_as_given(self):
        tally = {"en": 4, "de": 2}

        info = engine.MultilingualInfo(60.0, 55.0, tally, 0.9)

        assert info.languages is tally
        assert info.languages == {"en": 4, "de": 2}


class TestChunkCount:
    @pytest.mark.parametrize(
        ("total_samples", "expected_chunk_count"),
        [
            (0, 1),
            (1, 1),
            (SAMPLES_PER_SECOND, 1),
            (SAMPLES_PER_CHUNK, 1),
            (SAMPLES_PER_CHUNK + 1, 2),
            (45 * SAMPLES_PER_SECOND, 2),
            (2 * SAMPLES_PER_CHUNK, 2),
            (2 * SAMPLES_PER_CHUNK + 1, 3),
            (90 * SAMPLES_PER_SECOND, 3),
        ],
    )
    def test_transcribes_the_expected_number_of_chunks(self, install_decoded_audio, total_samples,
                                                       expected_chunk_count):
        install_decoded_audio(total_samples)
        model = build_model(chunk_response(["hello"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == expected_chunk_count

    def test_zero_length_audio_still_produces_one_chunk(self, install_decoded_audio):
        install_decoded_audio(0)
        model = build_model(chunk_response([]))

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 1
        assert len(model.calls[0]["audio"]) == 0

    def test_exactly_thirty_seconds_is_a_single_chunk(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response(["only chunk"]))

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 1

    def test_the_chunk_total_reported_to_on_chunk_matches_the_number_of_transcriptions(
            self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert [call[1] for call in on_chunk.calls] == [3, 3, 3]


class TestChunkSlicing:
    def test_a_full_chunk_carries_thirty_seconds_of_samples(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert [len(call["audio"]) for call in model.calls] == [
            SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK]

    def test_the_final_short_chunk_is_not_padded(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert [len(call["audio"]) for call in model.calls] == [
            SAMPLES_PER_CHUNK, 15 * SAMPLES_PER_SECOND]

    def test_a_single_trailing_sample_becomes_a_one_sample_chunk(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK + 1)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls[1]["audio"]) == 1

    def test_chunks_start_where_the_previous_chunk_ended(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        first_samples = [call["audio"][0] for call in model.calls]
        assert first_samples == [0.0, float(SAMPLES_PER_CHUNK), float(2 * SAMPLES_PER_CHUNK)]

    def test_the_chunks_together_cover_the_whole_decoded_array(self, install_decoded_audio):
        decoded = install_decoded_audio(70 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        rejoined = numpy.concatenate([call["audio"] for call in model.calls])
        assert numpy.array_equal(rejoined, decoded.array)


class TestDecodingAndTranscribeArguments:
    def test_decodes_the_requested_path_at_the_sixteen_kilohertz_sampling_rate(self, install_decoded_audio):
        decoded = install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response(["a"]))

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert decoded.calls == [(AUDIO_PATH, SAMPLES_PER_SECOND)]

    def test_decodes_the_audio_exactly_once_regardless_of_chunk_count(self, install_decoded_audio):
        decoded = install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(decoded.calls) == 1

    def test_passes_decoded_samples_to_the_model_rather_than_the_path(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response(["a"]))

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert isinstance(model.last_call["audio"], numpy.ndarray)

    def test_every_chunk_is_transcribed_with_automatic_language_detection(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language="en"),
            chunk_response(["b"], language="es"),
            chunk_response(["c"], language="pt"),
        )

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert [call["language"] for call in model.calls] == [None, None, None]

    def test_every_chunk_enables_the_voice_activity_filter(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert [call["vad_filter"] for call in model.calls] == [True, True]

    def test_every_chunk_uses_a_half_second_minimum_silence_window(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert [call["vad_parameters"] for call in model.calls] == [
            {"min_silence_duration_ms": 500}, {"min_silence_duration_ms": 500}]

    def test_every_chunk_uses_a_beam_size_of_five(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert [call["beam_size"] for call in model.calls] == [5, 5]


class TestTextCollection:
    def test_returns_the_stripped_segment_text_of_every_chunk_in_order(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["  hello  ", "\tworld\n"]),
            chunk_response([" hola "]),
        )

        text_parts, _info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert text_parts == ["hello", "world", "hola"]

    def test_a_chunk_without_segments_contributes_no_text(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["first"]),
            chunk_response([]),
            chunk_response(["third"]),
        )

        text_parts, _info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert text_parts == ["first", "third"]

    def test_silent_audio_returns_no_text_and_an_ok_status(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response([]))

        text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert text_parts == []
        assert status == "ok"

    def test_consumes_every_segment_of_a_chunk_when_not_cancelled(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response(["one", "two", "three"]))

        engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert model.iterators[0].consumed == 3


class TestLanguageTally:
    def test_repeated_detections_of_the_same_language_are_counted(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language="en"),
            chunk_response(["b"], language="en"),
            chunk_response(["c"], language="es"),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.languages == {"en": 2, "es": 1}

    def test_two_english_chunks_and_one_spanish_chunk_combine_into_en_plus_es(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language="en"),
            chunk_response(["b"], language="es"),
            chunk_response(["c"], language="en"),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.language == "en+es"

    def test_a_uniform_language_is_reported_alone(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"], language="pt"), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.languages == {"pt": 3}
        assert info.language == "pt"

    def test_three_distinct_languages_are_joined_in_alphabetical_order(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language="pt"),
            chunk_response(["b"], language="de"),
            chunk_response(["c"], language="ar"),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.language == "ar+de+pt"

    def test_a_chunk_without_segments_is_still_counted_in_the_language_tally(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response([], language="fr"),
            chunk_response(["bonjour"], language="fr"),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.languages == {"fr": 2}


class TestAverageProbability:
    def test_probability_is_the_mean_across_chunks(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language_probability=0.9),
            chunk_response(["b"], language_probability=0.6),
            chunk_response(["c"], language_probability=0.3),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.language_probability == pytest.approx(0.6)

    def test_a_single_chunk_reports_its_own_probability(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response(["a"], language_probability=0.42))

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.language_probability == pytest.approx(0.42)

    def test_probabilities_average_across_languages_not_within_them(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language="en", language_probability=1.0),
            chunk_response(["b"], language="en", language_probability=0.5),
            chunk_response(["c"], language="es", language_probability=0.0),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.language_probability == pytest.approx(0.5)

    def test_no_detections_yield_a_zero_probability(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(1))

        assert info.language_probability == 0.0

    def test_no_detections_report_an_unknown_language(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(1))

        assert info.language == "unknown"
        assert info.languages == {}


class TestDurationAfterVadAccumulation:
    def test_accumulates_the_vad_duration_of_every_chunk(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], duration_after_vad=28.0),
            chunk_response(["b"], duration_after_vad=25.5),
            chunk_response(["c"], duration_after_vad=10.25),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.duration_after_vad == pytest.approx(63.75)

    def test_an_info_object_lacking_the_attribute_is_skipped_without_error(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], duration_after_vad=None),
            chunk_response(["b"], duration_after_vad=None),
        )

        _text_parts, info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.duration_after_vad == 0.0
        assert status == "ok"

    def test_a_zero_vad_duration_contributes_nothing(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], duration_after_vad=0.0),
            chunk_response(["b"], duration_after_vad=12.0),
        )

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.duration_after_vad == pytest.approx(12.0)

    def test_present_missing_and_zero_vad_durations_mix_without_error(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], duration_after_vad=20.0),
            chunk_response(["b"], duration_after_vad=None),
            chunk_response(["c"], duration_after_vad=0.0),
        )

        _text_parts, info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.duration_after_vad == pytest.approx(20.0)
        assert status == "ok"

    def test_only_transcribed_chunks_contribute_their_vad_duration(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"], duration_after_vad=11.0), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3))

        assert info.duration_after_vad == pytest.approx(11.0)


class TestOnChunkCallback:
    def test_reports_one_based_chunk_numbers_and_the_total(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert [(call[0], call[1]) for call in on_chunk.calls] == [(1, 3), (2, 3), (3, 3)]

    def test_reports_the_real_second_boundaries_of_each_chunk(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert [(call[2], call[3]) for call in on_chunk.calls] == [(0.0, 30.0), (30.0, 45.0)]

    def test_the_final_boundary_stops_at_the_real_end_of_the_audio(self, install_decoded_audio):
        install_decoded_audio(70 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert on_chunk.calls[-1][2:4] == (60.0, 70.0)

    def test_reports_the_detected_language_and_probability_of_each_chunk(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["a"], language="en", language_probability=0.81),
            chunk_response(["b"], language="es", language_probability=0.64),
        )
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert [(call[4], call[5]) for call in on_chunk.calls] == [("en", 0.81), ("es", 0.64)]

    def test_reports_a_zero_length_chunk_as_an_empty_span(self, install_decoded_audio):
        install_decoded_audio(0)
        model = build_model(chunk_response([]))
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert on_chunk.calls == [(1, 1, 0.0, 0.0, "en", 0.9)]

    def test_passes_six_positional_arguments(self, install_decoded_audio):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        model = build_model(chunk_response(["a"]))
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert len(on_chunk.calls[0]) == 6

    def test_is_optional(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=None)

        assert text_parts == ["a", "a"]
        assert status == "ok"

    def test_is_not_called_for_chunks_skipped_by_cancellation(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3), on_chunk=on_chunk)

        assert on_chunk.call_count == 1


class TestOnProgressCallback:
    def test_reports_the_chunk_end_and_the_total_duration(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_progress = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_progress=on_progress)

        assert on_progress.calls == [(30.0, 45.0), (45.0, 45.0)]

    def test_the_last_report_reaches_the_full_duration(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_progress = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_progress=on_progress)

        assert on_progress.calls[-1] == (90.0, 90.0)

    def test_is_skipped_entirely_when_the_duration_is_zero(self, install_decoded_audio):
        install_decoded_audio(0)
        model = build_model(chunk_response([]))
        on_progress = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_progress=on_progress)

        assert on_progress.calls == []

    def test_is_reported_once_per_transcribed_chunk(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_progress = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_progress=on_progress)

        assert on_progress.call_count == len(model.calls)

    def test_is_optional(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, _info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, on_progress=None)

        assert status == "ok"

    def test_is_not_reported_for_chunks_skipped_by_cancellation(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_progress = CallRecorder()

        engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3), on_progress=on_progress)

        assert on_progress.calls == [(30.0, 90.0)]


class TestCancellation:
    def test_cancelling_before_the_first_chunk_transcribes_nothing(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        text_parts, _info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(1))

        assert model.calls == []
        assert text_parts == []
        assert status == "ok"

    def test_cancelling_before_a_later_chunk_returns_the_earlier_results(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["first", "second"], language="en"),
            chunk_response(["never"], language="es"),
        )

        text_parts, info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(4))

        assert text_parts == ["first", "second"]
        assert info.languages == {"en": 1}
        assert len(model.calls) == 1
        assert status == "ok"

    def test_a_partial_run_still_reports_the_full_audio_duration(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3))

        assert info.duration == 90.0

    def test_cancelling_inside_the_segment_loop_stops_collecting_text(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["kept", "dropped", "dropped too"], language="en"))

        text_parts, _info, _status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3))

        assert text_parts == ["kept"]
        assert model.iterators[0].consumed == 2

    def test_cancelling_inside_the_segment_loop_still_tallies_the_chunk_language(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(
            chunk_response(["kept", "dropped"], language="ja", language_probability=0.55,
                           duration_after_vad=7.0))

        _text_parts, info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3))

        assert info.languages == {"ja": 1}
        assert info.language == "ja"
        assert info.language_probability == pytest.approx(0.55)
        assert info.duration_after_vad == pytest.approx(7.0)
        assert status == "ok"

    def test_cancelling_inside_the_segment_loop_still_notifies_on_chunk(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["kept", "dropped"], language="ja", language_probability=0.55))
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3), on_chunk=on_chunk)

        assert on_chunk.calls == [(1, 2, 0.0, 30.0, "ja", 0.55)]

    def test_cancelling_inside_the_segment_loop_stops_the_remaining_chunks(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["kept", "dropped"], language="en"))

        engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(3))

        assert len(model.calls) == 1

    def test_a_callback_that_never_cancels_lets_every_chunk_run(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        is_cancelled = NeverCancelled()

        text_parts, _info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=is_cancelled)

        assert len(model.calls) == 3
        assert text_parts == ["a", "a", "a"]
        assert status == "ok"

    def test_the_cancellation_callback_is_consulted_before_each_chunk_and_each_segment(
            self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a", "b"]), chunk_response(["c"]))
        is_cancelled = NeverCancelled()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, is_cancelled=is_cancelled)

        assert is_cancelled.calls == 5

    def test_no_cancellation_callback_means_no_cancellation(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, _info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=None)

        assert len(model.calls) == 3
        assert status == "ok"


class TestTimeout:
    def test_exceeding_the_budget_before_any_chunk_reports_timed_out(self, install_decoded_audio, install_clock):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        install_clock(step=400.0)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert status == "timed_out"
        assert model.calls == []
        assert text_parts == []

    def test_a_timeout_before_any_chunk_reports_unknown_language_and_zero_probability(
            self, install_decoded_audio, install_clock):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        install_clock(step=400.0)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.language == "unknown"
        assert info.language_probability == 0.0
        assert info.duration_after_vad == 0.0

    def test_a_timeout_returns_the_chunks_completed_so_far(self, install_decoded_audio, install_clock):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        install_clock(step=400.0)
        model = build_model(
            chunk_response(["done"], language="en", duration_after_vad=9.0),
            chunk_response(["never"], language="es"),
            chunk_response(["never either"], language="pt"),
        )

        text_parts, info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert status == "timed_out"
        assert text_parts == ["done"]
        assert info.languages == {"en": 1}
        assert info.duration_after_vad == pytest.approx(9.0)

    def test_the_budget_is_eight_times_the_audio_duration_when_that_exceeds_thirty_seconds(
            self, install_decoded_audio, install_clock):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        install_clock(step=100.0)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 2
        assert status == "ok"

    def test_the_budget_never_falls_below_thirty_seconds_for_short_audio(
            self, install_decoded_audio, install_clock):
        install_decoded_audio(SAMPLES_PER_SECOND)
        install_clock(step=20.0)
        model = build_model(chunk_response(["a"]))

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 1
        assert status == "ok"

    def test_short_audio_still_times_out_past_the_thirty_second_floor(
            self, install_decoded_audio, install_clock):
        install_decoded_audio(SAMPLES_PER_SECOND)
        install_clock(step=31.0)
        model = build_model(chunk_response(["a"]))

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 0
        assert status == "timed_out"

    def test_zero_length_audio_uses_the_thirty_second_floor(self, install_decoded_audio, install_clock):
        install_decoded_audio(0)
        install_clock(step=29.0)
        model = build_model(chunk_response([]))

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 1
        assert status == "ok"

    def test_elapsed_time_exactly_equal_to_the_budget_does_not_time_out(
            self, install_decoded_audio, install_clock):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        install_clock(step=240.0)
        model = build_model(chunk_response(["a"]))

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 1
        assert status == "ok"

    def test_elapsed_time_just_past_the_budget_times_out(self, install_decoded_audio, install_clock):
        install_decoded_audio(SAMPLES_PER_CHUNK)
        install_clock(step=240.5)
        model = build_model(chunk_response(["a"]))

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(model.calls) == 0
        assert status == "timed_out"

    def test_a_timeout_mid_run_leaves_the_remaining_chunks_untranscribed(
            self, install_decoded_audio, install_clock):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        install_clock(step=400.0)
        model = build_model(chunk_response(["a"]), repeat_last=True)
        on_chunk = CallRecorder()

        engine.transcribe_audio_multilingual(model, AUDIO_PATH, on_chunk=on_chunk)

        assert len(model.calls) == 1
        assert on_chunk.call_count == 1

    def test_cancellation_takes_precedence_over_an_exceeded_budget(self, install_decoded_audio, install_clock):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        install_clock(step=400.0)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, _info, status = engine.transcribe_audio_multilingual(
            model, AUDIO_PATH, is_cancelled=CancelledFromCall(1))

        assert status == "ok"
        assert model.calls == []

    def test_a_frozen_clock_never_times_out(self, install_decoded_audio, frozen_clock):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert status == "ok"
        assert frozen_clock.reads == 4


class TestReturnValue:
    def test_returns_text_parts_a_multilingual_info_and_a_status(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        result = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert len(result) == 3
        assert isinstance(result[0], list)
        assert isinstance(result[1], engine.MultilingualInfo)
        assert result[2] == "ok"

    @pytest.mark.parametrize(
        ("total_samples", "expected_duration"),
        [
            (0, 0.0),
            (8000, 0.5),
            (SAMPLES_PER_CHUNK, 30.0),
            (72000, 4.5),
            (90 * SAMPLES_PER_SECOND, 90.0),
        ],
    )
    def test_duration_is_derived_from_the_decoded_sample_count(self, install_decoded_audio,
                                                               total_samples, expected_duration):
        install_decoded_audio(total_samples)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, info, _status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert info.duration == pytest.approx(expected_duration)

    def test_a_successful_run_reports_the_ok_status(self, install_decoded_audio):
        install_decoded_audio(90 * SAMPLES_PER_SECOND)
        model = build_model(chunk_response(["a"]), repeat_last=True)

        _text_parts, _info, status = engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert status == "ok"

    def test_a_model_failure_is_not_swallowed(self, install_decoded_audio):
        install_decoded_audio(45 * SAMPLES_PER_SECOND)
        model = build_model(RuntimeError("cuda out of memory"))

        with pytest.raises(RuntimeError, match="cuda out of memory"):
            engine.transcribe_audio_multilingual(model, AUDIO_PATH)

    def test_a_decode_failure_is_not_swallowed(self, monkeypatch):
        def failing_decode_audio(audio_path, sampling_rate=None):
            raise OSError("unsupported container")

        monkeypatch.setattr(engine, "decode_audio", failing_decode_audio)
        model = build_model(chunk_response(["a"]))

        with pytest.raises(OSError, match="unsupported container"):
            engine.transcribe_audio_multilingual(model, AUDIO_PATH)

        assert model.calls == []
