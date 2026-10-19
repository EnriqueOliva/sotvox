"""Unit tests for engine.transcribe_audio."""

import types

import pytest
from doubles import FakeSegment, FakeWhisperModel, make_info

import engine

AUDIO_PATH = "C:\\clips\\interview.mp3"
LANGUAGE_CODE = "es"
FIRST_CALL = 0
RETRY_CALL = 1


class ProgressRecorder:
    def __init__(self, clock=None, advance_seconds=0.0):
        self.calls = []
        self.clock = clock
        self.advance_seconds = advance_seconds

    def __call__(self, current_position, total_duration):
        self.calls.append((current_position, total_duration))
        if self.clock is not None:
            self.clock.advance(self.advance_seconds)


class CancelAfter:
    def __init__(self, allowed_checks):
        self.allowed_checks = allowed_checks
        self.invocations = 0

    def __call__(self):
        self.invocations += 1
        return self.invocations > self.allowed_checks


class NeverCancels:
    def __init__(self):
        self.invocations = 0

    def __call__(self):
        self.invocations += 1
        return False


class ExplodingCallback:
    def __init__(self, error):
        self.error = error
        self.invocations = 0

    def __call__(self, *args):
        self.invocations += 1
        raise self.error


def numbered_segments(count, seconds_per_segment=1.0):
    return [
        FakeSegment(
            f"part {index}",
            start=index * seconds_per_segment,
            end=(index + 1) * seconds_per_segment,
        )
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def clock(fake_clock, monkeypatch):
    monkeypatch.setattr(engine, "time", types.SimpleNamespace(time=fake_clock))
    return fake_clock


class TestTranscribeAudioHappyPath:
    def test_returns_the_stripped_text_of_every_segment_with_an_ok_status(self):
        model = FakeWhisperModel([
            ([FakeSegment("  good morning  "), FakeSegment("\nhow are you\t")], make_info()),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert text_parts == ["good morning", "how are you"]
        assert status == "ok"

    def test_keeps_the_segments_in_the_order_the_model_produced_them(self):
        model = FakeWhisperModel([
            ([FakeSegment("first"), FakeSegment("second"), FakeSegment("third")], make_info()),
        ])

        text_parts, _info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert text_parts == ["first", "second", "third"]

    @pytest.mark.parametrize(
        ("raw_text", "expected_text"),
        [
            ("  padded  ", "padded"),
            ("\ttabbed\t", "tabbed"),
            ("\nnewlines\n", "newlines"),
            ("\r\ncarriage\r\n", "carriage"),
            ("no padding", "no padding"),
            ("  inner  spaces  ", "inner  spaces"),
        ],
    )
    def test_strips_the_surrounding_whitespace_of_each_segment(self, raw_text, expected_text):
        model = FakeWhisperModel([([FakeSegment(raw_text)], make_info())])

        text_parts, _info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert text_parts == [expected_text]

    def test_consumes_every_segment_when_nothing_interrupts_the_loop(self):
        model = FakeWhisperModel([(numbered_segments(4), make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.iterators[FIRST_CALL].consumed == 4

    def test_calls_the_model_exactly_once_when_the_first_pass_produces_text(self):
        model = FakeWhisperModel([([FakeSegment("something")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert len(model.calls) == 1

    def test_a_single_segment_clip_is_transcribed_without_a_retry(self):
        model = FakeWhisperModel([([FakeSegment(" lonely ")], make_info(duration=0.4))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["lonely"], "ok")
        assert len(model.calls) == 1


class TestTranscribeAudioModelArguments:
    def test_passes_the_audio_path_to_the_model(self):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.calls[FIRST_CALL]["audio"] == AUDIO_PATH

    def test_requests_a_beam_size_of_five(self):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.calls[FIRST_CALL]["beam_size"] == 5

    def test_enables_the_voice_activity_filter_on_the_first_pass(self):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.calls[FIRST_CALL]["vad_filter"] is True

    def test_asks_for_a_five_hundred_millisecond_minimum_silence(self):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.calls[FIRST_CALL]["vad_parameters"] == {"min_silence_duration_ms": 500}

    @pytest.mark.parametrize("language_code", ["es", "en", "pt", "zh", None])
    def test_forwards_the_language_code_unchanged(self, language_code):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, language_code)

        assert model.calls[FIRST_CALL]["language"] == language_code

    def test_does_not_disable_conditioning_on_previous_text_on_the_first_pass(self):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert "condition_on_previous_text" not in model.calls[FIRST_CALL]

    def test_the_first_pass_keyword_set_is_pinned(self):
        model = FakeWhisperModel([([FakeSegment("text")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert set(model.calls[FIRST_CALL]) == {
            "audio", "language", "beam_size", "vad_filter", "vad_parameters"}


class TestTranscribeAudioProgressReporting:
    def test_reports_each_segment_end_against_the_clip_duration(self):
        segments = [FakeSegment("a", end=2.0), FakeSegment("b", end=5.5), FakeSegment("c", end=9.0)]
        model = FakeWhisperModel([(segments, make_info(duration=10.0))])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(2.0, 10.0), (5.5, 10.0), (9.0, 10.0)]

    def test_reports_the_raw_segment_end_even_when_it_runs_past_the_duration(self):
        model = FakeWhisperModel([([FakeSegment("a", end=99.0)], make_info(duration=10.0))])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(99.0, 10.0)]

    @pytest.mark.parametrize("duration", [0.0, -4.0])
    def test_does_not_report_progress_when_the_duration_is_not_positive(self, duration):
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=duration))])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == []

    def test_still_collects_the_text_when_a_zero_duration_suppresses_progress(self):
        model = FakeWhisperModel([([FakeSegment(" heard ")], make_info(duration=0.0))])
        progress = ProgressRecorder()

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert (text_parts, status) == (["heard"], "ok")

    def test_reports_progress_for_a_duration_just_above_zero(self):
        model = FakeWhisperModel([([FakeSegment("a", end=0.05)], make_info(duration=0.1))])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(0.05, 0.1)]

    def test_runs_without_a_progress_callback(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info())])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (len(text_parts), status) == (3, "ok")

    def test_does_not_report_progress_for_a_segment_dropped_by_the_deadline(self, clock):
        clock.step = 16.0
        model = FakeWhisperModel([(numbered_segments(4), make_info(duration=1.0))])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(1.0, 1.0)]


class TestTranscribeAudioCancellation:
    def test_stops_pulling_segments_once_the_cancel_callback_returns_true(self):
        model = FakeWhisperModel([(numbered_segments(5), make_info())])

        engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(2))

        assert model.iterators[FIRST_CALL].consumed == 3

    def test_returns_the_text_collected_before_the_cancellation(self):
        model = FakeWhisperModel([(numbered_segments(5), make_info())])

        text_parts, _info, _status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(2))

        assert text_parts == ["part 0", "part 1"]

    def test_reports_an_ok_status_after_a_cancellation(self):
        model = FakeWhisperModel([(numbered_segments(5), make_info())])

        _text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(2))

        assert status == "ok"

    def test_discards_the_segment_already_pulled_from_the_iterator(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info())])

        text_parts, _info, _status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(0))

        assert text_parts == []
        assert model.iterators[FIRST_CALL].consumed == 1

    def test_does_not_report_progress_for_the_cancelled_segment(self):
        model = FakeWhisperModel([(numbered_segments(5), make_info(duration=10.0))])
        progress = ProgressRecorder()

        engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE,
            on_progress=progress, is_cancelled=CancelAfter(2))

        assert progress.calls == [(1.0, 10.0), (2.0, 10.0)]

    def test_checks_the_cancel_flag_before_the_deadline(self, clock):
        clock.step = 1000.0
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=1.0))])

        _text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(0))

        assert status == "ok"
        assert clock.reads == 1

    def test_skips_the_retry_when_the_first_pass_was_cancelled_with_no_text(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info())])

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(0))

        assert (text_parts, status) == ([], "ok")
        assert len(model.calls) == 1

    def test_consults_the_cancel_callback_once_more_after_an_empty_loop(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info())])
        is_cancelled = CancelAfter(0)

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=is_cancelled)

        assert is_cancelled.invocations == 2

    def test_skips_the_retry_when_the_model_produced_nothing_and_cancellation_is_pending(self):
        model = FakeWhisperModel([([], make_info())])
        is_cancelled = CancelAfter(0)

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=is_cancelled)

        assert (text_parts, status) == ([], "ok")
        assert len(model.calls) == 1
        assert is_cancelled.invocations == 1

    def test_does_not_consult_the_cancel_callback_after_the_loop_when_text_was_collected(self):
        model = FakeWhisperModel([(numbered_segments(2), make_info())])
        is_cancelled = NeverCancels()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=is_cancelled)

        assert is_cancelled.invocations == 2

    def test_a_cancel_callback_that_never_fires_leaves_the_result_untouched(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info())])

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=NeverCancels())

        assert (text_parts, status) == (["part 0", "part 1", "part 2"], "ok")

    def test_the_cancel_flag_is_checked_once_per_segment(self):
        model = FakeWhisperModel([(numbered_segments(4), make_info())])
        is_cancelled = NeverCancels()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=is_cancelled)

        assert is_cancelled.invocations == 4


class TestTranscribeAudioFirstPassDeadline:
    def test_uses_the_thirty_second_floor_for_a_one_second_clip(self, clock):
        clock.step = 6.0
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=1.0))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (len(text_parts), status) == (3, "ok")

    def test_times_out_a_one_second_clip_once_thirty_seconds_elapse(self, clock):
        clock.step = 16.0
        model = FakeWhisperModel([(numbered_segments(4), make_info(duration=1.0))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["part 0"], "timed_out")

    def test_stops_pulling_segments_when_the_deadline_trips(self, clock):
        clock.step = 16.0
        model = FakeWhisperModel([(numbered_segments(5), make_info(duration=1.0))])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.iterators[FIRST_CALL].consumed == 2

    def test_uses_five_times_the_duration_for_a_hundred_second_clip(self, clock):
        clock.step = 100.0
        model = FakeWhisperModel([(numbered_segments(4), make_info(duration=100.0))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (len(text_parts), status) == (4, "ok")

    def test_times_out_a_hundred_second_clip_once_five_hundred_seconds_elapse(self, clock):
        clock.step = 200.0
        model = FakeWhisperModel([(numbered_segments(5), make_info(duration=100.0))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["part 0", "part 1"], "timed_out")

    @pytest.mark.parametrize("duration", [0.0, 1.0, 5.0, 6.0])
    def test_every_duration_up_to_six_seconds_clamps_to_the_thirty_second_floor(self, clock, duration):
        clock.step = 31.0
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=duration))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == ([], "timed_out")

    def test_a_duration_just_above_the_floor_extends_the_deadline(self, clock):
        clock.step = 31.0
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=7.0))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["part 0"], "timed_out")

    def test_does_not_time_out_when_the_elapsed_time_equals_the_deadline_exactly(self, clock):
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=1.0))])
        progress = ProgressRecorder(clock=clock, advance_seconds=30.0)

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert (text_parts, status) == (["part 0", "part 1"], "timed_out")

    def test_returns_the_first_pass_info_when_the_deadline_trips(self, clock):
        clock.step = 16.0
        expected_info = make_info(duration=1.0, language="de")
        model = FakeWhisperModel([(numbered_segments(4), expected_info)])

        _text_parts, info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert info is expected_info

    def test_skips_the_retry_when_the_first_pass_timed_out_before_collecting_text(self, clock):
        clock.step = 31.0
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=1.0))])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == ([], "timed_out")
        assert len(model.calls) == 1

    def test_an_empty_first_pass_never_reaches_the_deadline_check(self, clock):
        clock.step = 10_000.0
        model = FakeWhisperModel([([], make_info(duration=1.0)), ([FakeSegment("late")], make_info())])

        _text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert status == "retry_timed_out"
        assert len(model.calls) == 2


class TestTranscribeAudioEmptyResultRetry:
    def test_retries_when_the_first_pass_produced_no_segments(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert len(model.calls) == 2

    def test_returns_the_retry_text_with_an_ok_status(self):
        model = FakeWhisperModel([
            ([], make_info()),
            ([FakeSegment("  recovered  "), FakeSegment(" text ")], make_info()),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["recovered", "text"], "ok")

    def test_the_retry_disables_the_voice_activity_filter(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.calls[RETRY_CALL]["vad_filter"] is False

    def test_the_retry_sends_no_vad_parameters(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert "vad_parameters" not in model.calls[RETRY_CALL]

    def test_the_retry_stops_conditioning_on_previous_text(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.calls[RETRY_CALL]["condition_on_previous_text"] is False

    def test_the_retry_keeps_the_audio_path_language_and_beam_size(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, "pt")

        assert model.calls[RETRY_CALL]["audio"] == AUDIO_PATH
        assert model.calls[RETRY_CALL]["language"] == "pt"
        assert model.calls[RETRY_CALL]["beam_size"] == 5

    def test_the_retry_keyword_set_is_pinned(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert set(model.calls[RETRY_CALL]) == {
            "audio", "language", "beam_size", "vad_filter", "condition_on_previous_text"}

    def test_retries_when_the_cancel_callback_reports_no_cancellation(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])
        is_cancelled = NeverCancels()

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=is_cancelled)

        assert (text_parts, status) == (["recovered"], "ok")
        assert len(model.calls) == 2

    def test_returns_an_empty_result_when_the_retry_also_finds_nothing(self):
        model = FakeWhisperModel([([], make_info()), ([], make_info())])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == ([], "ok")
        assert len(model.calls) == 2

    def test_reports_progress_for_the_retry_segments(self):
        model = FakeWhisperModel([
            ([], make_info(duration=10.0)),
            ([FakeSegment("a", end=3.0), FakeSegment("b", end=8.0)], make_info(duration=10.0)),
        ])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(3.0, 10.0), (8.0, 10.0)]

    def test_retry_progress_uses_the_duration_of_the_first_pass_info(self):
        model = FakeWhisperModel([
            ([], make_info(duration=10.0)),
            ([FakeSegment("a", end=3.0)], make_info(duration=99.0)),
        ])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(3.0, 10.0)]

    def test_a_zero_duration_first_pass_suppresses_progress_during_the_retry(self):
        model = FakeWhisperModel([
            ([], make_info(duration=0.0)),
            ([FakeSegment("a", end=3.0)], make_info(duration=50.0)),
        ])
        progress = ProgressRecorder()

        text_parts, _info, _status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == []
        assert text_parts == ["a"]

    def test_the_retry_runs_without_a_progress_callback(self):
        model = FakeWhisperModel([([], make_info()), ([FakeSegment("recovered")], make_info())])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["recovered"], "ok")


class TestTranscribeAudioWhitespaceOnlyFirstPass:
    @pytest.mark.parametrize("raw_text", ["", " ", "   ", "\n", "\t", " \r\n "])
    def test_a_segment_that_strips_to_nothing_still_counts_as_a_result(self, raw_text):
        model = FakeWhisperModel([([FakeSegment(raw_text)], make_info())])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == ([""], "ok")

    @pytest.mark.parametrize("raw_text", ["", " ", "\n"])
    def test_a_segment_that_strips_to_nothing_suppresses_the_retry(self, raw_text):
        model = FakeWhisperModel([([FakeSegment(raw_text)], make_info())])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert len(model.calls) == 1

    def test_an_all_whitespace_transcript_is_returned_as_empty_strings(self):
        model = FakeWhisperModel([([FakeSegment("  "), FakeSegment("\t"), FakeSegment("")], make_info())])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["", "", ""], "ok")
        assert len(model.calls) == 1

    def test_a_whitespace_only_segment_still_reports_progress(self):
        model = FakeWhisperModel([([FakeSegment("   ", end=4.0)], make_info(duration=8.0))])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(4.0, 8.0)]


class TestTranscribeAudioRetryDeadline:
    def test_uses_the_fifteen_second_floor_for_a_one_second_clip(self, clock):
        clock.step = 7.0
        model = FakeWhisperModel([
            ([], make_info(duration=1.0)),
            (numbered_segments(2), make_info(duration=1.0)),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (len(text_parts), status) == (2, "ok")

    def test_times_out_the_retry_of_a_one_second_clip_after_fifteen_seconds(self, clock):
        clock.step = 8.0
        model = FakeWhisperModel([
            ([], make_info(duration=1.0)),
            (numbered_segments(3), make_info(duration=1.0)),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["part 0"], "retry_timed_out")

    def test_uses_three_times_the_duration_for_a_hundred_second_clip(self, clock):
        clock.step = 100.0
        model = FakeWhisperModel([
            ([], make_info(duration=100.0)),
            (numbered_segments(3), make_info(duration=100.0)),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (len(text_parts), status) == (3, "ok")

    def test_times_out_the_retry_of_a_hundred_second_clip_after_three_hundred_seconds(self, clock):
        clock.step = 200.0
        model = FakeWhisperModel([
            ([], make_info(duration=100.0)),
            (numbered_segments(3), make_info(duration=100.0)),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["part 0"], "retry_timed_out")

    def test_measures_the_retry_deadline_from_the_start_of_the_retry(self, clock):
        clock.step = 10.0
        model = FakeWhisperModel([
            ([], make_info(duration=1.0)),
            (numbered_segments(3), make_info(duration=1.0)),
        ])

        text_parts, _info, status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert (text_parts, status) == (["part 0"], "retry_timed_out")

    def test_stops_pulling_retry_segments_when_the_retry_deadline_trips(self, clock):
        clock.step = 8.0
        model = FakeWhisperModel([
            ([], make_info(duration=1.0)),
            (numbered_segments(5), make_info(duration=1.0)),
        ])

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert model.iterators[RETRY_CALL].consumed == 2

    def test_returns_the_retry_info_when_the_retry_times_out(self, clock):
        clock.step = 8.0
        retry_info = make_info(duration=1.0, language="fr")
        model = FakeWhisperModel([
            ([], make_info(duration=1.0, language="en")),
            (numbered_segments(3), retry_info),
        ])

        _text_parts, info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert info is retry_info

    def test_reports_progress_for_the_retry_segments_collected_before_the_deadline(self, clock):
        clock.step = 8.0
        model = FakeWhisperModel([
            ([], make_info(duration=1.0)),
            (numbered_segments(3), make_info(duration=1.0)),
        ])
        progress = ProgressRecorder()

        engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert progress.calls == [(1.0, 1.0)]

    def test_a_cancellation_during_the_retry_returns_ok_with_the_partial_text(self):
        model = FakeWhisperModel([([], make_info()), (numbered_segments(4), make_info())])

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(3))

        assert (text_parts, status) == (["part 0", "part 1"], "ok")
        assert model.iterators[RETRY_CALL].consumed == 3

    def test_the_cancel_flag_is_checked_before_the_retry_deadline(self, clock):
        clock.step = 1000.0
        model = FakeWhisperModel([
            ([], make_info(duration=1.0)),
            (numbered_segments(3), make_info(duration=1.0)),
        ])

        text_parts, _info, status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(1))

        assert (text_parts, status) == ([], "ok")
        assert clock.reads == 2


class TestTranscribeAudioReturnedInfo:
    def test_returns_the_first_pass_info_when_no_retry_is_needed(self):
        first_info = make_info(duration=30.0, language="en")
        model = FakeWhisperModel([([FakeSegment("text")], first_info)])

        _text_parts, info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert info is first_info

    def test_the_retry_info_replaces_the_first_pass_info(self):
        first_info = make_info(duration=30.0, language="en")
        retry_info = make_info(duration=30.0, language="fr")
        model = FakeWhisperModel([([], first_info), ([FakeSegment("bonjour")], retry_info)])

        _text_parts, info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert info is retry_info

    def test_the_retry_info_is_returned_even_when_the_retry_finds_nothing(self):
        first_info = make_info(language="en")
        retry_info = make_info(language="fr")
        model = FakeWhisperModel([([], first_info), ([], retry_info)])

        _text_parts, info, _status = engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert info is retry_info

    def test_the_first_pass_info_is_returned_when_the_first_pass_was_cancelled(self):
        first_info = make_info(language="en")
        model = FakeWhisperModel([(numbered_segments(3), first_info)])

        _text_parts, info, _status = engine.transcribe_audio(
            model, AUDIO_PATH, LANGUAGE_CODE, is_cancelled=CancelAfter(0))

        assert info is first_info


class TestTranscribeAudioFailures:
    def test_propagates_an_error_raised_by_the_first_transcribe_call(self):
        model = FakeWhisperModel([RuntimeError("model exploded")])

        with pytest.raises(RuntimeError, match="model exploded"):
            engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

    def test_propagates_an_error_raised_by_the_retry_transcribe_call(self):
        model = FakeWhisperModel([([], make_info()), ValueError("retry exploded")])

        with pytest.raises(ValueError, match="retry exploded"):
            engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)

        assert len(model.calls) == 2

    def test_propagates_an_error_raised_by_the_progress_callback(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info(duration=10.0))])
        progress = ExplodingCallback(OSError("progress exploded"))

        with pytest.raises(OSError, match="progress exploded"):
            engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE, on_progress=progress)

        assert model.iterators[FIRST_CALL].consumed == 1
        assert progress.invocations == 1

    def test_propagates_an_error_raised_by_the_cancel_callback(self):
        model = FakeWhisperModel([(numbered_segments(3), make_info())])

        with pytest.raises(KeyError, match="cancel exploded"):
            engine.transcribe_audio(
                model, AUDIO_PATH, LANGUAGE_CODE,
                is_cancelled=ExplodingCallback(KeyError("cancel exploded")))

        assert model.iterators[FIRST_CALL].consumed == 1

    def test_propagates_a_missing_duration_on_the_info_object(self):
        model = FakeWhisperModel([([FakeSegment("text")], object())])

        with pytest.raises(AttributeError):
            engine.transcribe_audio(model, AUDIO_PATH, LANGUAGE_CODE)
