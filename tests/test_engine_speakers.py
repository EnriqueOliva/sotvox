"""Unit tests for the per-speaker track transcription in engine."""

import types

import av
import numpy as np
import pytest
from doubles import FakeSegment, FakeWhisperModel, make_info

import engine

SAMPLE_RATE = 16000
PCM_PEAK = 32767
LOUD_PEAK = 0.5
QUIET_PEAK = 0.05


def write_tracks(path, container_format, codec, tracks):
    container = av.open(str(path), mode="w", format=container_format)
    streams = [container.add_stream(codec, rate=SAMPLE_RATE, layout="mono") for _ in tracks]
    for stream, samples in zip(streams, tracks, strict=True):
        pcm = (np.asarray(samples) * PCM_PEAK).astype(np.int16).reshape(1, -1)
        frame_size = stream.codec_context.frame_size or pcm.shape[1]
        for offset in range(0, pcm.shape[1], frame_size):
            chunk = pcm[:, offset:offset + frame_size]
            if codec == "aac":
                frame = av.AudioFrame.from_ndarray(chunk.astype(np.float32) / PCM_PEAK, format="fltp", layout="mono")
            else:
                frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = offset
            for packet in stream.encode(frame):
                container.mux(packet)
    for stream in streams:
        for packet in stream.encode(None):
            container.mux(packet)
    container.close()


def tone(peak, frequency, seconds=1.0):
    timeline = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return peak * np.sin(2 * np.pi * frequency * timeline)


def utterance(start, end, label, text, confidence=-0.3):
    return engine.Utterance(start, end, label, text, confidence)


class TestParseSpeakerTracks:
    def test_reads_track_numbers_and_labels_sorted_by_track(self):
        assert engine.parse_speaker_tracks({"3": "Otros", "2": "Enrique"}) == [(2, "Enrique"), (3, "Otros")]

    def test_strips_the_labels(self):
        assert engine.parse_speaker_tracks({"2": "  Enrique "}) == [(2, "Enrique")]

    @pytest.mark.parametrize("raw_tracks", [None, [], "2=Enrique", 7])
    def test_anything_but_a_mapping_means_no_speaker_tracks(self, raw_tracks):
        assert engine.parse_speaker_tracks(raw_tracks) == []

    @pytest.mark.parametrize("raw_number", ["two", "", "0", "-1", None])
    def test_skips_a_track_number_that_is_not_a_positive_integer(self, raw_number):
        assert engine.parse_speaker_tracks({raw_number: "Enrique", "3": "Otros"}) == [(3, "Otros")]

    @pytest.mark.parametrize("raw_label", ["", "   ", None, 5])
    def test_skips_a_label_that_is_not_a_non_empty_text(self, raw_label):
        assert engine.parse_speaker_tracks({"2": raw_label, "3": "Otros"}) == [(3, "Otros")]

    def test_track_one_is_a_valid_track(self):
        assert engine.parse_speaker_tracks({"1": "Mix"}) == [(1, "Mix")]


class TestSpeakerTracksFor:
    SPEAKER_TRACKS = [(2, "Enrique"), (3, "Otros")]

    def test_returns_the_tracks_when_the_file_has_every_one_of_them(self, monkeypatch):
        monkeypatch.setattr(engine, "audio_stream_count", lambda filepath: 3)

        assert engine.speaker_tracks_for("meeting.mp4", self.SPEAKER_TRACKS) == self.SPEAKER_TRACKS

    def test_returns_nothing_when_the_file_lacks_the_highest_track(self, monkeypatch):
        monkeypatch.setattr(engine, "audio_stream_count", lambda filepath: 2)

        assert engine.speaker_tracks_for("meeting.mp4", self.SPEAKER_TRACKS) == []

    def test_returns_nothing_for_a_single_track_recording(self, monkeypatch):
        monkeypatch.setattr(engine, "audio_stream_count", lambda filepath: 1)

        assert engine.speaker_tracks_for("meeting.mp4", self.SPEAKER_TRACKS) == []

    def test_does_not_open_the_file_when_no_speaker_tracks_are_configured(self, monkeypatch):
        opened = []
        monkeypatch.setattr(engine, "audio_stream_count", lambda filepath: opened.append(filepath) or 3)

        assert engine.speaker_tracks_for("meeting.mp4", []) == []
        assert opened == []


class TestAudioStreamDecoding:
    @pytest.mark.parametrize("container_format, codec, name", [
        ("matroska", "pcm_s16le", "two.mka"),
        ("mp4", "aac", "two.mp4"),
    ])
    def test_counts_and_decodes_each_audio_track_separately(self, tmp_path, container_format, codec, name):
        path = tmp_path / name
        write_tracks(path, container_format, codec, [tone(LOUD_PEAK, 440), tone(QUIET_PEAK, 220)])

        first = engine.decode_audio_stream(str(path), 0)
        second = engine.decode_audio_stream(str(path), 1)

        assert engine.audio_stream_count(str(path)) == 2
        assert float(np.max(np.abs(first))) == pytest.approx(LOUD_PEAK, rel=0.2)
        assert float(np.max(np.abs(second))) == pytest.approx(QUIET_PEAK, rel=0.2)

    def test_decodes_at_sixteen_kilohertz_mono_float_samples(self, tmp_path):
        path = tmp_path / "two.mka"
        write_tracks(path, "matroska", "pcm_s16le", [tone(LOUD_PEAK, 440, seconds=2.0), tone(QUIET_PEAK, 220)])

        samples = engine.decode_audio_stream(str(path), 0)

        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert len(samples) == 2 * SAMPLE_RATE

    def test_a_file_that_cannot_be_opened_has_no_audio_streams(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not media")

        assert engine.audio_stream_count(str(broken)) == 0


class TestWithoutEchoes:
    def test_drops_the_weaker_copy_of_a_sentence_heard_on_two_tracks(self):
        spoken = utterance(10.0, 14.0, "Otros", "revisá la factura del cliente", confidence=-0.2)
        echo = utterance(10.3, 14.2, "Enrique", "revisá la factura del cliente", confidence=-0.9)

        assert engine.without_echoes([echo, spoken]) == [spoken]

    def test_keeps_both_when_they_say_different_things_at_the_same_time(self):
        first = utterance(10.0, 14.0, "Otros", "revisá la factura del cliente", confidence=-0.2)
        second = utterance(10.5, 13.0, "Enrique", "dale ya la miro", confidence=-0.9)

        assert engine.without_echoes([first, second]) == [first, second]

    def test_keeps_both_when_the_same_words_are_said_at_different_times(self):
        first = utterance(10.0, 12.0, "Otros", "perfecto gracias", confidence=-0.2)
        later = utterance(40.0, 42.0, "Enrique", "perfecto gracias", confidence=-0.9)

        assert engine.without_echoes([first, later]) == [first, later]

    def test_keeps_two_similar_lines_from_the_same_speaker(self):
        first = utterance(10.0, 12.0, "Otros", "sí sí sí", confidence=-0.2)
        second = utterance(10.5, 12.5, "Otros", "sí sí sí", confidence=-0.9)

        assert engine.without_echoes([first, second]) == [first, second]

    def test_keeps_both_copies_when_their_confidence_is_equal(self):
        first = utterance(10.0, 12.0, "Otros", "perfecto gracias", confidence=-0.5)
        second = utterance(10.0, 12.0, "Enrique", "perfecto gracias", confidence=-0.5)

        assert engine.without_echoes([first, second]) == [first, second]

    def test_an_overlap_below_half_of_the_shorter_line_is_not_an_echo(self):
        first = utterance(10.0, 14.0, "Otros", "perfecto gracias", confidence=-0.2)
        second = utterance(13.0, 17.0, "Enrique", "perfecto gracias", confidence=-0.9)

        assert engine.without_echoes([first, second]) == [first, second]

    def test_returns_the_kept_lines_in_start_order(self):
        later = utterance(20.0, 21.0, "Otros", "segundo")
        earlier = utterance(5.0, 6.0, "Enrique", "primero")

        assert engine.without_echoes([later, earlier]) == [earlier, later]


class TestFormatTimestamp:
    @pytest.mark.parametrize("seconds, expected", [
        (0, "00:00"), (5.9, "00:05"), (65, "01:05"), (3599, "59:59"), (3600, "1:00:00"), (3725, "1:02:05"),
    ])
    def test_shows_minutes_and_seconds_and_hours_only_past_one_hour(self, seconds, expected):
        assert engine.format_timestamp(seconds) == expected


class TestConversationLines:
    def test_labels_each_turn_with_its_start_time_and_speaker(self):
        lines = engine.conversation_lines([
            utterance(3.0, 5.0, "Otros", "¿Revisaste la factura?"),
            utterance(6.0, 7.0, "Enrique", "Sí, quedó paga."),
        ])

        assert lines == ["[00:03] Otros: ¿Revisaste la factura?", "[00:06] Enrique: Sí, quedó paga."]

    def test_joins_consecutive_lines_of_the_same_speaker(self):
        lines = engine.conversation_lines([
            utterance(3.0, 5.0, "Otros", "Primero esto."),
            utterance(5.5, 8.0, "Otros", "Después lo otro."),
        ])

        assert lines == ["[00:03] Otros: Primero esto. Después lo otro."]

    def test_starts_a_new_paragraph_after_a_minute_of_the_same_speaker(self):
        lines = engine.conversation_lines([
            utterance(0.0, 30.0, "Otros", "uno"),
            utterance(30.0, 59.0, "Otros", "dos"),
            utterance(60.0, 62.0, "Otros", "tres"),
        ])

        assert lines == ["[00:00] Otros: uno dos", "[01:00] Otros: tres"]

    def test_orders_the_turns_by_start_time(self):
        lines = engine.conversation_lines([
            utterance(9.0, 10.0, "Enrique", "después"),
            utterance(1.0, 2.0, "Otros", "antes"),
        ])

        assert lines == ["[00:01] Otros: antes", "[00:09] Enrique: después"]

    def test_no_utterances_means_no_lines(self):
        assert engine.conversation_lines([]) == []


class RecordingStreamDecoder:
    def __init__(self):
        self.calls = []

    def __call__(self, filepath, stream_position):
        self.calls.append((filepath, stream_position))
        return np.full(4, stream_position + 1, dtype=np.float32)


@pytest.fixture
def stream_decoder(monkeypatch):
    decoder = RecordingStreamDecoder()
    monkeypatch.setattr(engine, "decode_audio_stream", decoder)
    return decoder


class TestTranscribeSpeakerTracks:
    SPEAKER_TRACKS = [(2, "Enrique"), (3, "Otros")]

    def two_track_model(self):
        return FakeWhisperModel([
            ([FakeSegment("Sí, quedó paga.", start=6.0, end=7.0)], make_info(duration=10.0, language="es")),
            ([FakeSegment("¿Revisaste la factura?", start=3.0, end=5.0)], make_info(duration=10.0, language="es")),
        ])

    def test_interleaves_the_tracks_as_labelled_turns(self, stream_decoder):
        lines, _info, status = engine.transcribe_speaker_tracks(
            self.two_track_model(), "meeting.mp4", "es", self.SPEAKER_TRACKS)

        assert lines == ["[00:03] Otros: ¿Revisaste la factura?", "[00:06] Enrique: Sí, quedó paga."]
        assert status == "ok"

    def test_decodes_track_n_as_audio_stream_n_minus_one(self, stream_decoder):
        engine.transcribe_speaker_tracks(self.two_track_model(), "meeting.mp4", "es", self.SPEAKER_TRACKS)

        assert stream_decoder.calls == [("meeting.mp4", 1), ("meeting.mp4", 2)]

    def test_transcribes_each_decoded_track_with_the_language(self, stream_decoder):
        model = self.two_track_model()

        engine.transcribe_speaker_tracks(model, "meeting.mp4", "es", self.SPEAKER_TRACKS)

        assert [float(call["audio"][0]) for call in model.calls] == [2.0, 3.0]
        assert [call["language"] for call in model.calls] == ["es", "es"]

    def test_reports_the_combined_duration_and_languages(self, stream_decoder):
        model = FakeWhisperModel([
            ([FakeSegment("hola", start=0.0, end=1.0)], make_info(duration=10.0, language="es", duration_after_vad=4.0)),
            ([FakeSegment("hi", start=2.0, end=3.0)], make_info(duration=12.0, language="en", duration_after_vad=5.0)),
        ])

        _lines, info, _status = engine.transcribe_speaker_tracks(model, "meeting.mp4", None, self.SPEAKER_TRACKS)

        assert info.duration == 12.0
        assert info.duration_after_vad == 9.0
        assert info.language == "en+es"

    def test_reports_overall_progress_across_the_tracks(self, stream_decoder):
        progress = []

        engine.transcribe_speaker_tracks(
            self.two_track_model(), "meeting.mp4", "es", self.SPEAKER_TRACKS,
            on_progress=lambda position, duration: progress.append(position / duration))

        assert progress == [pytest.approx(0.35), pytest.approx(0.75)]

    def test_stops_before_the_next_track_once_cancelled(self, stream_decoder):
        checks = []

        def cancel_after_first_track():
            checks.append(True)
            return len(stream_decoder.calls) >= 1 and len(checks) > 2

        lines, _info, _status = engine.transcribe_speaker_tracks(
            self.two_track_model(), "meeting.mp4", "es", self.SPEAKER_TRACKS,
            is_cancelled=cancel_after_first_track)

        assert stream_decoder.calls == [("meeting.mp4", 1)]
        assert lines == ["[00:06] Enrique: Sí, quedó paga."]

    def test_passes_a_track_timeout_through_as_the_status(self, stream_decoder, fake_clock, monkeypatch):
        monkeypatch.setattr(engine, "time", types.SimpleNamespace(time=fake_clock))
        model = FakeWhisperModel([
            ([FakeSegment("uno", start=0.0, end=1.0)], make_info(duration=1.0)),
            ([FakeSegment("dos", start=0.0, end=1.0)], make_info(duration=1.0)),
        ])
        original = model.transcribe

        def slow_second_track(audio, **kwargs):
            result = original(audio, **kwargs)
            if len(model.calls) == 2:
                fake_clock.advance(1000)
            return result

        model.transcribe = slow_second_track

        _lines, _info, status = engine.transcribe_speaker_tracks(model, "meeting.mp4", "es", self.SPEAKER_TRACKS)

        assert status == "timed_out"

    def test_drops_a_line_the_microphone_only_heard_through_the_speakers(self, stream_decoder):
        model = FakeWhisperModel([
            ([FakeSegment("revisá la factura del cliente", start=3.2, end=5.0)], make_info(duration=10.0)),
            ([FakeSegment("revisá la factura del cliente", start=3.0, end=5.0)], make_info(duration=10.0)),
        ])
        model_segments = [response[0][0] for response in model.responses]
        model_segments[0].avg_logprob = -0.9
        model_segments[1].avg_logprob = -0.2

        lines, _info, _status = engine.transcribe_speaker_tracks(model, "meeting.mp4", "es", self.SPEAKER_TRACKS)

        assert lines == ["[00:03] Otros: revisá la factura del cliente"]

    def test_a_track_with_no_speech_leaves_only_the_other_speaker(self, stream_decoder):
        model = FakeWhisperModel([
            ([], make_info(duration=10.0)),
            ([], make_info(duration=10.0)),
            ([FakeSegment("¿Revisaste la factura?", start=3.0, end=5.0)], make_info(duration=10.0)),
        ])

        lines, _info, _status = engine.transcribe_speaker_tracks(model, "meeting.mp4", "es", self.SPEAKER_TRACKS)

        assert lines == ["[00:03] Otros: ¿Revisaste la factura?"]
