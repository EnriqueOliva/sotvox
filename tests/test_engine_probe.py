"""Tests for engine.probe_file and engine.has_audio_stream."""

import re
from fractions import Fraction

import pytest
from doubles import (
    FakeCodecContext,
    FakeContainer,
    FakeStream,
    fake_av_open,
    write_silent_wav,
    write_video_without_audio,
)

import engine

DURATION_PATTERN = re.compile(r"-?\d+\.\d{3}")

OPEN_FAILURES = [
    ValueError("unreadable header"),
    OSError("device disappeared"),
    FileNotFoundError("no such file"),
    PermissionError("locked by another process"),
    RuntimeError("decoder exploded"),
    MemoryError(),
    engine.av.FFmpegError(2, "ffmpeg refused the file"),
]


def failure_identifier(error):
    return type(error).__name__


class CodecContextWithoutNameAttribute:
    def __init__(self, width=None, height=None, sample_rate=None, channels=None):
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.channels = channels


class StreamRaisingOnType:
    def __init__(self, error):
        self.codec_context = FakeCodecContext()
        self.average_rate = None
        self.error = error

    @property
    def type(self):
        raise self.error


class StreamRefusingCodecContextAccess:
    def __init__(self, stream_type):
        self.type = stream_type
        self.average_rate = None

    @property
    def codec_context(self):
        raise AssertionError("has_audio_stream must not read the codec context")


class ExplodingStreamCollection:
    def __iter__(self):
        raise OSError("stream table is corrupt")


def stream_without_codec_name(stream_type, average_rate=None, **codec_context_fields):
    stream = FakeStream(stream_type, average_rate=average_rate)
    stream.codec_context = CodecContextWithoutNameAttribute(**codec_context_fields)
    return stream


def install_container(monkeypatch, container):
    opener = fake_av_open(container=container)
    monkeypatch.setattr(engine.av, "open", opener)
    return opener


def install_open_failure(monkeypatch, error):
    opener = fake_av_open(error=error)
    monkeypatch.setattr(engine.av, "open", opener)
    return opener


def probe_streams(monkeypatch, *streams):
    install_container(monkeypatch, FakeContainer(streams=list(streams)))

    return engine.probe_file("clip.mp4")["streams"]


class TestProbeFileDuration:
    @pytest.mark.parametrize(
        ("microseconds", "expected"),
        [
            (5_000_000, "5.000"),
            (1_000_000, "1.000"),
            (1_234_567, "1.235"),
            (500_000, "0.500"),
            (999, "0.001"),
            (1, "0.000"),
            (86_400_000_000, "86400.000"),
            (-1_500_000, "-1.500"),
        ],
    )
    def test_converts_microseconds_to_seconds_with_three_decimals(self, monkeypatch, microseconds, expected):
        install_container(monkeypatch, FakeContainer(duration=microseconds))

        result = engine.probe_file("clip.mp4")

        assert result["format"]["duration"] == expected

    @pytest.mark.parametrize("falsy_duration", [0, 0.0, None, False])
    def test_reports_unknown_when_the_container_duration_is_falsy(self, monkeypatch, falsy_duration):
        install_container(monkeypatch, FakeContainer(duration=falsy_duration))

        result = engine.probe_file("clip.mp4")

        assert result["format"]["duration"] == "unknown"

    def test_reports_other_format_fields_even_when_the_duration_is_unknown(self, monkeypatch):
        install_container(monkeypatch, FakeContainer(duration=None, bit_rate=64000))

        result = engine.probe_file("clip.mp4")

        assert result["format"]["bit_rate"] == 64000
        assert result["format"]["duration"] == "unknown"

    @pytest.mark.parametrize("duration", [7_500_000, 0])
    def test_duration_is_always_reported_as_a_string(self, monkeypatch, duration):
        install_container(monkeypatch, FakeContainer(duration=duration))

        result = engine.probe_file("clip.mp4")

        assert isinstance(result["format"]["duration"], str)

    def test_a_known_duration_always_matches_the_three_decimal_pattern(self, monkeypatch):
        install_container(monkeypatch, FakeContainer(duration=333_333))

        result = engine.probe_file("clip.mp4")

        assert DURATION_PATTERN.fullmatch(result["format"]["duration"])


class TestProbeFileFormatBlock:
    def test_returns_the_container_format_long_name(self, monkeypatch):
        install_container(monkeypatch, FakeContainer(format_long_name="Matroska / WebM"))

        result = engine.probe_file("clip.mkv")

        assert result["format"]["format_long_name"] == "Matroska / WebM"

    @pytest.mark.parametrize("bit_rate", [128000, 1, 0, None])
    def test_passes_the_container_bit_rate_through_unchanged(self, monkeypatch, bit_rate):
        install_container(monkeypatch, FakeContainer(bit_rate=bit_rate))

        result = engine.probe_file("clip.mp4")

        assert result["format"]["bit_rate"] == bit_rate

    def test_the_result_has_exactly_a_format_and_a_streams_key(self, monkeypatch):
        install_container(monkeypatch, FakeContainer())

        result = engine.probe_file("clip.mp4")

        assert set(result) == {"format", "streams"}

    def test_the_format_block_has_exactly_the_three_documented_keys(self, monkeypatch):
        install_container(monkeypatch, FakeContainer())

        result = engine.probe_file("clip.mp4")

        assert set(result["format"]) == {"format_long_name", "duration", "bit_rate"}


class TestProbeFileVideoStreams:
    def test_records_the_codec_type_and_codec_name(self, monkeypatch):
        stream = FakeStream("video", codec_name="h264", width=1920, height=1080)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["codec_type"] == "video"
        assert recorded["codec_name"] == "h264"

    def test_records_the_frame_dimensions(self, monkeypatch):
        stream = FakeStream("video", codec_name="h264", width=1920, height=1080)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["width"] == 1920
        assert recorded["height"] == 1080

    def test_records_missing_dimensions_as_none(self, monkeypatch):
        stream = FakeStream("video", codec_name="h264", width=None, height=None)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["width"] is None
        assert recorded["height"] is None

    @pytest.mark.parametrize(
        ("average_rate", "expected"),
        [
            (Fraction(30000, 1001), "29.97"),
            (Fraction(24, 1), "24.00"),
            (Fraction(1, 3), "0.33"),
            (25, "25.00"),
            (0.5, "0.50"),
            (23.976, "23.98"),
            (120, "120.00"),
        ],
    )
    def test_records_the_average_frame_rate_with_two_decimals(self, monkeypatch, average_rate, expected):
        stream = FakeStream("video", width=640, height=480, average_rate=average_rate)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["r_frame_rate"] == expected

    @pytest.mark.parametrize("falsy_rate", [None, 0, 0.0, Fraction(0, 1)])
    def test_omits_the_frame_rate_when_the_average_rate_is_falsy(self, monkeypatch, falsy_rate):
        stream = FakeStream("video", width=640, height=480, average_rate=falsy_rate)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert "r_frame_rate" not in recorded

    def test_a_video_stream_with_a_frame_rate_has_exactly_five_keys(self, monkeypatch):
        stream = FakeStream("video", width=640, height=480, average_rate=Fraction(30, 1))

        recorded = probe_streams(monkeypatch, stream)[0]

        assert set(recorded) == {"codec_type", "codec_name", "width", "height", "r_frame_rate"}

    def test_a_video_stream_without_a_frame_rate_has_exactly_four_keys(self, monkeypatch):
        stream = FakeStream("video", width=640, height=480)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert set(recorded) == {"codec_type", "codec_name", "width", "height"}

    def test_does_not_record_audio_fields_for_a_video_stream(self, monkeypatch):
        stream = FakeStream("video", width=640, height=480, sample_rate=48000, channels=2)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert "sample_rate" not in recorded
        assert "channels" not in recorded


class TestProbeFileAudioStreams:
    def test_records_the_codec_type_and_codec_name(self, monkeypatch):
        stream = FakeStream("audio", codec_name="mp3", sample_rate=44100, channels=2)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["codec_type"] == "audio"
        assert recorded["codec_name"] == "mp3"

    def test_records_the_sample_rate_and_channel_count(self, monkeypatch):
        stream = FakeStream("audio", codec_name="aac", sample_rate=48000, channels=6)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["sample_rate"] == 48000
        assert recorded["channels"] == 6

    def test_records_a_missing_sample_rate_and_channel_count_as_none(self, monkeypatch):
        stream = FakeStream("audio", codec_name="aac")

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["sample_rate"] is None
        assert recorded["channels"] is None

    def test_an_audio_stream_has_exactly_four_keys(self, monkeypatch):
        stream = FakeStream("audio", sample_rate=16000, channels=1)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert set(recorded) == {"codec_type", "codec_name", "sample_rate", "channels"}

    def test_does_not_record_video_fields_for_an_audio_stream_that_carries_them(self, monkeypatch):
        stream = FakeStream("audio", width=1920, height=1080, sample_rate=16000,
                            channels=1, average_rate=Fraction(30, 1))

        recorded = probe_streams(monkeypatch, stream)[0]

        assert set(recorded) == {"codec_type", "codec_name", "sample_rate", "channels"}


class TestProbeFileOtherStreamTypes:
    @pytest.mark.parametrize("stream_type", ["subtitle", "data", "attachment", "unknown", ""])
    def test_records_only_the_codec_type_and_codec_name(self, monkeypatch, stream_type):
        stream = FakeStream(stream_type, codec_name="mov_text")

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded == {"codec_type": stream_type, "codec_name": "mov_text"}

    def test_ignores_video_and_audio_fields_on_a_subtitle_stream(self, monkeypatch):
        stream = FakeStream("subtitle", codec_name="subrip", width=720, height=576,
                            sample_rate=48000, channels=2, average_rate=Fraction(25, 1))

        recorded = probe_streams(monkeypatch, stream)[0]

        assert set(recorded) == {"codec_type", "codec_name"}

    @pytest.mark.parametrize("stream_type", ["Audio", "AUDIO", "audio ", "Video"])
    def test_stream_type_matching_is_case_and_whitespace_sensitive(self, monkeypatch, stream_type):
        stream = FakeStream(stream_type, sample_rate=16000, channels=1, width=64, height=48)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert set(recorded) == {"codec_type", "codec_name"}


class TestProbeFileCodecNames:
    @pytest.mark.parametrize("stream_type", ["video", "audio", "subtitle"])
    def test_falls_back_to_unknown_when_the_codec_context_has_no_name_attribute(self, monkeypatch, stream_type):
        stream = stream_without_codec_name(stream_type)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["codec_name"] == "unknown"

    def test_still_records_video_fields_when_the_codec_name_is_missing(self, monkeypatch):
        stream = stream_without_codec_name("video", average_rate=Fraction(50, 1), width=320, height=240)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded == {
            "codec_type": "video",
            "codec_name": "unknown",
            "width": 320,
            "height": 240,
            "r_frame_rate": "50.00",
        }

    def test_still_records_audio_fields_when_the_codec_name_is_missing(self, monkeypatch):
        stream = stream_without_codec_name("audio", sample_rate=22050, channels=2)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded == {
            "codec_type": "audio",
            "codec_name": "unknown",
            "sample_rate": 22050,
            "channels": 2,
        }

    def test_a_codec_name_that_is_none_is_kept_rather_than_replaced_by_unknown(self, monkeypatch):
        stream = FakeStream("audio", codec_name=None, sample_rate=16000, channels=1)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["codec_name"] is None

    def test_an_empty_codec_name_is_kept_rather_than_replaced_by_unknown(self, monkeypatch):
        stream = FakeStream("audio", codec_name="", sample_rate=16000, channels=1)

        recorded = probe_streams(monkeypatch, stream)[0]

        assert recorded["codec_name"] == ""


class TestProbeFileStreamCollection:
    def test_returns_an_empty_stream_list_when_the_container_has_none(self, monkeypatch):
        install_container(monkeypatch, FakeContainer(streams=[]))

        result = engine.probe_file("clip.mp4")

        assert result["streams"] == []

    def test_preserves_the_order_of_the_container_streams(self, monkeypatch):
        video = FakeStream("video", codec_name="h264", width=1280, height=720)
        audio = FakeStream("audio", codec_name="aac", sample_rate=48000, channels=2)
        subtitle = FakeStream("subtitle", codec_name="mov_text")

        recorded = probe_streams(monkeypatch, video, audio, subtitle)

        assert [stream["codec_type"] for stream in recorded] == ["video", "audio", "subtitle"]

    def test_records_every_stream_including_repeated_types(self, monkeypatch):
        first_audio = FakeStream("audio", codec_name="aac", sample_rate=48000, channels=2)
        second_audio = FakeStream("audio", codec_name="ac3", sample_rate=44100, channels=6)

        recorded = probe_streams(monkeypatch, first_audio, second_audio)

        assert [stream["codec_name"] for stream in recorded] == ["aac", "ac3"]
        assert [stream["channels"] for stream in recorded] == [2, 6]

    def test_records_a_full_multiplexed_container(self, monkeypatch):
        video = FakeStream("video", codec_name="h264", width=1920, height=1080,
                           average_rate=Fraction(30000, 1001))
        audio = FakeStream("audio", codec_name="aac", sample_rate=48000, channels=2)
        data = FakeStream("data", codec_name="bin_data")

        recorded = probe_streams(monkeypatch, video, audio, data)

        assert recorded == [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "29.97",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": 48000,
                "channels": 2,
            },
            {"codec_type": "data", "codec_name": "bin_data"},
        ]


class TestProbeFileErrorHandling:
    @pytest.mark.parametrize("error", OPEN_FAILURES, ids=failure_identifier)
    def test_returns_none_when_opening_the_container_raises(self, monkeypatch, error):
        install_open_failure(monkeypatch, error)

        assert engine.probe_file("clip.mp4") is None

    def test_returns_none_when_reading_a_stream_type_raises(self, monkeypatch):
        container = FakeContainer(streams=[StreamRaisingOnType(RuntimeError("stream is gone"))])
        install_container(monkeypatch, container)

        assert engine.probe_file("clip.mp4") is None

    def test_returns_none_when_iterating_the_stream_collection_raises(self, monkeypatch):
        container = FakeContainer()
        container.streams = ExplodingStreamCollection()
        install_container(monkeypatch, container)

        assert engine.probe_file("clip.mp4") is None

    def test_returns_none_when_formatting_the_frame_rate_raises(self, monkeypatch):
        stream = FakeStream("video", width=640, height=480, average_rate=object())
        install_container(monkeypatch, FakeContainer(streams=[stream]))

        assert engine.probe_file("clip.mp4") is None

    def test_returns_none_when_the_container_has_no_format(self, monkeypatch):
        container = FakeContainer()
        container.format = None
        install_container(monkeypatch, container)

        assert engine.probe_file("clip.mp4") is None

    def test_returns_none_when_the_duration_cannot_be_divided(self, monkeypatch):
        install_container(monkeypatch, FakeContainer(duration="not a number"))

        assert engine.probe_file("clip.mp4") is None

    def test_closes_the_container_even_when_a_stream_raises(self, monkeypatch):
        container = FakeContainer(streams=[StreamRaisingOnType(RuntimeError("stream is gone"))])
        install_container(monkeypatch, container)

        engine.probe_file("clip.mp4")

        assert container.closed is True

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(1)], ids=failure_identifier)
    def test_lets_base_exceptions_propagate_instead_of_returning_none(self, monkeypatch, error):
        install_open_failure(monkeypatch, error)

        with pytest.raises(type(error)):
            engine.probe_file("clip.mp4")


class TestProbeFileContainerLifecycle:
    def test_closes_the_container_on_the_happy_path(self, monkeypatch):
        container = FakeContainer(streams=[FakeStream("audio", sample_rate=16000, channels=1)])
        install_container(monkeypatch, container)

        engine.probe_file("clip.mp4")

        assert container.closed is True

    def test_opens_the_container_exactly_once_with_the_given_path(self, monkeypatch):
        opener = install_container(monkeypatch, FakeContainer())

        engine.probe_file(r"C:\media\my clip.mp4")

        assert opener.calls == [r"C:\media\my clip.mp4"]

    def test_attempts_the_open_once_and_does_not_retry_when_it_fails(self, monkeypatch):
        opener = install_open_failure(monkeypatch, OSError("cannot open"))

        result = engine.probe_file("clip.mp4")

        assert result is None
        assert opener.calls == ["clip.mp4"]


class TestHasAudioStream:
    def test_returns_true_when_the_container_has_an_audio_stream(self, monkeypatch):
        container = FakeContainer(streams=[FakeStream("audio", sample_rate=16000, channels=1)])
        install_container(monkeypatch, container)

        assert engine.has_audio_stream("clip.mp4") is True

    def test_returns_true_when_the_audio_stream_comes_after_other_streams(self, monkeypatch):
        streams = [
            FakeStream("video", width=640, height=480),
            FakeStream("subtitle", codec_name="subrip"),
            FakeStream("audio", sample_rate=48000, channels=2),
        ]
        install_container(monkeypatch, FakeContainer(streams=streams))

        assert engine.has_audio_stream("clip.mp4") is True

    def test_returns_false_when_the_container_has_no_streams_at_all(self, monkeypatch):
        install_container(monkeypatch, FakeContainer(streams=[]))

        assert engine.has_audio_stream("clip.mp4") is False

    @pytest.mark.parametrize("stream_type", ["video", "subtitle", "data", "attachment", "Audio", "AUDIO", "audio "])
    def test_returns_false_for_a_container_whose_only_stream_is_not_audio(self, monkeypatch, stream_type):
        install_container(monkeypatch, FakeContainer(streams=[FakeStream(stream_type)]))

        assert engine.has_audio_stream("clip.mp4") is False

    def test_returns_false_for_a_video_only_container(self, monkeypatch):
        streams = [
            FakeStream("video", codec_name="h264", width=1920, height=1080),
            FakeStream("data", codec_name="bin_data"),
        ]
        install_container(monkeypatch, FakeContainer(streams=streams))

        assert engine.has_audio_stream("clip.mp4") is False

    def test_stops_inspecting_streams_at_the_first_audio_stream(self, monkeypatch):
        streams = [
            FakeStream("audio", sample_rate=16000, channels=1),
            StreamRaisingOnType(RuntimeError("this stream must never be read")),
        ]
        install_container(monkeypatch, FakeContainer(streams=streams))

        assert engine.has_audio_stream("clip.mp4") is True

    def test_does_not_read_the_codec_context_of_a_stream(self, monkeypatch):
        container = FakeContainer(streams=[StreamRefusingCodecContextAccess("audio")])
        install_container(monkeypatch, container)

        assert engine.has_audio_stream("clip.mp4") is True

    def test_ignores_the_container_duration_and_format(self, monkeypatch):
        container = FakeContainer(duration=None, streams=[FakeStream("audio", sample_rate=8000, channels=1)])
        container.format = None
        install_container(monkeypatch, container)

        assert engine.has_audio_stream("clip.mp4") is True

    @pytest.mark.parametrize("error", OPEN_FAILURES, ids=failure_identifier)
    def test_returns_false_when_opening_the_container_raises(self, monkeypatch, error):
        install_open_failure(monkeypatch, error)

        assert engine.has_audio_stream("clip.mp4") is False

    def test_returns_false_when_reading_a_stream_type_raises(self, monkeypatch):
        container = FakeContainer(streams=[StreamRaisingOnType(ValueError("broken stream"))])
        install_container(monkeypatch, container)

        assert engine.has_audio_stream("clip.mp4") is False

    def test_returns_false_when_iterating_the_stream_collection_raises(self, monkeypatch):
        container = FakeContainer()
        container.streams = ExplodingStreamCollection()
        install_container(monkeypatch, container)

        assert engine.has_audio_stream("clip.mp4") is False

    def test_closes_the_container_when_an_audio_stream_is_found(self, monkeypatch):
        container = FakeContainer(streams=[FakeStream("audio", sample_rate=16000, channels=1)])
        install_container(monkeypatch, container)

        engine.has_audio_stream("clip.mp4")

        assert container.closed is True

    def test_closes_the_container_when_no_audio_stream_is_found(self, monkeypatch):
        container = FakeContainer(streams=[FakeStream("video", width=64, height=48)])
        install_container(monkeypatch, container)

        engine.has_audio_stream("clip.mp4")

        assert container.closed is True

    def test_closes_the_container_when_a_stream_raises(self, monkeypatch):
        container = FakeContainer(streams=[StreamRaisingOnType(ValueError("broken stream"))])
        install_container(monkeypatch, container)

        engine.has_audio_stream("clip.mp4")

        assert container.closed is True

    def test_opens_the_container_exactly_once_with_the_given_path(self, monkeypatch):
        opener = install_container(monkeypatch, FakeContainer())

        engine.has_audio_stream(r"C:\media\my clip.mkv")

        assert opener.calls == [r"C:\media\my clip.mkv"]

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(1)], ids=failure_identifier)
    def test_lets_base_exceptions_propagate_instead_of_returning_false(self, monkeypatch, error):
        install_open_failure(monkeypatch, error)

        with pytest.raises(type(error)):
            engine.has_audio_stream("clip.mp4")


class TestProbeFileAgainstRealAudio:
    @pytest.mark.integration
    def test_describes_a_real_wav_file(self, tmp_path):
        wav_path = write_silent_wav(tmp_path / "silence.wav")

        result = engine.probe_file(str(wav_path))

        assert "WAV" in result["format"]["format_long_name"].upper()
        assert len(result["streams"]) == 1
        assert result["streams"][0]["codec_type"] == "audio"

    @pytest.mark.integration
    def test_reports_the_codec_name_of_a_real_wav_file(self, tmp_path):
        wav_path = write_silent_wav(tmp_path / "silence.wav")

        result = engine.probe_file(str(wav_path))

        assert result["streams"][0]["codec_name"] == "pcm_s16le"

    @pytest.mark.integration
    @pytest.mark.parametrize(("sample_rate", "channels"), [(16000, 1), (8000, 2), (44100, 2)])
    def test_reports_the_sample_rate_and_channels_of_a_real_wav_file(self, tmp_path, sample_rate, channels):
        wav_path = write_silent_wav(tmp_path / "silence.wav", sample_rate=sample_rate, channels=channels)

        recorded = engine.probe_file(str(wav_path))["streams"][0]

        assert recorded["sample_rate"] == sample_rate
        assert recorded["channels"] == channels

    @pytest.mark.integration
    @pytest.mark.parametrize("seconds", [0.25, 0.5, 1.0])
    def test_reports_the_real_duration_of_a_wav_file(self, tmp_path, seconds):
        wav_path = write_silent_wav(tmp_path / "silence.wav", seconds=seconds)

        duration = engine.probe_file(str(wav_path))["format"]["duration"]

        assert DURATION_PATTERN.fullmatch(duration)
        assert float(duration) == pytest.approx(seconds, abs=0.01)

    @pytest.mark.integration
    def test_reports_a_positive_bit_rate_for_a_real_wav_file(self, tmp_path):
        wav_path = write_silent_wav(tmp_path / "silence.wav")

        result = engine.probe_file(str(wav_path))

        assert result["format"]["bit_rate"] > 0

    @pytest.mark.integration
    def test_has_audio_stream_is_true_for_a_real_wav_file(self, tmp_path):
        wav_path = write_silent_wav(tmp_path / "silence.wav")

        assert engine.has_audio_stream(str(wav_path)) is True


class TestProbeFileAgainstRealVideo:
    @pytest.mark.integration
    def test_describes_a_real_video_without_an_audio_track(self, tmp_path):
        video_path = write_video_without_audio(tmp_path / "clip.mp4")
        if video_path is None:
            pytest.skip("no usable video encoder in this PyAV build")

        result = engine.probe_file(str(video_path))

        assert [stream["codec_type"] for stream in result["streams"]] == ["video"]
        assert result["streams"][0]["width"] == 64
        assert result["streams"][0]["height"] == 48

    @pytest.mark.integration
    def test_reports_the_frame_rate_of_a_real_video(self, tmp_path):
        video_path = write_video_without_audio(tmp_path / "clip.mp4", fps=10)
        if video_path is None:
            pytest.skip("no usable video encoder in this PyAV build")

        recorded = engine.probe_file(str(video_path))["streams"][0]

        assert recorded["r_frame_rate"] == "10.00"

    @pytest.mark.integration
    def test_has_audio_stream_is_false_for_a_real_video_without_an_audio_track(self, tmp_path):
        video_path = write_video_without_audio(tmp_path / "clip.mp4")
        if video_path is None:
            pytest.skip("no usable video encoder in this PyAV build")

        assert engine.has_audio_stream(str(video_path)) is False


class TestRealFilesThatAreNotMedia:
    @pytest.mark.integration
    def test_probe_file_returns_none_for_a_path_that_does_not_exist(self, tmp_path):
        missing_path = tmp_path / "absent.wav"

        assert engine.probe_file(str(missing_path)) is None

    @pytest.mark.integration
    def test_has_audio_stream_returns_false_for_a_path_that_does_not_exist(self, tmp_path):
        missing_path = tmp_path / "absent.wav"

        assert engine.has_audio_stream(str(missing_path)) is False

    @pytest.mark.integration
    def test_probe_file_returns_none_for_a_plain_text_file(self, tmp_path):
        text_path = tmp_path / "notes.txt"
        text_path.write_text("this is not media at all", encoding="utf-8")

        assert engine.probe_file(str(text_path)) is None

    @pytest.mark.integration
    def test_has_audio_stream_returns_false_for_a_plain_text_file(self, tmp_path):
        text_path = tmp_path / "notes.txt"
        text_path.write_text("this is not media at all", encoding="utf-8")

        assert engine.has_audio_stream(str(text_path)) is False

    @pytest.mark.integration
    def test_a_media_extension_does_not_make_a_text_file_probeable(self, tmp_path):
        disguised_path = tmp_path / "pretend.wav"
        disguised_path.write_text("RIFF is nowhere to be found here", encoding="utf-8")

        assert engine.probe_file(str(disguised_path)) is None
        assert engine.has_audio_stream(str(disguised_path)) is False

    @pytest.mark.integration
    def test_probe_file_returns_none_for_an_empty_file(self, tmp_path):
        empty_path = tmp_path / "empty.wav"
        empty_path.write_bytes(b"")

        assert engine.probe_file(str(empty_path)) is None

    @pytest.mark.integration
    def test_probe_file_returns_none_for_a_truncated_wav_file(self, tmp_path):
        truncated_path = tmp_path / "truncated.wav"
        truncated_path.write_bytes(b"RIFF")

        assert engine.probe_file(str(truncated_path)) is None
        assert engine.has_audio_stream(str(truncated_path)) is False

    @pytest.mark.integration
    def test_probe_file_returns_none_for_a_directory(self, tmp_path):
        directory = tmp_path / "a_folder"
        directory.mkdir()

        assert engine.probe_file(str(directory)) is None
        assert engine.has_audio_stream(str(directory)) is False
