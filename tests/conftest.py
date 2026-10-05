import os
import shutil
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

ENGLISH_TEXT = "The quick brown fox jumps over the lazy dog. Sotvox turns speech into text."
SPANISH_TEXT = "Hola, buenos días. Hoy vamos a hablar sobre el tiempo y la comida en Madrid."
LONG_ENGLISH_TEXT = " ".join([
    "This is the first part of the recording and it is spoken in English.",
    "We are testing the multilingual mode, which listens to each thirty second chunk on its own.",
    "The weather today is sunny and warm, and the children are playing in the park.",
    "After lunch we will go to the library to read some books about history and science.",
]) * 2
LONG_SPANISH_TEXT = " ".join([
    "Esta es la segunda parte de la grabación y está hablada en español.",
    "Estamos probando el modo multilingüe, que escucha cada fragmento de treinta segundos.",
    "El tiempo hoy es soleado y cálido, y los niños están jugando en el parque.",
    "Después del almuerzo vamos a ir a la biblioteca para leer libros de historia y ciencia.",
]) * 2

# (file name, container format, audio codec, sample rate)
AUDIO_FORMATS = [
    ("english.wav", "wav", "pcm_s16le", 16000),
    ("english.mp3", "mp3", "libmp3lame", 44100),
    ("english.m4a", "ipod", "aac", 44100),
    ("english.flac", "flac", "flac", 44100),
    ("english.ogg", "ogg", "libopus", 48000),
    ("english.wma", "asf", "wmav2", 44100),
]

requires_say = pytest.mark.skipif(shutil.which("say") is None, reason="needs the macOS 'say' command")


def _say(text, voice, aiff_path):
    subprocess.run(["say", "-v", voice, "-o", aiff_path, text], check=True)


def _read_audio(path):
    import av
    from av.audio.resampler import AudioResampler

    resampler = AudioResampler(format="s16", layout="mono", rate=16000)
    frames = []
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            frames.extend(resampler.resample(frame))
    frames.extend(resampler.resample(None))
    return frames


def _write_audio(frames, path, container_format, codec_name, rate, video=None):
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    with av.open(path, "w", format=container_format) as output:
        video_stream = None
        if video:
            video_stream = output.add_stream(video, rate=10)
            video_stream.width, video_stream.height = 160, 120
            video_stream.pix_fmt = "yuv420p"
        audio_stream = None
        if codec_name:
            audio_stream = output.add_stream(codec_name, rate=rate)
            audio_stream.layout = "mono"
            audio_stream.bit_rate = 64000
            sample_format = audio_stream.codec_context.codec.audio_formats[0].name
            resampler = AudioResampler(format=sample_format, layout="mono", rate=rate)
            for frame in frames:
                frame.pts = None
                for converted in resampler.resample(frame):
                    output.mux(audio_stream.encode(converted))
            for converted in resampler.resample(None):
                output.mux(audio_stream.encode(converted))
            output.mux(audio_stream.encode(None))
        if video_stream:
            duration = sum(frame.samples for frame in frames) / 16000 if frames else 3
            for index in range(int(duration * 10)):
                image = np.full((120, 160, 3), (index * 5) % 255, dtype=np.uint8)
                video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                output.mux(video_stream.encode(video_frame))
            output.mux(video_stream.encode(None))


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    if shutil.which("say") is None:
        pytest.skip("needs the macOS 'say' command")
    import av
    import numpy as np

    folder = tmp_path_factory.mktemp("media")
    english_aiff = str(folder / "english.aiff")
    spanish_aiff = str(folder / "spanish.aiff")
    _say(ENGLISH_TEXT, "Samantha", english_aiff)
    _say(SPANISH_TEXT, "Paulina", spanish_aiff)
    english = _read_audio(english_aiff)
    spanish = _read_audio(spanish_aiff)

    for name, container_format, codec_name, rate in AUDIO_FORMATS:
        _write_audio(english, str(folder / name), container_format, codec_name, rate)
    _write_audio(spanish, str(folder / "spanish.m4a"), "ipod", "aac", 44100)
    _write_audio(english, str(folder / "video.mp4"), "mp4", "aac", 44100, video="libx264")
    _write_audio(spanish, str(folder / "video.mkv"), "matroska", "libopus", 48000, video="libx264")
    _write_audio([], str(folder / "no_audio.mp4"), "mp4", None, 0, video="libx264")

    silence = av.AudioFrame.from_ndarray(np.zeros((1, 16000 * 3), dtype=np.int16), format="s16", layout="mono")
    silence.sample_rate = 16000
    _write_audio([silence], str(folder / "silence.wav"), "wav", "pcm_s16le", 16000)

    long_english_aiff = str(folder / "long_english.aiff")
    long_spanish_aiff = str(folder / "long_spanish.aiff")
    _say(LONG_ENGLISH_TEXT, "Samantha", long_english_aiff)
    _say(LONG_SPANISH_TEXT, "Paulina", long_spanish_aiff)
    _write_audio(_read_audio(long_english_aiff) + _read_audio(long_spanish_aiff),
                 str(folder / "mixed.wav"), "wav", "pcm_s16le", 16000)

    for leftover in folder.glob("*.aiff"):
        leftover.unlink()
    return folder


@pytest.fixture(scope="session")
def tiny_model():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="cpu", compute_type="int8")
