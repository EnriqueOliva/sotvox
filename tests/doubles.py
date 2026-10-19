"""Test doubles shared by the Sotvox suite.

The Tk doubles let SotvoxApp methods run against a real instance whose
constructor was skipped, so the production methods under test are the real
ones. tests/test_app_contract.py pins the double surface against a genuinely
constructed app so the two cannot drift apart silently.
"""

import json
import wave
import zipfile


class FakeSegment:
    def __init__(self, text, start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end


def make_info(duration=10.0, language="en", language_probability=0.99,
              duration_after_vad=None, **extra):
    info = FakeInfo(duration, language, language_probability, **extra)
    if duration_after_vad is not None:
        info.duration_after_vad = duration_after_vad
    return info


class FakeInfo:
    def __init__(self, duration=10.0, language="en", language_probability=0.99, **extra):
        self.duration = duration
        self.language = language
        self.language_probability = language_probability
        for key, value in extra.items():
            setattr(self, key, value)


class TrackingIterator:
    def __init__(self, items):
        self._items = list(items)
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed >= len(self._items):
            raise StopIteration
        item = self._items[self.consumed]
        self.consumed += 1
        return item


class FakeWhisperModel:
    def __init__(self, responses=None, repeat_last=False):
        self.responses = list(responses or [])
        self.repeat_last = repeat_last
        self.calls = []
        self.iterators = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        index = len(self.calls) - 1
        if index >= len(self.responses):
            if not self.responses:
                raise AssertionError("FakeWhisperModel has no configured responses")
            if not self.repeat_last:
                raise AssertionError(
                    f"unexpected transcribe call #{index + 1}; only "
                    f"{len(self.responses)} response(s) configured")
            index = len(self.responses) - 1
        response = self.responses[index]
        if isinstance(response, BaseException):
            raise response
        segments, info = response
        iterator = TrackingIterator(segments)
        self.iterators.append(iterator)
        return iterator, info

    @property
    def last_call(self):
        return self.calls[-1]


class FakeClock:
    def __init__(self, start=1000.0, step=0.0):
        self.now = start
        self.step = step
        self.reads = 0

    def __call__(self):
        self.reads += 1
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds):
        self.now += seconds


class FakeVar:
    def __init__(self, value=""):
        self._value = value
        self.traces = []
        self.set_history = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        self.set_history.append(value)
        for callback in list(self.traces):
            callback("varname", "", "write")

    def trace_add(self, mode, callback):
        self.traces.append(callback)
        return f"trace#{len(self.traces)}"


class FakeWidget:
    def __init__(self, **initial):
        self.state = dict(initial)
        self.config_history = []
        self.packed = True
        self.pack_calls = []
        self.pack_forget_count = 0
        self.destroyed = False
        self.bindings = {}

    def configure(self, **kwargs):
        self.state.update(kwargs)
        self.config_history.append(kwargs)

    config = configure

    def cget(self, key):
        return self.state.get(key)

    def pack(self, **kwargs):
        self.packed = True
        self.pack_calls.append(kwargs)

    def pack_forget(self):
        self.packed = False
        self.pack_forget_count += 1

    def pack_configure(self, **kwargs):
        self.pack_calls.append(kwargs)

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = callback

    def destroy(self):
        self.destroyed = True

    def winfo_exists(self):
        return not self.destroyed

    @property
    def text(self):
        return self.state.get("text")


class FakeButton(FakeWidget):
    def __init__(self, **initial):
        super().__init__(**initial)
        self.texts = []
        self.enabled = True
        self.enabled_history = []

    def set_text(self, value):
        self.state["text"] = value
        self.texts.append(value)

    def set_enabled(self, value):
        self.enabled = value
        self.enabled_history.append(value)


class FakeGlyphButton(FakeWidget):
    def __init__(self, glyph="max", **initial):
        super().__init__(**initial)
        self.glyph = glyph
        self.glyph_history = []

    def set_glyph(self, value):
        self.glyph = value
        self.glyph_history.append(value)


class FakeListbox(FakeWidget):
    def __init__(self, **initial):
        super().__init__(**initial)
        self.items = []
        self.selection = ()

    def delete(self, first, last=None):
        if first == 0 and last in ("end", None):
            self.items.clear()
        else:
            del self.items[first:(len(self.items) if last == "end" else last + 1)]

    def insert(self, index, *values):
        if index == "end":
            self.items.extend(values)
        else:
            self.items[index:index] = values

    def curselection(self):
        return self.selection

    def size(self):
        return len(self.items)


class FakeText(FakeWidget):
    def __init__(self, **initial):
        super().__init__(**initial)
        self.content = []
        self.see_calls = []

    def insert(self, index, text):
        self.content.append(text)

    def see(self, index):
        self.see_calls.append(index)

    @property
    def value(self):
        return "".join(self.content)


class FakeCanvas(FakeWidget):
    def __init__(self, width=200, height=20, **initial):
        super().__init__(**initial)
        self._width = width
        self._height = height
        self.shapes = []

    def delete(self, *tags):
        self.shapes.clear()

    def _record(self, kind, coords, kwargs):
        self.shapes.append({"kind": kind, "coords": tuple(coords), "options": kwargs})
        return len(self.shapes)

    def create_rectangle(self, *coords, **kwargs):
        return self._record("rectangle", coords, kwargs)

    def create_polygon(self, *coords, **kwargs):
        return self._record("polygon", coords, kwargs)

    def create_line(self, *coords, **kwargs):
        return self._record("line", coords, kwargs)

    def create_text(self, *coords, **kwargs):
        return self._record("text", coords, kwargs)

    def winfo_width(self):
        return self._width

    def winfo_height(self):
        return self._height

    def set_size(self, width, height):
        self._width = width
        self._height = height

    def shapes_of(self, kind):
        return [shape for shape in self.shapes if shape["kind"] == kind]

    def shapes_filled(self, color):
        return [shape for shape in self.shapes if shape["options"].get("fill") == color]


class FakeRoot:
    def __init__(self, run_immediate=True, x=100, y=50, width=588, height=690,
                 screen_width=1920, screen_height=1080):
        self.run_immediate = run_immediate
        self.after_calls = []
        self.pending = []
        self.destroyed = False
        self.iconified = False
        self.geometry_calls = []
        self.update_idletasks_count = 0
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._screen_width = screen_width
        self._screen_height = screen_height

    def after(self, delay, callback=None):
        self.after_calls.append((delay, callback))
        if callback is not None and self.run_immediate and delay == 0:
            callback()
        elif callback is not None:
            self.pending.append((delay, callback))
        return f"after#{len(self.after_calls)}"

    def run_pending(self):
        due = list(self.pending)
        self.pending.clear()
        for _delay, callback in due:
            callback()
        return len(due)

    def destroy(self):
        self.destroyed = True

    def iconify(self):
        self.iconified = True

    def deiconify(self):
        self.iconified = False

    def withdraw(self):
        pass

    def update_idletasks(self):
        self.update_idletasks_count += 1

    def geometry(self, spec=None):
        if spec is None:
            return f"{self._width}x{self._height}+{self._x}+{self._y}"
        self.geometry_calls.append(spec)
        return None

    def winfo_x(self):
        return self._x

    def winfo_y(self):
        return self._y

    def winfo_width(self):
        return self._width

    def winfo_height(self):
        return self._height

    def winfo_screenwidth(self):
        return self._screen_width

    def winfo_screenheight(self):
        return self._screen_height

    def winfo_id(self):
        return 4242

    def bind(self, sequence, callback, add=None):
        pass

    def protocol(self, name, callback):
        pass


APP_ATTRIBUTES = (
    "root", "S", "_ui_line", "_min_w", "_min_h", "_normal_geometry", "_maximized",
    "_resize", "_drag_offset", "files", "model", "_loaded_model_name", "_loaded_device",
    "is_transcribing", "cancel_requested", "_progress_value", "_session_log",
    "_log_flush_idx", "_log_path", "_session_start", "_gpu_name", "_gpu_ready",
    "language_var", "model_var", "device_var", "multilingual_var", "output_var",
    "_icon_big", "_icon_small", "_hwnd", "_icon_images", "_shell", "titlebar",
    "title_label", "_max_button", "file_listbox", "gpu_status_label", "gpu_button",
    "transcribe_btn", "progress_label", "progress_pct", "progress_canvas", "log_text",
    "status_left", "status_files",
)


class FakeUiQueue:
    def __init__(self, app):
        self.app = app

    def put(self, callback):
        self.app.root.after(0, callback)


def make_app(**overrides):
    import ui

    app = object.__new__(ui.SotvoxApp)
    defaults = {
        "root": FakeRoot(),
        "S": 1.0,
        "_ui_line": 13,
        "_min_w": 560,
        "_min_h": 520,
        "_normal_geometry": "588x690+100+50",
        "_maximized": False,
        "_resize": None,
        "_drag_offset": (0, 0),
        "files": [],
        "model": None,
        "_loaded_model_name": None,
        "_loaded_device": None,
        "is_transcribing": False,
        "cancel_requested": False,
        "_progress_value": 0,
        "_session_log": [],
        "_log_flush_idx": 0,
        "_log_path": None,
        "_session_start": 1000.0,
        "_gpu_name": None,
        "_gpu_ready": False,
        "language_var": FakeVar("Spanish"),
        "model_var": FakeVar("large-v3-turbo"),
        "device_var": FakeVar("Auto"),
        "multilingual_var": FakeVar(False),
        "output_var": FakeVar("C:\\sotvox-out"),
        "_icon_big": None,
        "_icon_small": None,
        "_hwnd": None,
        "_icon_images": [],
        "_shell": FakeWidget(),
        "titlebar": FakeWidget(),
        "title_label": FakeWidget(),
        "_max_button": FakeGlyphButton(),
        "file_listbox": FakeListbox(),
        "gpu_status_label": FakeWidget(text="GPU: checking..."),
        "gpu_button": FakeButton(),
        "transcribe_btn": FakeButton(text="Transcribe"),
        "progress_label": FakeWidget(text="Ready"),
        "progress_pct": FakeWidget(text=""),
        "progress_canvas": FakeCanvas(),
        "log_text": FakeText(),
        "status_left": FakeWidget(text="Ready"),
        "status_files": FakeWidget(text="0 files"),
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(app, name, value)
    if "_ui_queue" not in overrides:
        app._ui_queue = FakeUiQueue(app)
    return app


class FakeCodecContext:
    def __init__(self, name="aac", width=None, height=None, sample_rate=None, channels=None):
        self.name = name
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.channels = channels


class FakeStream:
    def __init__(self, stream_type, codec_name="aac", width=None, height=None,
                 sample_rate=None, channels=None, average_rate=None):
        self.type = stream_type
        self.codec_context = FakeCodecContext(codec_name, width, height, sample_rate, channels)
        self.average_rate = average_rate


class FakeFormat:
    def __init__(self, long_name="QuickTime / MOV"):
        self.long_name = long_name


class FakeContainer:
    def __init__(self, duration=5_000_000, bit_rate=128000, streams=None,
                 format_long_name="QuickTime / MOV"):
        self.duration = duration
        self.bit_rate = bit_rate
        self.streams = list(streams or [])
        self.format = FakeFormat(format_long_name)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return False


def fake_av_open(container=None, error=None):
    calls = []

    def opener(filepath, *args, **kwargs):
        calls.append(filepath)
        if error is not None:
            raise error
        return container

    opener.calls = calls
    return opener


class FakeHTTPResponse:
    def __init__(self, payload=b"", headers=None, read_error_after=None):
        self.payload = payload
        self.headers = headers if headers is not None else {}
        self._position = 0
        self._read_error_after = read_error_after
        self._read_count = 0

    def read(self, size=-1):
        self._read_count += 1
        if self._read_error_after is not None and self._read_count > self._read_error_after:
            raise OSError("connection reset")
        if size is None or size < 0:
            chunk = self.payload[self._position:]
            self._position = len(self.payload)
        else:
            chunk = self.payload[self._position:self._position + size]
            self._position += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def json_response(payload, headers=None):
    return FakeHTTPResponse(json.dumps(payload).encode("utf-8"), headers)


def pypi_metadata(releases):
    return {"releases": releases}


def wheel_file(filename, url=None, size=1024, yanked=False):
    return {
        "filename": filename,
        "url": url or f"https://files.pythonhosted.org/{filename}",
        "size": size,
        "yanked": yanked,
    }


def build_wheel_archive(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def write_silent_wav(path, seconds=0.5, sample_rate=16000, channels=1):
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count * channels)
    return path


def write_video_without_audio(path, seconds=0.4, fps=10, width=64, height=48):
    try:
        import av
        import numpy
    except ImportError:
        return None
    try:
        container = av.open(str(path), mode="w")
        try:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for index in range(int(seconds * fps)):
                image = numpy.full((height, width, 3), index * 5 % 256, dtype=numpy.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    except Exception:
        return None
    return path
