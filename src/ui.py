import os
import re
import sys
import time
import ctypes
import platform
import threading
import traceback
import subprocess
import winsound
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
from datetime import datetime
import tkinterdnd2 as tkdnd
from faster_whisper import WhisperModel

from constants import (
    SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, DEFAULT_OUTPUT_DIR, LANG_MAP,
    LOG_DIR, SOUNDS_DIR, ICON_PATH, ASSETS_DIR,
)
from engine import (
    has_audio_stream, transcribe_audio, transcribe_audio_multilingual,
    save_transcript, probe_file,
)
import gpu_pack


FACE = "#c0c0c0"
LIGHT = "#dfdfdf"
HILIGHT = "#ffffff"
SHADOW = "#808080"
DKSHADOW = "#000000"
WHITE = "#ffffff"
BLACK = "#000000"
GRAYTEXT = "#808080"
NAVY = "#000080"
TITLETEXT = "#ffffff"
TEAL = "#008080"
SELBG = "#000080"
SELFG = "#ffffff"

FONT_UI = ("MS Sans Serif", 8)
FONT_TITLE = ("MS Sans Serif", 8, "bold")
FONT_MONO = ("Fixedsys", 9)


def _bevel_lines(canvas, x0, y0, x1, y1, raised=True, fill=FACE, t=1):
    if fill:
        canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
    if raised:
        tl_out, br_out, tl_in, br_in = HILIGHT, DKSHADOW, LIGHT, SHADOW
    else:
        tl_out, br_out, tl_in, br_in = SHADOW, HILIGHT, DKSHADOW, LIGHT
    canvas.create_rectangle(x0, y0, x1, y0 + t, fill=tl_out, outline="")
    canvas.create_rectangle(x0, y0, x0 + t, y1, fill=tl_out, outline="")
    canvas.create_rectangle(x0, y1 - t, x1, y1, fill=br_out, outline="")
    canvas.create_rectangle(x1 - t, y0, x1, y1, fill=br_out, outline="")
    canvas.create_rectangle(x0 + t, y0 + t, x1 - t, y0 + 2 * t, fill=tl_in, outline="")
    canvas.create_rectangle(x0 + t, y0 + t, x0 + 2 * t, y1 - t, fill=tl_in, outline="")
    canvas.create_rectangle(x0 + t, y1 - 2 * t, x1 - t, y1 - t, fill=br_in, outline="")
    canvas.create_rectangle(x1 - 2 * t, y0 + t, x1 - t, y1 - t, fill=br_in, outline="")


class _Win95Scrollbar(tk.Canvas):
    def __init__(self, parent, command=None, scale=1.0, **kwargs):
        self._scale = scale
        self.BAR = max(12, int(round(16 * scale)))
        super().__init__(parent, width=self.BAR, height=int(round(48 * scale)),
                         highlightthickness=0, bd=0, bg=FACE, **kwargs)
        self._command = command
        self._lo = 0.0
        self._hi = 1.0
        self._drag = None
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

    def set(self, lo, hi):
        self._lo = float(lo)
        self._hi = float(hi)
        self._draw()

    def _metrics(self):
        h = self.winfo_height()
        bar = self.BAR
        track_h = max(0, h - bar * 2)
        span = self._hi - self._lo
        if span >= 1.0:
            return h, bar, track_h, bar, track_h
        thumb_h = max(bar, int(track_h * span))
        movable = max(1, track_h - thumb_h)
        thumb_y = bar + int(movable * (self._lo / max(1e-6, 1.0 - span)))
        return h, bar, track_h, thumb_y, thumb_h

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        h, bar, track_h, thumb_y, thumb_h = self._metrics()
        if h < bar * 2 + 4:
            return
        t = max(1, int(round(self._scale)))
        self.create_rectangle(0, bar, w, h - bar, fill=FACE, outline="")
        self.create_rectangle(0, bar, w, h - bar, fill=WHITE, outline="", stipple="gray50")
        _bevel_lines(self, 0, 0, w - 1, bar - 1, raised=True, fill=FACE, t=t)
        self._triangle(w // 2, bar // 2, "up")
        _bevel_lines(self, 0, h - bar, w - 1, h - 1, raised=True, fill=FACE, t=t)
        self._triangle(w // 2, h - bar // 2 - 1, "down")
        if self._hi - self._lo < 1.0:
            _bevel_lines(self, 0, thumb_y, w - 1, thumb_y + thumb_h, raised=True, fill=FACE, t=t)

    def _triangle(self, cx, cy, direction):
        a = int(round(4 * self._scale))
        b = int(round(3 * self._scale))
        if direction == "up":
            self.create_polygon(cx, cy - b, cx - a, cy + b, cx + a, cy + b, fill=BLACK, outline="")
        else:
            self.create_polygon(cx, cy + b, cx - a, cy - b, cx + a, cy - b, fill=BLACK, outline="")

    def _press(self, event):
        if not self._command:
            return
        h, bar, track_h, thumb_y, thumb_h = self._metrics()
        if event.y < bar:
            self._command("scroll", -1, "units")
        elif event.y > h - bar:
            self._command("scroll", 1, "units")
        elif thumb_y <= event.y <= thumb_y + thumb_h:
            self._drag = (event.y, self._lo)
        elif event.y < thumb_y:
            self._command("scroll", -1, "pages")
        else:
            self._command("scroll", 1, "pages")

    def _motion(self, event):
        if not self._drag or not self._command:
            return
        h, bar, track_h, thumb_y, thumb_h = self._metrics()
        start_y, start_lo = self._drag
        span = self._hi - self._lo
        movable = max(1, track_h - thumb_h)
        delta = (event.y - start_y) / movable
        new_lo = min(max(0.0, start_lo + delta * (1.0 - span)), max(0.0, 1.0 - span))
        self._command("moveto", str(new_lo))


class SotvoxApp:
    def __init__(self):
        self.root = tkdnd.Tk()
        self.root.title("Sotvox")
        self.root.configure(bg=FACE)

        self.S = self.root.winfo_fpixels("1i") / 96.0
        self._ui_line = tkfont.Font(family="MS Sans Serif", size=8).metrics("linespace")

        win_w, win_h = self.px(588), self.px(690)
        self._min_w, self._min_h = self.px(560), self.px(520)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = (screen_w - win_w) // 2
        pos_y = max(0, (screen_h - win_h) // 2 - self.px(20))
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self._normal_geometry = f"{win_w}x{win_h}+{pos_x}+{pos_y}"
        self._maximized = False
        self._resize = None
        self._drag_offset = (0, 0)

        self.files = []
        self.model = None
        self._loaded_model_name = None
        self._loaded_device = None
        self.is_transcribing = False
        self.cancel_requested = False
        self._progress_value = 0
        self._session_log = []
        self._log_flush_idx = 0
        self._log_path = None
        self._session_start = time.time()
        self._gpu_name = None
        self._gpu_ready = False

        self.language_var = tk.StringVar(value="Spanish")
        self.model_var = tk.StringVar(value="large-v3")
        self.device_var = tk.StringVar(value="Auto")
        self.multilingual_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)

        self._icon_big = None
        self._icon_small = None
        self._hwnd = None

        self.root.withdraw()
        self.root.bind("<Map>", self._on_remap)

        self._set_icon()
        self._build_ui()
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(0, self._reveal)

        self._init_log()
        self.language_var.trace_add("write", lambda *_: self._slog(f"Setting changed: language = {self.language_var.get()}"))
        self.model_var.trace_add("write", lambda *_: self._slog(f"Setting changed: model = {self.model_var.get()}"))
        self.device_var.trace_add("write", lambda *_: self._slog(f"Setting changed: device = {self.device_var.get()}"))
        self.multilingual_var.trace_add("write", lambda *_: self._slog(f"Setting changed: multilingual = {self.multilingual_var.get()}"))
        self.output_var.trace_add("write", lambda *_: self._slog(f"Setting changed: output path = {self.output_var.get()}"))

    def px(self, n):
        return max(1, int(round(n * self.S)))

    def _set_icon(self):
        self._icon_images = []
        try:
            for size in (48, 32, 16):
                png = os.path.join(ASSETS_DIR, f"sotvox_{size}.png")
                if os.path.exists(png):
                    self._icon_images.append(tk.PhotoImage(file=png))
            if self._icon_images:
                self.root.iconphoto(True, *self._icon_images)
        except Exception:
            pass
        try:
            if os.path.exists(ICON_PATH):
                self.root.iconbitmap(default=ICON_PATH)
        except Exception:
            pass
        try:
            if os.path.exists(ICON_PATH):
                user32 = ctypes.windll.user32
                user32.LoadImageW.restype = ctypes.c_void_p
                user32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_uint]
                LR_LOADFROMFILE = 0x00000010
                IMAGE_ICON = 1
                self._icon_big = user32.LoadImageW(None, ICON_PATH, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
                self._icon_small = user32.LoadImageW(None, ICON_PATH, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        except Exception:
            pass

    def _reapply_icon(self):
        try:
            if self._icon_images:
                self.root.iconphoto(True, *self._icon_images)
        except Exception:
            pass
        try:
            user32 = ctypes.windll.user32
            user32.GetParent.restype = ctypes.c_void_p
            user32.GetParent.argtypes = [ctypes.c_void_p]
            user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
            WM_SETICON = 0x0080
            hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            if self._icon_big:
                user32.SendMessageW(hwnd, WM_SETICON, ctypes.c_void_p(1), ctypes.c_void_p(self._icon_big))
            if self._icon_small:
                user32.SendMessageW(hwnd, WM_SETICON, ctypes.c_void_p(0), ctypes.c_void_p(self._icon_small))
        except Exception:
            pass

    def _reveal(self):
        self.root.update_idletasks()
        self._make_borderless()
        self.root.deiconify()
        self.root.after(30, self._reapply_icon)

    def _make_borderless(self):
        try:
            user32 = ctypes.windll.user32
            user32.GetParent.restype = ctypes.c_void_p
            user32.GetParent.argtypes = [ctypes.c_void_p]
            GWL_STYLE = -16
            GWL_EXSTYLE = -20
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_BORDER = 0x00800000
            WS_DLGFRAME = 0x00400000
            WS_MINIMIZEBOX = 0x00020000
            WS_MAXIMIZEBOX = 0x00010000
            WS_SYSMENU = 0x00080000
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_WINDOWEDGE = 0x00000100
            WS_EX_CLIENTEDGE = 0x00000200
            WS_EX_DLGMODALFRAME = 0x00000001
            WS_EX_STATICEDGE = 0x00020000
            SWP_FRAMECHANGED = 0x0020
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            self._hwnd = hwnd
            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            style = (style & ~(WS_CAPTION | WS_THICKFRAME | WS_BORDER | WS_DLGFRAME)) \
                | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU
            user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex = (ex & ~(WS_EX_TOOLWINDOW | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE
                         | WS_EX_DLGMODALFRAME | WS_EX_STATICEDGE)) | WS_EX_APPWINDOW
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)
        except Exception:
            pass

    def _bevel(self, parent, style="raised", bg=FACE):
        if style == "raised":
            ring = [HILIGHT, DKSHADOW, LIGHT, SHADOW]
        elif style == "sunken":
            ring = [SHADOW, HILIGHT, DKSHADOW, LIGHT]
        else:
            ring = [SHADOW, HILIGHT, HILIGHT, SHADOW]
        t = self.px(1)
        a = tk.Frame(parent, bg=ring[0])
        b = tk.Frame(a, bg=ring[1]); b.pack(fill="both", expand=True, padx=(t, 0), pady=(t, 0))
        c = tk.Frame(b, bg=ring[2]); c.pack(fill="both", expand=True, padx=(0, t), pady=(0, t))
        d = tk.Frame(c, bg=ring[3]); d.pack(fill="both", expand=True, padx=(t, 0), pady=(t, 0))
        content = tk.Frame(d, bg=bg); content.pack(fill="both", expand=True, padx=(0, t), pady=(0, t))
        return a, content

    def _mk_button(self, parent, text, command=None, font=None, pad_x=10, pad_y=3,
                   width_chars=0, default=False):
        font = font or FONT_UI
        raised = [HILIGHT, DKSHADOW, LIGHT, SHADOW, FACE]
        sunken = [SHADOW, HILIGHT, DKSHADOW, LIGHT, FACE]
        state = {"enabled": True}
        t = self.px(1)
        px_pad, py_pad = self.px(pad_x), self.px(pad_y)

        holder = tk.Frame(parent, bg=BLACK if default else FACE)
        ring_pad = t if default else 0
        a = tk.Frame(holder, bg=raised[0]); a.pack(padx=ring_pad, pady=ring_pad)
        b = tk.Frame(a, bg=raised[1]); b.pack(padx=(t, 0), pady=(t, 0))
        c = tk.Frame(b, bg=raised[2]); c.pack(padx=(0, t), pady=(0, t))
        d = tk.Frame(c, bg=raised[3]); d.pack(padx=(t, 0), pady=(t, 0))
        face = tk.Frame(d, bg=FACE); face.pack(padx=(0, t), pady=(0, t))
        label_kwargs = {"text": text, "font": font, "bg": FACE, "fg": BLACK}
        if width_chars:
            label_kwargs["width"] = width_chars
        lbl = tk.Label(face, **label_kwargs)
        lbl.pack(padx=px_pad, pady=py_pad)
        rings = [a, b, c, d, face]

        def press(_=None):
            if not state["enabled"]:
                return
            for fr, col in zip(rings, sunken):
                fr.configure(bg=col)
            lbl.pack_configure(padx=(px_pad + t, px_pad - t), pady=(py_pad + t, py_pad - t))

        def release(event=None):
            if not state["enabled"]:
                return
            for fr, col in zip(rings, raised):
                fr.configure(bg=col)
            lbl.pack_configure(padx=px_pad, pady=py_pad)
            if event is not None and command:
                ex, ey = event.x_root, event.y_root
                fx, fy = face.winfo_rootx(), face.winfo_rooty()
                if 0 <= ex - fx <= face.winfo_width() and 0 <= ey - fy <= face.winfo_height():
                    command()

        for widget in (lbl, face):
            widget.bind("<ButtonPress-1>", press)
            widget.bind("<ButtonRelease-1>", release)

        def set_text(text_value):
            lbl.configure(text=text_value)

        def set_enabled(value):
            state["enabled"] = value
            lbl.configure(fg=BLACK if value else GRAYTEXT)

        holder.set_text = set_text
        holder.set_enabled = set_enabled
        return holder

    def _group(self, parent, title):
        holder = tk.Frame(parent, bg=FACE)
        border, content = self._bevel(holder, style="etched", bg=FACE)
        border.pack(fill="both", expand=True, pady=(self._ui_line // 2, 0))
        tk.Label(holder, text=f" {title} ", bg=FACE, fg=BLACK, font=FONT_UI).place(x=self.px(9), y=0)
        return holder, content

    def _combo(self, parent, variable, values, width_chars=14):
        border, content = self._bevel(parent, style="sunken", bg=WHITE)
        content.configure(bg=WHITE)
        lbl = tk.Label(content, textvariable=variable, font=FONT_UI, bg=WHITE, fg=BLACK,
                       anchor="w", width=width_chars, padx=self.px(3), pady=self.px(1))
        lbl.pack(side="left", fill="both", expand=True)

        raised = [HILIGHT, DKSHADOW, LIGHT, SHADOW]
        t = self.px(1)
        aa = tk.Frame(content, bg=raised[0])
        ab = tk.Frame(aa, bg=raised[1]); ab.pack(fill="both", expand=True, padx=(t, 0), pady=(t, 0))
        ac = tk.Frame(ab, bg=raised[2]); ac.pack(fill="both", expand=True, padx=(0, t), pady=(0, t))
        ad = tk.Frame(ac, bg=raised[3]); ad.pack(fill="both", expand=True, padx=(t, 0), pady=(t, 0))
        aface = tk.Frame(ad, bg=FACE); aface.pack(fill="both", expand=True, padx=(0, t), pady=(0, t))
        size = self.px(15)
        arrow_canvas = tk.Canvas(aface, width=size, height=size, bg=FACE, highlightthickness=0, bd=0)
        arrow_canvas.pack(padx=self.px(2), pady=self.px(1))
        cx = size // 2
        w = self.px(4)
        arrow_canvas.create_polygon(cx - w, cx - self.px(1), cx + w, cx - self.px(1),
                                    cx, cx + self.px(3), fill=BLACK, outline="")
        aa.pack(side="right", fill="y")

        menu = tk.Menu(content, tearoff=0, bg=FACE, fg=BLACK,
                       activebackground=NAVY, activeforeground=WHITE,
                       font=FONT_UI, borderwidth=1, relief="raised")
        for val in values:
            menu.add_command(label=val, command=lambda v=val: variable.set(v))

        def show(_=None):
            menu.post(border.winfo_rootx(), border.winfo_rooty() + border.winfo_height())

        for w2 in (content, lbl, arrow_canvas, aface):
            w2.bind("<Button-1>", show)
        return border

    def _check(self, parent, variable, text):
        outer = tk.Frame(parent, bg=FACE, cursor="arrow")
        sz = self.px(13)
        box = tk.Canvas(outer, width=sz, height=sz, bg=FACE, highlightthickness=0, bd=0)
        box.pack(side="left", padx=(0, self.px(5)))
        lbl = tk.Label(outer, text=text, font=FONT_UI, bg=FACE, fg=BLACK)
        lbl.pack(side="left")
        t = self.px(1)

        def draw():
            box.delete("all")
            _bevel_lines(box, 0, 0, sz - 1, sz - 1, raised=False, fill=WHITE, t=t)
            if variable.get():
                s = self.S
                pts = [(3, 6), (5, 9), (10, 3)]
                for width_off in range(t):
                    self._polyline(box, [(int(x * s), int(y * s) + width_off) for x, y in pts], BLACK)
                    self._polyline(box, [(int(x * s) + width_off, int(y * s)) for x, y in pts], BLACK)

        def toggle(_=None):
            variable.set(not variable.get())

        for w in (outer, box, lbl):
            w.bind("<Button-1>", toggle)
        variable.trace_add("write", lambda *_: draw())
        draw()
        return outer

    def _polyline(self, canvas, points, color):
        for i in range(len(points) - 1):
            canvas.create_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], fill=color)

    def _title_button(self, parent, glyph, command, enabled=True):
        raised = [HILIGHT, DKSHADOW, LIGHT, SHADOW]
        t = self.px(1)
        a = tk.Frame(parent, bg=raised[0])
        b = tk.Frame(a, bg=raised[1]); b.pack(padx=(t, 0), pady=(t, 0))
        c = tk.Frame(b, bg=raised[2]); c.pack(padx=(0, t), pady=(0, t))
        d = tk.Frame(c, bg=raised[3]); d.pack(padx=(t, 0), pady=(t, 0))
        face = tk.Frame(d, bg=FACE); face.pack(padx=(0, t), pady=(0, t))
        cw, ch = self.px(13), self.px(11)
        canvas = tk.Canvas(face, width=cw, height=ch, bg=FACE, highlightthickness=0, bd=0)
        canvas.pack(padx=self.px(1), pady=self.px(1))
        gcol = BLACK if enabled else SHADOW
        s = self.S

        def scaled(x, y):
            return int(x * s), int(y * s)

        def draw(g):
            canvas.delete("all")
            if g == "min":
                x0, y0 = scaled(2, 8); x1, y1 = scaled(9, 10)
                canvas.create_rectangle(x0, y0, x1, y1, fill=gcol, outline="")
            elif g == "max":
                x0, y0 = scaled(1, 1); x1, y1 = scaled(11, 10)
                canvas.create_rectangle(x0, y0, x1, y1, outline=gcol, width=self.px(1))
                canvas.create_rectangle(x0, y0, x1, scaled(11, 2)[1], fill=gcol, outline="")
            elif g == "restore":
                canvas.create_rectangle(*scaled(3, 1), *scaled(10, 6), outline=gcol, width=self.px(1))
                canvas.create_rectangle(*scaled(3, 1), *scaled(10, 2), fill=gcol, outline="")
                canvas.create_rectangle(*scaled(1, 4), *scaled(8, 9), outline=gcol, width=self.px(1))
                canvas.create_rectangle(*scaled(1, 4), *scaled(8, 5), fill=gcol, outline="")
                canvas.create_rectangle(*scaled(2, 5), *scaled(7, 8), fill=FACE, outline="")
            elif g == "close":
                x0, y0 = scaled(2, 1)
                x1, y1 = scaled(10, 9)
                lw = self.px(1)
                canvas.create_line(x0, y0, x1, y1, fill=gcol, width=lw)
                canvas.create_line(x1, y0, x0, y1, fill=gcol, width=lw)

        draw(glyph)
        a.set_glyph = draw
        if enabled and command:
            for w in (canvas, face):
                w.bind("<Button-1>", lambda e: command())
        return a

    def _build_ui(self):
        shell_border, shell = self._bevel(self.root, style="raised", bg=FACE)
        shell_border.pack(fill="both", expand=True)
        self._shell = shell

        self._build_titlebar()
        self._build_menubar()
        self._build_statusbar()

        main = tk.Frame(shell, bg=FACE)
        main.pack(fill="both", expand=True, padx=self.px(8), pady=(self.px(4), self.px(2)))

        self._build_files(main)
        self._build_options(main)
        self._build_output(main)
        self._build_actions(main)
        self._build_progress(main)
        self._build_log(main)

    def _build_titlebar(self):
        bar = tk.Frame(self._shell, bg=NAVY, height=self.px(20))
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        self.titlebar = bar

        isz = self.px(16)
        icon = tk.Canvas(bar, width=isz, height=isz, bg=NAVY, highlightthickness=0, bd=0)
        icon.pack(side="left", padx=(self.px(3), self.px(3)), pady=self.px(2))
        s = self.S
        icon.create_rectangle(int(2 * s), int(1 * s), int(12 * s), int(15 * s), fill=WHITE, outline=BLACK)
        for yy in (4, 6, 8):
            icon.create_line(int(4 * s), int(yy * s), int(10 * s), int(yy * s), fill=NAVY)
        icon.create_polygon(int(11 * s), int(9 * s), int(14 * s), int(6 * s),
                            int(14 * s), int(14 * s), int(11 * s), int(11 * s), fill=TEAL, outline=BLACK)

        self.title_label = tk.Label(bar, text="Sotvox", bg=NAVY, fg=TITLETEXT, font=FONT_TITLE)
        self.title_label.pack(side="left")

        btns = tk.Frame(bar, bg=NAVY)
        btns.pack(side="right", padx=self.px(2), pady=self.px(1))
        self._title_button(btns, "close", self._on_close).pack(side="right", padx=(self.px(2), 0))
        self._max_button = self._title_button(btns, "max", self._toggle_max)
        self._max_button.pack(side="right")
        self._title_button(btns, "min", self._minimize).pack(side="right")

        for w in (bar, icon, self.title_label):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)
            w.bind("<Double-Button-1>", lambda e: self._toggle_max())

    def _build_menubar(self):
        bar = tk.Frame(self._shell, bg=FACE)
        bar.pack(fill="x", side="top")

        file_menu = tk.Menu(bar, tearoff=0, bg=FACE, fg=BLACK, activebackground=NAVY,
                            activeforeground=WHITE, font=FONT_UI, borderwidth=1, relief="raised")
        file_menu.add_command(label="Add Files...", command=self._browse_files, underline=0)
        file_menu.add_command(label="Open Output Folder", command=self._open_output, underline=0)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close, underline=1)

        edit_menu = tk.Menu(bar, tearoff=0, bg=FACE, fg=BLACK, activebackground=NAVY,
                            activeforeground=WHITE, font=FONT_UI, borderwidth=1, relief="raised")
        edit_menu.add_command(label="Remove Selected", command=self._remove_selected, underline=0)
        edit_menu.add_command(label="Clear List", command=self._clear_files, underline=0)

        help_menu = tk.Menu(bar, tearoff=0, bg=FACE, fg=BLACK, activebackground=NAVY,
                            activeforeground=WHITE, font=FONT_UI, borderwidth=1, relief="raised")
        help_menu.add_command(label="About Sotvox...", command=self._about, underline=0)

        for text, menu in (("File", file_menu), ("Edit", edit_menu), ("Help", help_menu)):
            self._menu_item(bar, text, menu)

    def _menu_item(self, bar, text, menu):
        font = tkfont.Font(family="MS Sans Serif", size=8)
        pad = self.px(7)
        first_w = font.measure(text[0])
        baseline = self.px(2) + font.metrics("ascent")
        canvas = tk.Canvas(bar, width=font.measure(text) + pad * 2,
                           height=baseline + self.px(4), bg=FACE, highlightthickness=0, bd=0)
        canvas.pack(side="left")

        def redraw(bg, fg):
            canvas.delete("all")
            canvas.configure(bg=bg)
            canvas.create_text(pad, self.px(2), text=text, anchor="nw", fill=fg, font=font)
            canvas.create_line(pad, baseline + 1, pad + first_w, baseline + 1, fill=fg)

        def post(event):
            menu.post(canvas.winfo_rootx(), canvas.winfo_rooty() + canvas.winfo_height())

        canvas.bind("<Button-1>", post)
        canvas.bind("<Enter>", lambda e: redraw(NAVY, WHITE))
        canvas.bind("<Leave>", lambda e: redraw(FACE, BLACK))
        redraw(FACE, BLACK)

    def _build_files(self, parent):
        group, content = self._group(parent, "Files")
        group.pack(fill="both", expand=True)

        hint = tk.Label(content, text="Drag audio or video files onto the list, or click Add Files.",
                        bg=FACE, fg=BLACK, font=FONT_UI, anchor="w")
        hint.pack(fill="x", padx=self.px(8), pady=(self.px(6), self.px(4)))

        list_row = tk.Frame(content, bg=FACE)
        list_row.pack(fill="both", expand=True, padx=self.px(8), pady=(0, self.px(6)))

        border, well = self._bevel(list_row, style="sunken", bg=WHITE)
        border.pack(side="left", fill="both", expand=True)

        self.file_listbox = tk.Listbox(
            well, bg=WHITE, fg=BLACK, font=FONT_UI, height=6,
            borderwidth=0, highlightthickness=0, relief="flat",
            selectbackground=SELBG, selectforeground=SELFG,
            activestyle="none", selectmode="extended", exportselection=False)
        scrollbar = _Win95Scrollbar(well, command=self.file_listbox.yview, scale=self.S)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.pack(side="left", fill="both", expand=True)

        self.file_listbox.drop_target_register(tkdnd.DND_FILES)
        self.file_listbox.dnd_bind("<<Drop>>", self._on_drop)

        btn_col = tk.Frame(list_row, bg=FACE)
        btn_col.pack(side="left", fill="y", padx=(self.px(6), 0))
        self._mk_button(btn_col, "Add Files...", command=self._browse_files,
                        width_chars=10).pack(fill="x", pady=(0, self.px(4)))
        self._mk_button(btn_col, "Remove", command=self._remove_selected,
                        width_chars=10).pack(fill="x", pady=(0, self.px(4)))
        self._mk_button(btn_col, "Clear All", command=self._clear_files,
                        width_chars=10).pack(fill="x")

    def _build_options(self, parent):
        group, content = self._group(parent, "Options")
        group.pack(fill="x", pady=(self.px(6), 0))

        row1 = tk.Frame(content, bg=FACE)
        row1.pack(fill="x", padx=self.px(10), pady=(self.px(8), self.px(4)))

        tk.Label(row1, text="Language:", bg=FACE, fg=BLACK, font=FONT_UI).pack(side="left")
        self._combo(row1, self.language_var,
                    ["Auto-detect", "Spanish", "English", "Portuguese", "French",
                     "German", "Italian", "Japanese", "Chinese", "Korean"],
                    width_chars=11).pack(side="left", padx=(self.px(4), self.px(14)))

        tk.Label(row1, text="Model:", bg=FACE, fg=BLACK, font=FONT_UI).pack(side="left")
        self._combo(row1, self.model_var,
                    ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"],
                    width_chars=13).pack(side="left", padx=(self.px(4), self.px(14)))

        tk.Label(row1, text="Device:", bg=FACE, fg=BLACK, font=FONT_UI).pack(side="left")
        self._combo(row1, self.device_var,
                    ["Auto", "CPU", "GPU (CUDA)"],
                    width_chars=9).pack(side="left", padx=(self.px(4), 0))

        row2 = tk.Frame(content, bg=FACE)
        row2.pack(fill="x", padx=self.px(10), pady=(self.px(2), self.px(4)))
        self._check(row2, self.multilingual_var,
                    "Multilingual mode  (auto-detect language per 30s chunk; overrides Language)").pack(side="left")

        row3 = tk.Frame(content, bg=FACE)
        row3.pack(fill="x", padx=self.px(10), pady=(0, self.px(8)))
        self.gpu_status_label = tk.Label(row3, text="GPU: checking...", bg=FACE, fg=BLACK,
                                        font=FONT_UI, anchor="w")
        self.gpu_status_label.pack(side="left")
        self.gpu_button = self._mk_button(row3, "Enable GPU Acceleration...",
                                         command=self._prompt_gpu_install)
        self._refresh_gpu_row()
        threading.Thread(target=self._detect_gpu, daemon=True).start()

    def _post(self, callback):
        try:
            self.root.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _detect_gpu(self):
        detected_name = gpu_pack.detect_nvidia_gpu()
        self._gpu_name = detected_name
        self._slog(f"GPU detection: {detected_name or 'no NVIDIA GPU found'}")
        self._post(self._refresh_gpu_row)

    def _refresh_gpu_row(self):
        self._gpu_ready = gpu_pack.libraries_available()
        if self._gpu_name is None:
            self.gpu_status_label.configure(text="GPU: no NVIDIA GPU detected — using CPU")
            self.gpu_button.pack_forget()
        elif self._gpu_ready:
            self.gpu_status_label.configure(text=f"GPU: {self._gpu_name} — acceleration ready")
            self.gpu_button.pack_forget()
        else:
            self.gpu_status_label.configure(text=f"GPU: {self._gpu_name} — acceleration not installed")
            self.gpu_button.pack(side="right")

    def _prompt_gpu_install(self):
        dialog, body = self._modal_dialog("GPU Acceleration", self.px(430), self.px(210))

        tk.Label(body, text="Enable GPU acceleration", bg=FACE, fg=BLACK,
                 font=("MS Sans Serif", 10, "bold")).pack(anchor="w")
        message = (
            f"Detected: {self._gpu_name}\n\n"
            "Sotvox can use your NVIDIA GPU to transcribe much faster.\n"
            "This downloads NVIDIA's CUDA libraries (about 1.2 GB) from\n"
            "the official Python package index. It is only needed once."
        )
        tk.Label(body, text=message, bg=FACE, fg=BLACK, font=FONT_UI,
                 justify="left", anchor="w").pack(anchor="w", pady=(self.px(6), 0))

        button_row = tk.Frame(body, bg=FACE)
        button_row.pack(side="bottom", anchor="e", pady=(self.px(10), 0))

        def start():
            dialog.destroy()
            self._run_gpu_install()

        self._mk_button(button_row, "Download and Install", command=start,
                        default=True).pack(side="left", padx=(0, self.px(6)))
        self._mk_button(button_row, "Cancel", command=dialog.destroy,
                        width_chars=8).pack(side="left")

    def _run_gpu_install(self):
        dialog, body = self._modal_dialog("Installing GPU Acceleration", self.px(430), self.px(170))
        cancel_state = {"cancelled": False}

        status_label = tk.Label(body, text="Starting...", bg=FACE, fg=BLACK,
                                font=FONT_UI, anchor="w")
        status_label.pack(fill="x")
        detail_label = tk.Label(body, text="", bg=FACE, fg=BLACK, font=FONT_UI, anchor="w")
        detail_label.pack(fill="x", pady=(self.px(2), self.px(6)))

        bar_border, bar_well = self._bevel(body, style="sunken", bg=FACE)
        bar_border.pack(fill="x")
        bar_canvas = tk.Canvas(bar_well, bg=FACE, highlightthickness=0, bd=0, height=self.px(18))
        bar_canvas.pack(fill="x", padx=self.px(1), pady=self.px(1))

        button_row = tk.Frame(body, bg=FACE)
        button_row.pack(side="bottom", anchor="e", pady=(self.px(10), 0))
        cancel_button = self._mk_button(
            button_row, "Cancel", width_chars=8,
            command=lambda: cancel_state.__setitem__("cancelled", True))
        cancel_button.pack()

        def draw_bar(fraction):
            bar_canvas.delete("all")
            width = bar_canvas.winfo_width()
            height = bar_canvas.winfo_height()
            if width < 4:
                return
            margin = self.px(2)
            filled = int((width - margin * 2) * max(0.0, min(1.0, fraction)))
            block_width = self.px(8)
            gap = self.px(2)
            position = margin
            while position + block_width <= margin + filled:
                bar_canvas.create_rectangle(position, margin, position + block_width,
                                            height - margin, fill=NAVY, outline="")
                position += block_width + gap

        def on_status(text):
            self._post(lambda: status_label.configure(text=text))

        def on_progress(done_bytes, total_bytes):
            fraction = (done_bytes / total_bytes) if total_bytes else 0.0
            text = f"{done_bytes / (1024 * 1024):.0f} MB of {total_bytes / (1024 * 1024):.0f} MB"
            def update():
                detail_label.configure(text=text)
                draw_bar(fraction)
            self._post(update)

        def finish(success, error_text):
            def update():
                if dialog.winfo_exists():
                    dialog.destroy()
                self._refresh_gpu_row()
                if success:
                    self._log("GPU acceleration installed — GPU will be used automatically.")
                    self._message_dialog("GPU Acceleration",
                                         "GPU acceleration is ready.\n\nSotvox will now use your GPU automatically.")
                elif cancel_state["cancelled"]:
                    self._log("GPU acceleration download cancelled.")
                else:
                    self._log(f"GPU acceleration install failed: {error_text}")
                    self._message_dialog("GPU Acceleration",
                                         f"Could not install GPU acceleration.\n\n{error_text}")
            self._post(update)

        def worker():
            try:
                gpu_pack.install(on_status=on_status, on_progress=on_progress,
                                 is_cancelled=lambda: cancel_state["cancelled"],
                                 log=self._slog)
                finish(True, "")
            except Exception as error:
                finish(False, str(error))

        threading.Thread(target=worker, daemon=True).start()

    def _modal_dialog(self, title, width, height):
        dialog = tk.Toplevel(self.root, bg=FACE)
        dialog.overrideredirect(True)
        position_x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        position_y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        dialog.geometry(f"{width}x{height}+{position_x}+{position_y}")

        shell_border, shell = self._bevel(dialog, style="raised", bg=FACE)
        shell_border.pack(fill="both", expand=True)

        bar = tk.Frame(shell, bg=NAVY, height=self.px(20))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text=title, bg=NAVY, fg=WHITE, font=FONT_TITLE).pack(side="left", padx=self.px(6))
        self._title_button(bar, "close", dialog.destroy).pack(side="right", padx=self.px(2), pady=self.px(2))

        body = tk.Frame(shell, bg=FACE)
        body.pack(fill="both", expand=True, padx=self.px(12), pady=self.px(10))

        bar.bind("<ButtonPress-1>", lambda e: dialog.__setattr__(
            "_offset", (e.x_root - dialog.winfo_x(), e.y_root - dialog.winfo_y())))
        bar.bind("<B1-Motion>", lambda e: dialog.geometry(
            f"+{e.x_root - dialog._offset[0]}+{e.y_root - dialog._offset[1]}"))
        dialog.transient(self.root)
        dialog.grab_set()
        return dialog, body

    def _message_dialog(self, title, message):
        dialog, body = self._modal_dialog(title, self.px(390), self.px(170))
        tk.Label(body, text=message, bg=FACE, fg=BLACK, font=FONT_UI,
                 justify="left", anchor="w").pack(anchor="w")
        self._mk_button(body, "OK", command=dialog.destroy, width_chars=8,
                        default=True).pack(side="bottom", pady=(self.px(10), 0))

    def _build_output(self, parent):
        group, content = self._group(parent, "Output")
        group.pack(fill="x", pady=(self.px(6), 0))

        row = tk.Frame(content, bg=FACE)
        row.pack(fill="x", padx=self.px(10), pady=(self.px(8), self.px(8)))

        tk.Label(row, text="Folder:", bg=FACE, fg=BLACK, font=FONT_UI).pack(side="left")
        border, well = self._bevel(row, style="sunken", bg=WHITE)
        border.pack(side="left", fill="x", expand=True, padx=(self.px(4), self.px(6)))
        entry = tk.Entry(well, textvariable=self.output_var, font=FONT_UI, bg=WHITE, fg=BLACK,
                         relief="flat", borderwidth=0, highlightthickness=0,
                         insertbackground=BLACK)
        entry.pack(fill="both", expand=True, ipady=self.px(1), padx=self.px(1), pady=self.px(1))
        self._mk_button(row, "Browse...", command=self._browse_output,
                        width_chars=8).pack(side="left")

    def _build_actions(self, parent):
        action = tk.Frame(parent, bg=FACE)
        action.pack(fill="x", pady=(self.px(8), self.px(2)))

        self.transcribe_btn = self._mk_button(
            action, "Transcribe", command=self._toggle_transcription,
            pad_x=22, pad_y=3, default=True)
        self.transcribe_btn.pack(side="left")

        self._mk_button(action, "Open Output Folder", command=self._open_output,
                        pad_x=12, pad_y=3).pack(side="right")

    def _build_progress(self, parent):
        group, content = self._group(parent, "Progress")
        group.pack(fill="x", pady=(self.px(6), 0))

        status_row = tk.Frame(content, bg=FACE)
        status_row.pack(fill="x", padx=self.px(10), pady=(self.px(8), self.px(4)))
        self.progress_label = tk.Label(status_row, text="Ready", bg=FACE, fg=BLACK,
                                       font=FONT_UI, anchor="w")
        self.progress_label.pack(side="left")
        self.progress_pct = tk.Label(status_row, text="", bg=FACE, fg=BLACK, font=FONT_UI)
        self.progress_pct.pack(side="right")

        bar_border, bar_well = self._bevel(content, style="sunken", bg=FACE)
        bar_border.pack(fill="x", padx=self.px(10), pady=(0, self.px(8)))
        self.progress_canvas = tk.Canvas(bar_well, bg=FACE, highlightthickness=0, bd=0, height=self.px(18))
        self.progress_canvas.pack(fill="x", padx=self.px(1), pady=self.px(1))
        self.progress_canvas.bind("<Configure>", lambda e: self._draw_progress())

    def _build_log(self, parent):
        group, content = self._group(parent, "Log")
        group.pack(fill="both", expand=True, pady=(self.px(6), 0))

        border, well = self._bevel(content, style="sunken", bg=WHITE)
        border.pack(fill="both", expand=True, padx=self.px(8), pady=self.px(8))

        self.log_text = tk.Text(well, bg=WHITE, fg=BLACK, font=FONT_MONO, height=6,
                                wrap="word", state="disabled", borderwidth=0,
                                relief="flat", highlightthickness=0, padx=self.px(3), pady=self.px(2),
                                insertbackground=BLACK, selectbackground=SELBG,
                                selectforeground=SELFG)
        log_scroll = _Win95Scrollbar(well, command=self.log_text.yview, scale=self.S)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def _build_statusbar(self):
        bar = tk.Frame(self._shell, bg=FACE)
        bar.pack(fill="x", side="bottom")
        inner = tk.Frame(bar, bg=FACE)
        inner.pack(fill="x", padx=self.px(2), pady=self.px(2))

        grip = tk.Canvas(inner, width=self.px(16), height=self.px(16), bg=FACE,
                         highlightthickness=0, bd=0, cursor="sizing")
        grip.pack(side="right", padx=(self.px(2), 0))
        s = self.S
        for off in (0, 4, 8):
            x2, y2 = int((15) * s), int((15 - off) * s)
            x1, y1 = int((15 - off) * s), int(15 * s)
            grip.create_line(x1, int(15 * s), int(15 * s), y2, fill=HILIGHT, width=self.px(1))
            grip.create_line(x1 + self.px(1), int(15 * s), int(15 * s), y2 + self.px(1), fill=SHADOW, width=self.px(1))
        grip.bind("<ButtonPress-1>", self._resize_start)
        grip.bind("<B1-Motion>", self._resize_move)

        p1_border, p1 = self._bevel(inner, style="sunken", bg=FACE)
        p1_border.pack(side="left", fill="x", expand=True)
        self.status_left = tk.Label(p1, text="Ready", bg=FACE, fg=BLACK, font=FONT_UI,
                                    anchor="w", padx=self.px(4), pady=self.px(1))
        self.status_left.pack(fill="x")

        p2_border, p2 = self._bevel(inner, style="sunken", bg=FACE)
        p2_border.pack(side="left", padx=(self.px(2), self.px(2)))
        self.status_files = tk.Label(p2, text="0 files", bg=FACE, fg=BLACK, font=FONT_UI,
                                     anchor="w", padx=self.px(6), pady=self.px(1), width=10)
        self.status_files.pack()

    def _drag_start(self, event):
        self._drag_offset = (event.x_root - self.root.winfo_x(),
                             event.y_root - self.root.winfo_y())

    def _drag_move(self, event):
        if self._maximized:
            return
        x = event.x_root - self._drag_offset[0]
        y = event.y_root - self._drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def _resize_start(self, event):
        if self._maximized:
            self._toggle_max()
        self._resize = (event.x_root, event.y_root, self.root.winfo_width(), self.root.winfo_height())

    def _resize_move(self, event):
        if not self._resize:
            return
        start_x, start_y, start_w, start_h = self._resize
        new_w = max(self._min_w, start_w + (event.x_root - start_x))
        new_h = max(self._min_h, start_h + (event.y_root - start_y))
        self.root.geometry(f"{new_w}x{new_h}")

    def _minimize(self):
        self.root.iconify()

    def _on_remap(self, event):
        if str(event.widget) == ".":
            self.root.after(10, self._make_borderless)

    def _toggle_max(self):
        if self._maximized:
            self.root.geometry(self._normal_geometry)
            self._maximized = False
            self._max_button.set_glyph("max")
        else:
            self._normal_geometry = self.root.geometry()
            work_left, work_top, work_right, work_bottom = self._work_area()
            self.root.geometry(f"{work_right - work_left}x{work_bottom - work_top}+{work_left}+{work_top}")
            self._maximized = True
            self._max_button.set_glyph("restore")

    def _work_area(self):
        try:
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            rect = RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            return rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            return 0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def _about(self):
        dialog = tk.Toplevel(self.root, bg=FACE)
        dialog.overrideredirect(True)
        dw, dh = self.px(320), self.px(150)
        dx = self.root.winfo_x() + (self.root.winfo_width() - dw) // 2
        dy = self.root.winfo_y() + (self.root.winfo_height() - dh) // 2
        dialog.geometry(f"{dw}x{dh}+{dx}+{dy}")

        shell_border, shell = self._bevel(dialog, style="raised", bg=FACE)
        shell_border.pack(fill="both", expand=True)

        bar = tk.Frame(shell, bg=NAVY, height=self.px(20))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="About Sotvox", bg=NAVY, fg=WHITE, font=FONT_TITLE).pack(side="left", padx=self.px(6))
        self._title_button(bar, "close", dialog.destroy).pack(side="right", padx=self.px(2), pady=self.px(2))

        body = tk.Frame(shell, bg=FACE)
        body.pack(fill="both", expand=True, padx=self.px(12), pady=self.px(12))
        tk.Label(body, text="Sotvox", bg=FACE, fg=BLACK, font=("MS Sans Serif", 14, "bold")).pack(anchor="w")
        tk.Label(body, text="Local audio / video transcription", bg=FACE, fg=BLACK,
                 font=FONT_UI).pack(anchor="w", pady=(self.px(2), 0))
        tk.Label(body, text="Powered by faster-whisper. Runs 100% on your machine.",
                 bg=FACE, fg=BLACK, font=FONT_UI).pack(anchor="w", pady=(self.px(8), 0))
        self._mk_button(body, "OK", command=dialog.destroy, width_chars=8).pack(pady=(self.px(12), 0))

        bar.bind("<ButtonPress-1>", lambda e: dialog.__setattr__("_off", (e.x_root - dialog.winfo_x(), e.y_root - dialog.winfo_y())))
        bar.bind("<B1-Motion>", lambda e: dialog.geometry(f"+{e.x_root - dialog._off[0]}+{e.y_root - dialog._off[1]}"))
        dialog.grab_set()

    def _draw_progress(self):
        c = self.progress_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 4:
            return
        c.create_rectangle(0, 0, w, h, fill=FACE, outline="")
        if self._progress_value > 0:
            margin = self.px(2)
            filled = int((w - margin * 2) * self._progress_value / 100)
            block_w = self.px(8)
            gap = self.px(2)
            x = margin
            while x + block_w <= margin + filled:
                c.create_rectangle(x, margin, x + block_w, h - margin, fill=NAVY, outline="")
                x += block_w + gap

    def _on_drop(self, event):
        raw = event.data
        paths = []
        for match in re.finditer(r'\{([^}]+)\}|(\S+)', raw):
            p = match.group(1) or match.group(2)
            if p:
                paths.append(p)
        self._slog(f"Files dropped: {len(paths)} item(s)")
        self._add_files(paths)

    def _browse_files(self):
        self._slog("Opened file browser")
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[("Media files", exts), ("All files", "*.*")]
        )
        if paths:
            self._add_files(paths)
        else:
            self._slog("File browser cancelled")

    def _browse_output(self):
        self._slog("Opened output folder browser")
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_var.set(d)
        else:
            self._slog("Output folder browser cancelled")

    def _add_files(self, paths):
        added = 0
        for p in paths:
            p = p.strip('"').strip("'")
            ext = os.path.splitext(p)[1].lower()
            if ext in SUPPORTED_EXTENSIONS and p not in self.files:
                self.files.append(p)
                self._slog(f"File added: {os.path.basename(p)}")
                self._slog(f"  Path: {p}")
                try:
                    self._slog(f"  Size: {self._fmt_size(os.path.getsize(p))}")
                except OSError:
                    pass
                added += 1
        if added:
            self._slog(f"Total files in queue: {len(self.files)}")
            self._refresh_file_list()
        elif paths:
            self._slog(f"Rejected {len(paths)} unsupported file(s)")
            self._log("No supported files found in the dropped items.")

    def _remove_file(self, path):
        if path in self.files:
            self._slog(f"File removed: {os.path.basename(path)}")
            self.files.remove(path)
            self._slog(f"Total files in queue: {len(self.files)}")
            self._refresh_file_list()

    def _remove_selected(self):
        selection = list(self.file_listbox.curselection())
        if not selection:
            return
        removed = 0
        for index in sorted(selection, reverse=True):
            if 0 <= index < len(self.files):
                name = os.path.basename(self.files[index])
                del self.files[index]
                self._slog(f"File removed: {name}")
                removed += 1
        if removed:
            self._slog(f"Total files in queue: {len(self.files)}")
            self._refresh_file_list()

    def _clear_files(self):
        count = len(self.files)
        self.files.clear()
        self._slog(f"Cleared all files ({count} removed)")
        self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_listbox.delete(0, "end")
        for path in self.files:
            name = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()
            tag = "[VID]" if ext in VIDEO_EXTENSIONS else "[AUD]"
            self.file_listbox.insert("end", f" {tag}  {name}")
        count = len(self.files)
        self.status_files.configure(text=f"{count} file{'s' if count != 1 else ''}")

    def _toggle_transcription(self):
        if self.is_transcribing:
            self._slog("Cancellation requested by user")
            self.cancel_requested = True
            self.transcribe_btn.set_text("Cancelling...")
            self.transcribe_btn.set_enabled(False)
            return

        if not self.files:
            self._log("No files to transcribe. Add some files first.")
            return

        self._slog(f"Transcription started: {len(self.files)} file(s), model={self.model_var.get()}, lang={self.language_var.get()}, multilingual={self.multilingual_var.get()}")
        self.is_transcribing = True
        self.cancel_requested = False
        self.transcribe_btn.set_text("Cancel")
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _resolve_device(self):
        choice = self.device_var.get()
        libraries_ready = gpu_pack.libraries_available()

        if choice == "CPU":
            self._slog("Device resolved: CPU (user selected)")
            return "cpu", "int8"
        elif choice == "GPU (CUDA)":
            if libraries_ready:
                self._slog("Device resolved: CUDA (user selected)")
                return "cuda", "float16"
            else:
                self._slog("GPU requested but CUDA libraries are not installed — using CPU")
                self._log("GPU acceleration is not installed yet. Using CPU — "
                          "click 'Enable GPU Acceleration...' to enable it.")
                return "cpu", "int8"
        else:
            try:
                import ctranslate2
                gpu_count = ctranslate2.get_cuda_device_count()
                self._slog(f"Auto-detect: {gpu_count} CUDA device(s), libraries_ready={libraries_ready}")
                if gpu_count > 0 and libraries_ready:
                    return "cuda", "float16"
            except Exception as e:
                self._slog(f"CUDA detection failed ({e}), defaulting to CPU")
            return "cpu", "int8"

    def _transcribe_worker(self):
        output_dir = self.output_var.get()
        os.makedirs(output_dir, exist_ok=True)
        total = len(self.files)

        self._slog(f"Output directory: {output_dir}")
        self._slog(f"Files in queue ({total}):")
        for f in self.files:
            self._slog(f"  - {f}")

        try:
            model_name = self.model_var.get()
            device, compute_type = self._resolve_device()
            self._slog(f"Loading model: name={model_name}, device={device}, compute_type={compute_type}")
            self._update_status(f"Loading model '{model_name}'...")
            self._set_progress(0)

            model_load_start = time.time()
            if self.model is None or self._loaded_model_name != model_name or self._loaded_device != device:
                try:
                    self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                except Exception as e:
                    if device == "cuda":
                        self._log(f"CUDA load failed ({e}), falling back to CPU")
                        self._slog(f"CUDA load traceback:\n{traceback.format_exc()}")
                        device, compute_type = "cpu", "int8"
                        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                    else:
                        raise
                self._loaded_model_name = model_name
                self._loaded_device = device
                self._slog(f"Model loaded in {time.time() - model_load_start:.1f}s")
            else:
                self._slog(f"Model already loaded: {model_name} on {device}")

            device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
            self._log(f"Model '{model_name}' loaded on {device_label}")

            if device == "cpu" and model_name == "large-v3":
                self._log("Note: large-v3 on CPU is slow. Try large-v3-turbo — much faster, "
                          "almost as accurate.")

            language = self.language_var.get()
            multilingual = self.multilingual_var.get()
            lang_code = None if language == "Auto-detect" else LANG_MAP.get(language)

            if multilingual:
                self._slog("Multilingual mode active — language selection will be ignored")

            succeeded = 0
            failed = 0

            for i, filepath in enumerate(self.files):
                if self.cancel_requested:
                    self._log("Transcription cancelled.")
                    break

                name = os.path.basename(filepath)
                self._update_status(f"[{i + 1}/{total}] {name}")
                self._set_progress((i / total) * 100)
                self._log(f"Processing: {name}")

                start_time = time.time()

                try:
                    audio_path = filepath
                    ext = os.path.splitext(filepath)[1].lower()

                    self._slog(f"  Full path: {filepath}")
                    try:
                        self._slog(f"  File size: {self._fmt_size(os.path.getsize(filepath))}")
                    except OSError:
                        pass

                    if ext in VIDEO_EXTENSIONS:
                        try:
                            probe = probe_file(filepath)
                            if probe:
                                fmt = probe.get("format", {})
                                self._slog(f"  Probe format: {fmt.get('format_long_name', 'unknown')}, duration={fmt.get('duration', '?')}s, bitrate={fmt.get('bit_rate', '?')} bps")
                                for stream in probe.get("streams", []):
                                    st = stream.get("codec_type", "?")
                                    cn = stream.get("codec_name", "?")
                                    if st == "video":
                                        self._slog(f"  Probe stream [video]: {cn} {stream.get('width', '?')}x{stream.get('height', '?')} @ {stream.get('r_frame_rate', '?')} fps")
                                    elif st == "audio":
                                        self._slog(f"  Probe stream [audio]: {cn} {stream.get('sample_rate', '?')} Hz, {stream.get('channels', '?')} ch")
                                    else:
                                        self._slog(f"  Probe stream [{st}]: {cn}")
                            else:
                                self._slog(f"  Probe: failed to read file metadata")
                        except Exception as e:
                            self._slog(f"  Probe error (non-fatal): {e}")

                        if not has_audio_stream(filepath):
                            raise RuntimeError("Video has no audio track — nothing to transcribe")

                    def on_progress(seg_end, duration):
                        pct = (i / total + (seg_end / duration) / total) * 100
                        self._set_progress(min(pct, 99))

                    if multilingual:
                        def on_chunk(chunk_number, total_chunks, time_start, time_end, detected_lang, probability):
                            self._slog(f"  Chunk [{chunk_number}/{total_chunks}] {time_start:.1f}s-{time_end:.1f}s: language={detected_lang} (probability={probability:.4f})")

                        self._slog("  Transcribing audio (multilingual mode, 30s chunks)...")
                        text_parts, info, status = transcribe_audio_multilingual(
                            self.model, audio_path,
                            on_progress=on_progress,
                            is_cancelled=lambda: self.cancel_requested,
                            on_chunk=on_chunk
                        )
                    else:
                        self._slog(f"  Transcribing audio (lang_code={lang_code})...")
                        text_parts, info, status = transcribe_audio(
                            self.model, audio_path, lang_code,
                            on_progress=on_progress,
                            is_cancelled=lambda: self.cancel_requested
                        )

                    self._slog(f"  Transcription result: status={status}, segments={len(text_parts)}")
                    self._slog(f"  Audio duration: {info.duration:.1f}s")
                    self._slog(f"  Detected language: {info.language} (probability: {info.language_probability:.4f})")
                    if hasattr(info, "duration_after_vad") and info.duration_after_vad:
                        self._slog(f"  Duration after VAD: {info.duration_after_vad:.1f}s")

                    if status == "timed_out":
                        self._log(f"  Timed out, skipping")
                        continue

                    if status == "retry_timed_out":
                        self._log(f"  Retry timed out, skipping")

                    if self.cancel_requested:
                        break

                    full_text = "\n".join(text_parts)

                    if not full_text.strip():
                        self._log(f"  No speech detected, skipping")
                        continue

                    saved_name = save_transcript(filepath, full_text, output_dir)
                    self._slog(f"  Text length: {len(full_text)} chars")
                    try:
                        out_path = os.path.join(output_dir, saved_name)
                        self._slog(f"  Output file: {out_path} ({self._fmt_size(os.path.getsize(out_path))})")
                    except OSError:
                        pass

                    elapsed = time.time() - start_time
                    if multilingual or language == "Auto-detect":
                        detected = info.language
                    else:
                        detected = language
                    self._log(f"  Done in {elapsed:.1f}s · {detected} · {saved_name}")
                    succeeded += 1

                except Exception as e:
                    self._log(f"  Error: {str(e)}")
                    self._slog(f"  Traceback:\n{traceback.format_exc()}")
                    failed += 1
                    continue

            if not self.cancel_requested:
                self._set_progress(100)
                if failed == 0:
                    self._update_status(f"Complete — {succeeded} file(s) transcribed")
                    winsound.PlaySound(os.path.join(SOUNDS_DIR, "success.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
                elif succeeded == 0:
                    self._update_status(f"Failed — {failed} file(s) had errors")
                    winsound.PlaySound(os.path.join(SOUNDS_DIR, "error.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    self._update_status(f"Done — {succeeded} transcribed, {failed} failed")
                    winsound.PlaySound(os.path.join(SOUNDS_DIR, "success.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
                self._log(f"All done. Output: {output_dir}")

        except Exception as e:
            self._log(f"Fatal error: {str(e)}")
            self._slog(f"Fatal traceback:\n{traceback.format_exc()}")
            self._update_status("Error occurred")
            winsound.PlaySound(os.path.join(SOUNDS_DIR, "error.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)

        finally:
            self.is_transcribing = False
            self.cancel_requested = False
            self.root.after(0, lambda: (self.transcribe_btn.set_text("Transcribe"),
                                        self.transcribe_btn.set_enabled(True)))

    def _open_output(self):
        output_dir = self.output_var.get()
        os.makedirs(output_dir, exist_ok=True)
        self._slog(f"Opened output folder: {output_dir}")
        os.startfile(output_dir)

    def _update_status(self, text):
        self._slog(f"[STATUS] {text}")
        def _update():
            self.progress_label.configure(text=text)
            self.status_left.configure(text=text)
        self.root.after(0, _update)

    def _set_progress(self, value):
        self._progress_value = value
        def _update():
            self._draw_progress()
            self.progress_pct.configure(text=f"{int(value)}%" if value > 0 else "")
        self.root.after(0, _update)

    def _log(self, text):
        self._slog(text)
        def _append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(0, _append)

    def _slog(self, text):
        self._session_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def _fmt_size(self, size_bytes):
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _init_log(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("[%d-%m-%Y] - [%H-%M-%S]")
        self._log_path = os.path.join(LOG_DIR, f"{stamp}.txt")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 60,
            f"SESSION STARTED: {now}",
            "=" * 60,
            "",
            "SYSTEM",
            f"  OS: {platform.platform()}",
            f"  Python: {sys.version}",
        ]

        try:
            import av
            import ctranslate2
            import faster_whisper
            lines.append(f"  PyAV: {av.__version__} (bundled FFmpeg libraries)")
            lines.append(f"  faster-whisper: {faster_whisper.__version__}")
            lines.append(f"  CTranslate2: {ctranslate2.__version__}")
        except Exception as e:
            lines.append(f"  Component versions unavailable ({e})")

        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0 and r.stdout.strip():
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                lines.append(f"  GPU: {parts[0]} ({float(parts[1])/1024:.1f} GB VRAM)")
                lines.append(f"  NVIDIA driver: {parts[2]}")
        except Exception:
            lines.append("  GPU: not detected")

        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            for line in r.stdout.splitlines():
                if "CUDA Version" in line:
                    lines.append(f"  CUDA: {line.split('CUDA Version:')[1].strip().rstrip('|').strip()}")
                    break
        except Exception:
            pass

        try:
            class _MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = _MEMSTAT(dwLength=ctypes.sizeof(_MEMSTAT))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            lines.append(f"  RAM: {stat.ullTotalPhys / (1024**3):.1f} GB total, {stat.ullAvailPhys / (1024**3):.1f} GB available")
        except Exception:
            pass

        lines += [
            "",
            "SETTINGS",
            f"  Model: {self.model_var.get()}",
            f"  Language: {self.language_var.get()}",
            f"  Device: {self.device_var.get()}",
            f"  Multilingual: {self.multilingual_var.get()}",
            f"  Output path: {self.output_var.get()}",
            "",
            "=" * 60,
        ]

        with open(self._log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self._slog("Application started")
        self._flush_log()

    def _flush_log(self):
        if self._log_path and self._log_flush_idx < len(self._session_log):
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    for line in self._session_log[self._log_flush_idx:]:
                        f.write(line + "\n")
                self._log_flush_idx = len(self._session_log)
            except Exception:
                pass
        self.root.after(2000, self._flush_log)

    def _on_close(self):
        elapsed = time.time() - self._session_start
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        self._slog(f"Session duration: {hours}h {minutes}m {seconds}s")
        self._slog("Application closed")
        if self._log_path:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    for line in self._session_log[self._log_flush_idx:]:
                        f.write(line + "\n")
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()
