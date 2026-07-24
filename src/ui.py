import os
import re
import sys
import glob
import time
import math
import ctypes
import random
import platform
import threading
import traceback
import subprocess
import winsound
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import tkinterdnd2 as tkdnd
from faster_whisper import WhisperModel

from constants import (
    SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, DEFAULT_OUTPUT_DIR, LANG_MAP,
    LOG_DIR, SOUNDS_DIR,
)
from engine import extract_audio, transcribe_audio, transcribe_audio_multilingual, save_transcript, probe_file


P = {
    "bg":           "#08080f",
    "elevated":     "#111119",
    "surface":      "#1a1a24",
    "border":       "#252530",
    "text":         "#f5f0eb",
    "text_sec":     "#b0a8c0",
    "text_dim":     "#5a5570",
    "accent":       "#a78bfa",
    "accent_hover": "#c4b5fd",
    "cyan":         "#67e8f9",
    "amber":        "#fbbf24",
    "red":          "#f87171",
    "red_hover":    "#fca5a5",
    "log_bg":       "#0a0a12",
    "log_fg":       "#8b85a0",
    "entry_bg":     "#0e0e16",
    "entry_fg":     "#d8d0e8",
}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r, g, b):
    return f"#{max(0,min(255,int(r))):02x}{max(0,min(255,int(g))):02x}{max(0,min(255,int(b))):02x}"


def _lerp(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def _ease(t):
    return t * t * (3 - 2 * t)


class _ThumbScrollbar(tk.Canvas):
    def __init__(self, parent, command=None, track_color=None, **kwargs):
        super().__init__(parent, width=10, highlightthickness=0, bd=0,
                         bg=track_color or P["elevated"], **kwargs)
        self._command = command
        self._track_color = track_color or P["elevated"]
        self._lo = 0.0
        self._hi = 1.0
        self._dragging = False
        self._drag_y = 0
        self._drag_pos = 0.0
        self._hovered = False

        self.bind("<Configure>", self._draw)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: (setattr(self, "_hovered", True), self._draw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hovered", False), self._draw()))

    def set(self, lo, hi):
        self._lo = float(lo)
        self._hi = float(hi)
        self._draw()

    def _draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if h < 1 or self._hi - self._lo >= 1.0:
            return
        thumb_h = max(24, int(h * (self._hi - self._lo)))
        thumb_y = int((h - thumb_h) * self._lo / max(0.001, 1.0 - (self._hi - self._lo)))
        pad_x = 2
        r = 3
        color = P["text_dim"] if (self._hovered or self._dragging) else P["border"]
        x0, y0, x1, y1 = pad_x, thumb_y, w - pad_x, thumb_y + thumb_h
        self.create_rectangle(x0 + r, y0, x1 - r, y1, fill=color, outline="")
        self.create_rectangle(x0, y0 + r, x1, y1 - r, fill=color, outline="")
        self.create_oval(x0, y0, x0 + r * 2, y0 + r * 2, fill=color, outline="")
        self.create_oval(x1 - r * 2, y0, x1, y0 + r * 2, fill=color, outline="")
        self.create_oval(x0, y1 - r * 2, x0 + r * 2, y1, fill=color, outline="")
        self.create_oval(x1 - r * 2, y1 - r * 2, x1, y1, fill=color, outline="")

    def _on_press(self, event):
        h = self.winfo_height()
        if h < 1 or self._hi - self._lo >= 1.0:
            return
        thumb_h = max(24, int(h * (self._hi - self._lo)))
        thumb_y = int((h - thumb_h) * self._lo / max(0.001, 1.0 - (self._hi - self._lo)))
        if thumb_y <= event.y <= thumb_y + thumb_h:
            self._dragging = True
            self._drag_y = event.y
            self._drag_pos = self._lo
        else:
            frac = event.y / h
            self._command("moveto", str(max(0.0, min(1.0, frac))))

    def _on_drag(self, event):
        if not self._dragging:
            return
        h = self.winfo_height()
        if h < 1:
            return
        dy = event.y - self._drag_y
        span = self._hi - self._lo
        delta = dy / max(1, h - max(24, int(h * span)))
        new_pos = max(0.0, min(1.0 - span, self._drag_pos + delta * (1.0 - span)))
        self._command("moveto", str(new_pos))

    def _on_release(self, event):
        self._dragging = False
        self._draw()


class TranscriberApp:
    def __init__(self):
        self.root = tkdnd.Tk()
        self.root.title("Whisper Transcriber")
        _s = self.root.winfo_fpixels('1i') / 96.0
        self.root.geometry(f"{int(840 * _s)}x{int(800 * _s)}")
        self.root.minsize(int(700 * _s), int(660 * _s))
        self.root.configure(bg=P["bg"])

        self.files = []
        self.model = None
        self._loaded_model_name = None
        self._loaded_device = None
        self.is_transcribing = False
        self.cancel_requested = False
        self._progress_value = 0
        self._tick_count = 0
        self._shimmer_phase = 0.0
        self._particles = []
        self._waves = []
        self._session_log = []
        self._log_flush_idx = 0
        self._log_path = None
        self._session_start = time.time()

        self.language_var = tk.StringVar(value="Spanish")
        self.model_var = tk.StringVar(value="large-v3")
        self.device_var = tk.StringVar(value="Auto")
        self.multilingual_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)

        self._build_ui()
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

        self._init_log()
        self.language_var.trace_add("write", lambda *_: self._slog(f"Setting changed: language = {self.language_var.get()}"))
        self.model_var.trace_add("write", lambda *_: self._slog(f"Setting changed: model = {self.model_var.get()}"))
        self.device_var.trace_add("write", lambda *_: self._slog(f"Setting changed: device = {self.device_var.get()}"))
        self.multilingual_var.trace_add("write", lambda *_: self._slog(f"Setting changed: multilingual = {self.multilingual_var.get()}"))
        self.output_var.trace_add("write", lambda *_: self._slog(f"Setting changed: output path = {self.output_var.get()}"))

    def _hover_bind(self, widget, bg_from, bg_to):
        anim = [0.0, 0, None]

        def step():
            if anim[2]:
                widget.after_cancel(anim[2])
                anim[2] = None
            anim[0] = max(0.0, min(1.0, anim[0] + 0.1 * anim[1]))
            color = _lerp(bg_from, bg_to, _ease(anim[0]))
            widget.configure(bg=color, activebackground=bg_to)
            if 0.0 < anim[0] < 1.0:
                anim[2] = widget.after(16, step)

        widget.bind("<Enter>", lambda e: (anim.__setitem__(1, 1), step()))
        widget.bind("<Leave>", lambda e: (anim.__setitem__(1, -1), step()))

    def _dropdown(self, parent, variable, values, width=16):
        outer = tk.Frame(parent, bg=P["entry_bg"], highlightbackground=P["border"],
                         highlightthickness=1, cursor="hand2")
        lbl = tk.Label(outer, textvariable=variable, font=("Bahnschrift", 10),
                       bg=P["entry_bg"], fg=P["entry_fg"], anchor="w",
                       width=width, padx=8, pady=4, cursor="hand2")
        lbl.pack(side="left", fill="both", expand=True)
        arrow = tk.Label(outer, text="\u25be", font=("Bahnschrift", 11),
                         bg=P["entry_bg"], fg=P["text_dim"], padx=6, cursor="hand2")
        arrow.pack(side="right")
        menu = tk.Menu(outer, tearoff=0, bg=P["surface"], fg=P["text_sec"],
                       activebackground=P["accent"], activeforeground=P["text"],
                       font=("Bahnschrift", 10), borderwidth=1, relief="solid")
        for val in values:
            menu.add_command(label=val, command=lambda v=val: variable.set(v))

        def show(e=None):
            menu.post(outer.winfo_rootx(), outer.winfo_rooty() + outer.winfo_height())

        for w in (outer, lbl, arrow):
            w.bind("<Button-1>", show)
        outer.bind("<Enter>", lambda e: outer.configure(highlightbackground=P["accent"]))
        outer.bind("<Leave>", lambda e: outer.configure(highlightbackground=P["border"]))
        return outer

    def _toggle(self, parent, variable, label_text, hint_text=None):
        outer = tk.Frame(parent, bg=P["elevated"], cursor="hand2")

        box = tk.Canvas(outer, width=32, height=18, bg=P["elevated"],
                        highlightthickness=0, cursor="hand2")
        box.pack(side="left", padx=(0, 8))

        label = tk.Label(outer, text=label_text, font=("Bahnschrift", 10),
                         bg=P["elevated"], fg=P["text_sec"], cursor="hand2")
        label.pack(side="left")

        state = {"hovered": False}

        def draw():
            box.delete("all")
            is_on = variable.get()
            canvas_width = 32
            canvas_height = 18
            if is_on:
                track_color = P["accent_hover"] if state["hovered"] else P["accent"]
                knob_color = P["bg"]
            else:
                track_color = P["text_dim"] if state["hovered"] else P["border"]
                knob_color = P["text"]
            corner_radius = canvas_height // 2
            box.create_oval(0, 0, canvas_height, canvas_height, fill=track_color, outline="")
            box.create_oval(canvas_width - canvas_height, 0, canvas_width, canvas_height,
                            fill=track_color, outline="")
            box.create_rectangle(corner_radius, 0, canvas_width - corner_radius, canvas_height,
                                 fill=track_color, outline="")
            knob_diameter = canvas_height - 4
            knob_x = (canvas_width - knob_diameter - 2) if is_on else 2
            box.create_oval(knob_x, 2, knob_x + knob_diameter, 2 + knob_diameter,
                            fill=knob_color, outline="")

        def on_enter(e):
            state["hovered"] = True
            draw()

        def on_leave(e):
            state["hovered"] = False
            draw()

        def on_click(e=None):
            variable.set(not variable.get())

        widgets_to_bind = [outer, label, box]

        if hint_text:
            hint = tk.Label(outer, text=hint_text, font=("Bahnschrift Light", 9),
                            bg=P["elevated"], fg=P["text_dim"], cursor="hand2")
            hint.pack(side="left", padx=(8, 0))
            widgets_to_bind.append(hint)

        for widget in widgets_to_bind:
            widget.bind("<Button-1>", on_click)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        variable.trace_add("write", lambda *_: draw())
        draw()
        return outer

    def _build_ui(self):
        main = tk.Frame(self.root, bg=P["bg"])
        main.pack(fill="both", expand=True, padx=26, pady=(0, 20))

        self._build_header(main)
        self._build_drop_zone(main)
        self._build_file_list(main)
        self._build_settings(main)
        self._build_action_bar(main)
        self._build_progress(main)

    def _build_header(self, parent):
        self.header_canvas = tk.Canvas(parent, bg=P["bg"], highlightthickness=0, height=72)
        self.header_canvas.pack(fill="x", pady=(14, 14))

        wave_alphas = [0.07, 0.10, 0.05]
        for a in wave_alphas:
            color = _lerp(P["bg"], P["accent"], a)
            line = self.header_canvas.create_line(0, 36, 1, 36, fill=color, width=1.5, smooth=True)
            self._waves.append(line)

        self.header_canvas.create_text(
            26, 16, text="WHISPER TRANSCRIBER", anchor="w",
            fill=P["text"], font=("Bahnschrift SemiBold", 18)
        )
        self.header_canvas.create_text(
            26, 54, text="Drag & drop audio/video files to transcribe them locally using AI",
            anchor="w", fill=P["text_dim"], font=("Bahnschrift Light", 9)
        )

    def _build_drop_zone(self, parent):
        self.drop_canvas = tk.Canvas(parent, bg=P["surface"], highlightthickness=0,
                                      height=90, cursor="hand2")
        self.drop_canvas.pack(fill="x", pady=(0, 12))

        self._init_particles()

        self.drop_canvas.create_text(0, 0, text="", tags="drop_main",
                                      fill=P["text_dim"], font=("Bahnschrift", 11))
        self.drop_canvas.create_text(0, 0, text="", tags="drop_sub",
                                      fill=P["text_dim"], font=("Bahnschrift Light", 9))

        self.drop_canvas.bind("<Configure>", self._redraw_drop_zone)
        self.drop_canvas.bind("<Enter>", lambda e: self._redraw_drop_zone(None, hover=True))
        self.drop_canvas.bind("<Leave>", lambda e: self._redraw_drop_zone(None, hover=False))
        self.drop_canvas.bind("<Button-1>", lambda e: self._browse_files())

        self.drop_canvas.drop_target_register(tkdnd.DND_FILES)
        self.drop_canvas.dnd_bind("<<Drop>>", self._on_drop)
        self._drop_hover = False

    def _init_particles(self):
        self._particles = []
        for _ in range(18):
            self._particles.append({
                "x": random.random(), "y": random.random(),
                "dx": random.uniform(-0.0008, 0.0008),
                "dy": random.uniform(-0.0008, 0.0008),
                "r": random.uniform(1.5, 3.0),
                "phase": random.uniform(0, math.pi * 2),
                "id": None,
            })

    def _create_particle_items(self):
        for p in self._particles:
            if p["id"] is None:
                p["id"] = self.drop_canvas.create_oval(0, 0, 1, 1, fill=P["surface"], outline="")
                self.drop_canvas.tag_lower(p["id"])

    def _redraw_drop_zone(self, event=None, hover=None):
        if hover is not None:
            self._drop_hover = hover
        w = self.drop_canvas.winfo_width()
        h = self.drop_canvas.winfo_height()
        if w < 10:
            return
        text_color = P["accent"] if self._drop_hover else P["text_dim"]
        self.drop_canvas.itemconfigure("drop_main", fill=text_color,
                                        text="Drop audio or video files here")
        self.drop_canvas.coords("drop_main", w // 2, h // 2 - 10)
        sub_color = P["text_sec"] if self._drop_hover else _lerp(P["text_dim"], P["bg"], 0.3)
        self.drop_canvas.itemconfigure("drop_sub", fill=sub_color,
                                        text="or click to browse")
        self.drop_canvas.coords("drop_sub", w // 2, h // 2 + 14)
        self._create_particle_items()

    def _build_file_list(self, parent):
        list_frame = tk.Frame(parent, bg=P["elevated"])
        list_frame.pack(fill="both", expand=True, pady=(0, 12))

        header = tk.Frame(list_frame, bg=P["elevated"])
        header.pack(fill="x", padx=(14, 44), pady=(10, 0))
        self.file_count_label = tk.Label(header, text="Files (0)", bg=P["elevated"],
                                          fg=P["text"], font=("Bahnschrift SemiBold", 10))
        self.file_count_label.pack(side="left")

        clear_frame = tk.Frame(header, bg=P["border"], padx=1, pady=1)
        clear_frame.pack(side="right")
        clear_btn = tk.Button(clear_frame, text="Clear all", font=("Bahnschrift", 9),
                              bg=P["surface"], fg=P["text_sec"],
                              activebackground=P["border"], activeforeground=P["accent"],
                              borderwidth=0, relief="flat", cursor="hand2",
                              padx=14, pady=7, command=self._clear_files)
        clear_btn.pack()
        self._hover_bind(clear_btn, P["surface"], P["border"])

        canvas_frame = tk.Frame(list_frame, bg=P["elevated"])
        canvas_frame.pack(fill="both", expand=True, padx=(14, 6), pady=(6, 12))

        self.file_canvas = tk.Canvas(canvas_frame, bg=P["elevated"], highlightthickness=0,
                                     borderwidth=0, height=120)
        scrollbar = _ThumbScrollbar(canvas_frame, command=self.file_canvas.yview,
                                    track_color=P["elevated"])
        self.file_inner = tk.Frame(self.file_canvas, bg=P["elevated"])

        self.file_inner.bind("<Configure>",
                             lambda e: self.file_canvas.configure(scrollregion=self.file_canvas.bbox("all")))
        self._file_window_id = self.file_canvas.create_window((0, 0), window=self.file_inner, anchor="nw")
        self.file_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y", padx=(4, 6))
        self.file_canvas.pack(side="left", fill="both", expand=True)

        self.file_canvas.bind("<Configure>", self._on_file_canvas_configure)
        self.file_canvas.bind("<Enter>", lambda e: self.file_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.file_canvas.bind("<Leave>", lambda e: self.file_canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self.file_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_file_canvas_configure(self, event):
        self.file_canvas.itemconfigure(self._file_window_id, width=event.width)

    def _build_settings(self, parent):
        settings = tk.Frame(parent, bg=P["elevated"])
        settings.pack(fill="x", pady=(0, 12))

        inner = tk.Frame(settings, bg=P["elevated"])
        inner.pack(fill="x", padx=14, pady=12)

        row1 = tk.Frame(inner, bg=P["elevated"])
        row1.pack(fill="x", pady=(0, 8))

        tk.Label(row1, text="Language", bg=P["elevated"], fg=P["text_sec"],
                 font=("Bahnschrift", 10)).pack(side="left")
        self._dropdown(row1, self.language_var,
                       ["Auto-detect", "Spanish", "English", "Portuguese", "French",
                        "German", "Italian", "Japanese", "Chinese", "Korean"],
                       width=14).pack(side="left", padx=(8, 28))

        tk.Label(row1, text="Model", bg=P["elevated"], fg=P["text_sec"],
                 font=("Bahnschrift", 10)).pack(side="left")
        self._dropdown(row1, self.model_var,
                       ["large-v3", "medium", "small", "base", "tiny"],
                       width=14).pack(side="left", padx=(8, 28))

        tk.Label(row1, text="Device", bg=P["elevated"], fg=P["text_sec"],
                 font=("Bahnschrift", 10)).pack(side="left")
        self._dropdown(row1, self.device_var,
                       ["Auto", "CPU", "GPU (CUDA)"],
                       width=12).pack(side="left", padx=(8, 0))

        row_mid = tk.Frame(inner, bg=P["elevated"])
        row_mid.pack(fill="x", pady=(0, 8))
        self._toggle(row_mid, self.multilingual_var, "Multilingual mode",
                     hint_text="auto-detect language per 30s chunk; overrides Language setting").pack(side="left")

        row2 = tk.Frame(inner, bg=P["elevated"])
        row2.pack(fill="x")

        tk.Label(row2, text="Output path", bg=P["elevated"], fg=P["text_sec"],
                 font=("Bahnschrift", 10)).pack(side="left")
        output_entry = tk.Entry(row2, textvariable=self.output_var, font=("Bahnschrift", 10),
                                bg=P["entry_bg"], fg=P["entry_fg"], insertbackground=P["accent"],
                                relief="flat", highlightbackground=P["border"], highlightthickness=1)
        output_entry.pack(side="left", padx=(8, 8), fill="x", expand=True, ipady=4)

        browse_frame = tk.Frame(row2, bg=P["border"], padx=1, pady=1)
        browse_frame.pack(side="left")
        browse_btn = tk.Button(browse_frame, text="Browse", font=("Bahnschrift", 9),
                               bg=P["surface"], fg=P["text_sec"],
                               activebackground=P["border"], activeforeground=P["accent"],
                               borderwidth=0, relief="flat", cursor="hand2",
                               padx=14, pady=7, command=self._browse_output)
        browse_btn.pack()
        self._hover_bind(browse_btn, P["surface"], P["border"])

    def _build_action_bar(self, parent):
        action = tk.Frame(parent, bg=P["bg"])
        action.pack(fill="x", pady=(0, 12))

        self.transcribe_btn = tk.Button(
            action, text="\u25b6  TRANSCRIBE", font=("Bahnschrift SemiBold", 11),
            bg=P["accent"], fg=P["bg"], activebackground=P["accent_hover"],
            activeforeground=P["bg"], borderwidth=0, cursor="hand2",
            padx=32, pady=11, relief="flat"
        )
        self.transcribe_btn.configure(command=self._toggle_transcription)
        self.transcribe_btn.pack(side="left")

        self._btn_anim = [0.0, 0, None]

        def btn_step():
            if self._btn_anim[2]:
                self.transcribe_btn.after_cancel(self._btn_anim[2])
                self._btn_anim[2] = None
            self._btn_anim[0] = max(0.0, min(1.0, self._btn_anim[0] + 0.1 * self._btn_anim[1]))
            text = self.transcribe_btn.cget("text")
            if "TRANSCRIBE" in text:
                color = _lerp(P["accent"], P["accent_hover"], _ease(self._btn_anim[0]))
                self.transcribe_btn.configure(bg=color)
            elif "CANCEL" in text and "CANCELLING" not in text:
                color = _lerp(P["red"], P["red_hover"], _ease(self._btn_anim[0]))
                self.transcribe_btn.configure(bg=color)
            if 0.0 < self._btn_anim[0] < 1.0:
                self._btn_anim[2] = self.transcribe_btn.after(16, btn_step)

        self.transcribe_btn.bind("<Enter>", lambda e: (self._btn_anim.__setitem__(1, 1), btn_step()))
        self.transcribe_btn.bind("<Leave>", lambda e: (self._btn_anim.__setitem__(1, -1), btn_step()))

        open_frame = tk.Frame(action, bg=P["border"], padx=1, pady=1)
        open_frame.pack(side="right")
        open_btn = tk.Button(open_frame, text="Open output \u2192", font=("Bahnschrift", 9),
                             bg=P["surface"], fg=P["text_sec"],
                             activebackground=P["border"], activeforeground=P["accent"],
                             borderwidth=0, relief="flat", cursor="hand2",
                             padx=14, pady=7, command=self._open_output)
        open_btn.pack()
        self._hover_bind(open_btn, P["surface"], P["border"])

    def _build_progress(self, parent):
        prog_frame = tk.Frame(parent, bg=P["elevated"])
        prog_frame.pack(fill="both", expand=True)

        inner = tk.Frame(prog_frame, bg=P["elevated"])
        inner.pack(fill="both", expand=True, padx=14, pady=12)

        status_row = tk.Frame(inner, bg=P["elevated"])
        status_row.pack(fill="x")

        self.progress_label = tk.Label(status_row, text="Ready", bg=P["elevated"],
                                        fg=P["text_sec"], font=("Bahnschrift", 9), anchor="w")
        self.progress_label.pack(side="left")

        self.progress_pct = tk.Label(status_row, text="", bg=P["elevated"],
                                      fg=P["text_dim"], font=("Bahnschrift", 9))
        self.progress_pct.pack(side="right")

        self.progress_canvas = tk.Canvas(inner, bg=P["surface"], highlightthickness=0, height=5)
        self.progress_canvas.pack(fill="x", pady=(8, 10))

        log_frame = tk.Frame(inner, bg=P["log_bg"], highlightbackground=P["border"],
                             highlightthickness=1)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, bg=P["log_bg"], fg=P["log_fg"],
                                font=("Cascadia Mono", 9), height=7, wrap="word",
                                state="disabled", borderwidth=6, relief="flat",
                                highlightthickness=0, insertbackground=P["accent"],
                                selectbackground=P["accent"], selectforeground=P["text"])
        log_scroll = _ThumbScrollbar(log_frame, command=self.log_text.yview,
                                     track_color=P["log_bg"])
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y", padx=(4, 6))
        self.log_text.pack(side="left", fill="both", expand=True)

    def _tick(self):
        self._tick_count += 1
        self._animate_waveform()
        self._animate_particles()
        if self.is_transcribing and self._progress_value > 0:
            self._shimmer_phase = (self._shimmer_phase + 0.025) % 1.0
            self._draw_progress()
        self.root.after(33, self._tick)

    def _animate_waveform(self):
        w = self.header_canvas.winfo_width()
        if w < 20:
            return
        t = self._tick_count * 0.04
        configs = [
            (14, 0.012, 1.0),
            (10, 0.018, 1.4),
            (8, 0.008, 0.6),
        ]
        for i, line_id in enumerate(self._waves):
            amp, freq, speed = configs[i]
            phase = t * speed
            points = []
            for x in range(0, w, 5):
                y = 36 + math.sin(x * freq + phase) * amp + math.sin(x * freq * 2.3 + phase * 0.7) * amp * 0.3
                points.extend([x, y])
            if len(points) >= 4:
                self.header_canvas.coords(line_id, *points)

    def _animate_particles(self):
        w = self.drop_canvas.winfo_width()
        h = self.drop_canvas.winfo_height()
        if w < 10 or h < 10:
            return
        t = self._tick_count * 0.03
        for p in self._particles:
            if p["id"] is None:
                continue
            p["x"] += p["dx"]
            p["y"] += p["dy"]
            if p["x"] < 0: p["x"] += 1
            if p["x"] > 1: p["x"] -= 1
            if p["y"] < 0: p["y"] += 1
            if p["y"] > 1: p["y"] -= 1
            brightness = 0.12 + 0.08 * math.sin(t + p["phase"])
            color = _lerp(P["surface"], P["accent"], brightness)
            px, py, r = p["x"] * w, p["y"] * h, p["r"]
            self.drop_canvas.coords(p["id"], px - r, py - r, px + r, py + r)
            self.drop_canvas.itemconfig(p["id"], fill=color)

    def _draw_progress(self):
        c = self.progress_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 4:
            return
        c.create_rectangle(0, 0, w, h, fill=P["surface"], outline="")
        if self._progress_value > 0:
            fill_w = max(2, int(w * self._progress_value / 100))
            c.create_rectangle(0, 0, fill_w, h, fill=P["accent"], outline="")
            band_w = 60
            band_pos = self._shimmer_phase * (fill_w + band_w) - band_w
            bx0 = max(0, int(band_pos))
            bx1 = min(fill_w, int(band_pos + band_w))
            if bx1 > bx0:
                mid = (bx0 + bx1) / 2
                half = (bx1 - bx0) / 2
                for i in range(3):
                    frac = 1.0 - i * 0.3
                    sx0 = int(mid - half * frac)
                    sx1 = int(mid + half * frac)
                    alpha = 0.4 - i * 0.12
                    shimmer_color = _lerp(P["accent"], P["cyan"], alpha)
                    c.create_rectangle(sx0, 0, sx1, h, fill=shimmer_color, outline="")

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

    def _clear_files(self):
        count = len(self.files)
        self.files.clear()
        self._slog(f"Cleared all files ({count} removed)")
        self._refresh_file_list()

    def _refresh_file_list(self):
        for widget in self.file_inner.winfo_children():
            widget.destroy()

        self.file_count_label.configure(text=f"Files ({len(self.files)})")

        for idx, path in enumerate(self.files):
            bg = P["elevated"] if idx % 2 == 0 else _lerp(P["elevated"], P["surface"], 0.3)
            row = tk.Frame(self.file_inner, bg=bg)
            row.pack(fill="x", pady=0)

            name = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()
            is_video = ext in VIDEO_EXTENSIONS
            tag = "VID" if is_video else "AUD"
            color = P["amber"] if is_video else P["cyan"]

            tk.Label(row, text=tag, bg=bg, fg=color,
                     font=("Cascadia Mono", 7, "bold")).pack(side="left", padx=(10, 10), pady=5)

            tk.Label(row, text=name, bg=bg, fg=P["text_sec"],
                     font=("Bahnschrift", 10), anchor="w").pack(side="left", fill="x", expand=True)

            remove_btn = tk.Button(row, text="\u00d7", bg=bg, fg=P["text_dim"],
                                   font=("Bahnschrift", 14), borderwidth=0, cursor="hand2",
                                   activebackground=bg, activeforeground=P["red"],
                                   padx=6, pady=2,
                                   command=lambda p=path: self._remove_file(p))
            remove_btn.pack(side="right", padx=(0, 10))

    def _toggle_transcription(self):
        if self.is_transcribing:
            self._slog("Cancellation requested by user")
            self.cancel_requested = True
            self.transcribe_btn.configure(text="CANCELLING...", bg=P["text_dim"],
                                           activebackground=P["text_dim"], state="disabled")
            return

        if not self.files:
            self._log("No files to transcribe. Add some files first.")
            return

        self._slog(f"Transcription started: {len(self.files)} file(s), model={self.model_var.get()}, lang={self.language_var.get()}, multilingual={self.multilingual_var.get()}")
        self.is_transcribing = True
        self.cancel_requested = False
        self.transcribe_btn.configure(text="\u25a0  CANCEL", bg=P["red"],
                                       activebackground=P["red_hover"])
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _resolve_device(self):
        choice = self.device_var.get()
        if choice == "CPU":
            self._slog("Device resolved: CPU (user selected)")
            return "cpu", "int8"
        if choice == "GPU (CUDA)":
            self._slog("Device resolved: CUDA (user selected)")
            return "cuda", "float16"
        try:
            import ctranslate2
            gpu_count = ctranslate2.get_cuda_device_count()
            self._slog(f"Auto-detect: {gpu_count} CUDA device(s) found")
            if gpu_count > 0:
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

            if device == "cpu" and model_name in ("large-v3",):
                self._log("Note: large-v3 on CPU may be slow. Consider small or medium.")

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
                    temp_audio = None

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

                        self._log(f"  Extracting audio from video...")
                        temp_audio = extract_audio(filepath, output_dir, log=self._slog)
                        audio_path = temp_audio

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
                        if temp_audio and os.path.exists(temp_audio):
                            os.remove(temp_audio)
                        continue

                    if status == "retry_timed_out":
                        self._log(f"  Retry timed out, skipping")

                    if self.cancel_requested:
                        if temp_audio and os.path.exists(temp_audio):
                            os.remove(temp_audio)
                        break

                    full_text = "\n".join(text_parts)

                    if not full_text.strip():
                        self._log(f"  No speech detected, skipping")
                        if temp_audio and os.path.exists(temp_audio):
                            os.remove(temp_audio)
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
                    self._log(f"  Done in {elapsed:.1f}s \u00b7 {detected} \u00b7 {saved_name}")
                    succeeded += 1

                    if temp_audio and os.path.exists(temp_audio):
                        os.remove(temp_audio)

                except Exception as e:
                    self._log(f"  Error: {str(e)}")
                    self._slog(f"  Traceback:\n{traceback.format_exc()}")
                    failed += 1
                    continue

            if not self.cancel_requested:
                self._set_progress(100)
                if failed == 0:
                    self._update_status(f"Complete \u2014 {succeeded} file(s) transcribed")
                    winsound.PlaySound(os.path.join(SOUNDS_DIR, "success.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
                elif succeeded == 0:
                    self._update_status(f"Failed \u2014 {failed} file(s) had errors")
                    winsound.PlaySound(os.path.join(SOUNDS_DIR, "error.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    self._update_status(f"Done \u2014 {succeeded} transcribed, {failed} failed")
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
            self.root.after(0, lambda: self.transcribe_btn.configure(
                text="\u25b6  TRANSCRIBE", bg=P["accent"],
                activebackground=P["accent_hover"], state="normal"))

    def _open_output(self):
        output_dir = self.output_var.get()
        os.makedirs(output_dir, exist_ok=True)
        self._slog(f"Opened output folder: {output_dir}")
        os.startfile(output_dir)

    def _update_status(self, text):
        self._slog(f"[STATUS] {text}")
        self.root.after(0, lambda: self.progress_label.configure(text=text))

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
            r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            lines.append(f"  FFmpeg: {r.stdout.splitlines()[0]}")
        except Exception as e:
            lines.append(f"  FFmpeg: not found ({e})")

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
