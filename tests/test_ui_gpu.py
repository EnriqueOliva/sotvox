"""Unit tests for the SotvoxApp helpers that marshal work onto the Tk thread,
play native notification sounds and keep the GPU status row honest."""

import queue
import time
import tkinter
import winsound

import pytest
from doubles import FakeRoot, make_app

import ui

GPU_NAME = "NVIDIA GeForce RTX 4090"


class FakeWinsound:
    MB_ICONASTERISK = winsound.MB_ICONASTERISK
    MB_ICONHAND = winsound.MB_ICONHAND

    def __init__(self, error=None):
        self.error = error
        self.beeps = []

    def MessageBeep(self, sound):
        self.beeps.append(sound)
        if self.error is not None:
            raise self.error


class RecordingCallback:
    def __init__(self, name="callback", journal=None):
        self.name = name
        self.journal = journal
        self.invocations = 0

    def __call__(self):
        self.invocations += 1
        if self.journal is not None:
            self.journal.append(self.name)


class CallbackThatRaises:
    def __init__(self, error):
        self.error = error

    def __call__(self):
        raise self.error


class RootThatRefusesToSchedule(FakeRoot):
    def __init__(self, error):
        super().__init__()
        self.error = error
        self.schedule_attempts = 0

    def after(self, delay, callback=None):
        self.schedule_attempts += 1
        raise self.error


class RecordingLibraryProbe:
    def __init__(self, available):
        self.available = available
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        return self.available


class RecordingGpuDetector:
    def __init__(self, name=None, error=None):
        self.name = name
        self.error = error
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.name


def session_log_text(app):
    return "\n".join(app._session_log)


def install_library_probe(monkeypatch, available):
    probe = RecordingLibraryProbe(available)
    monkeypatch.setattr(ui.gpu_pack, "libraries_available", probe)
    return probe


def install_gpu_detector(monkeypatch, name=None, error=None):
    detector = RecordingGpuDetector(name=name, error=error)
    monkeypatch.setattr(ui.gpu_pack, "detect_nvidia_gpu", detector)
    return detector


def install_winsound(monkeypatch, error=None):
    fake = FakeWinsound(error=error)
    monkeypatch.setattr(ui, "winsound", fake)
    return fake


def app_with_deferred_event_loop():
    return make_app(root=FakeRoot(run_immediate=False))


class TestSoundConstants:
    def test_success_sound_is_the_windows_asterisk_beep(self):
        assert ui.SOUND_SUCCESS == winsound.MB_ICONASTERISK
        assert ui.SOUND_SUCCESS == 0x00000040

    def test_failure_sound_is_the_windows_critical_stop_beep(self):
        assert ui.SOUND_FAILURE == winsound.MB_ICONHAND
        assert ui.SOUND_FAILURE == 0x00000010

    def test_success_and_failure_are_audibly_different(self):
        assert ui.SOUND_SUCCESS != ui.SOUND_FAILURE


class TestNotify:
    @pytest.mark.parametrize("sound", [ui.SOUND_SUCCESS, ui.SOUND_FAILURE])
    def test_plays_the_requested_system_sound(self, app, monkeypatch, sound):
        fake = install_winsound(monkeypatch)

        app._notify(sound)

        assert fake.beeps == [sound]

    def test_passes_arbitrary_message_beep_flags_straight_through(self, app, monkeypatch):
        fake = install_winsound(monkeypatch)

        app._notify(winsound.MB_ICONEXCLAMATION)

        assert fake.beeps == [winsound.MB_ICONEXCLAMATION]

    def test_each_call_plays_its_own_sound(self, app, monkeypatch):
        fake = install_winsound(monkeypatch)

        app._notify(ui.SOUND_SUCCESS)
        app._notify(ui.SOUND_FAILURE)
        app._notify(ui.SOUND_SUCCESS)

        assert fake.beeps == [ui.SOUND_SUCCESS, ui.SOUND_FAILURE, ui.SOUND_SUCCESS]

    def test_writes_no_breadcrumb_when_the_sound_plays(self, app, monkeypatch):
        install_winsound(monkeypatch)

        app._notify(ui.SOUND_SUCCESS)

        assert app._session_log == []

    def test_returns_none(self, app, monkeypatch):
        install_winsound(monkeypatch)

        assert app._notify(ui.SOUND_SUCCESS) is None

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("no audio device"),
            OSError("winmm.dll is missing"),
            ValueError("bad flag"),
        ],
    )
    def test_a_failing_beep_never_reaches_the_caller(self, app, monkeypatch, error):
        install_winsound(monkeypatch, error=error)

        assert app._notify(ui.SOUND_FAILURE) is None

    def test_records_the_beep_failure_in_the_session_log(self, app, monkeypatch):
        install_winsound(monkeypatch, error=RuntimeError("no audio device"))

        app._notify(ui.SOUND_SUCCESS)

        assert app._session_log == [app._session_log[0]]
        assert "Could not play notification sound: no audio device" in session_log_text(app)

    def test_the_attempt_is_still_made_before_the_failure_is_recorded(self, app, monkeypatch):
        fake = install_winsound(monkeypatch, error=RuntimeError("no audio device"))

        app._notify(ui.SOUND_FAILURE)

        assert fake.beeps == [ui.SOUND_FAILURE]

    def test_a_failing_beep_is_not_shown_to_the_user(self, app, monkeypatch):
        install_winsound(monkeypatch, error=RuntimeError("no audio device"))

        app._notify(ui.SOUND_SUCCESS)

        assert app.log_text.value == ""

    def test_a_second_failure_is_recorded_again(self, app, monkeypatch):
        install_winsound(monkeypatch, error=RuntimeError("no audio device"))

        app._notify(ui.SOUND_SUCCESS)
        app._notify(ui.SOUND_FAILURE)

        assert len(app._session_log) == 2

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit()])
    def test_interpreter_level_interrupts_are_not_swallowed(self, app, monkeypatch, error):
        install_winsound(monkeypatch, error=error)

        with pytest.raises(type(error)):
            app._notify(ui.SOUND_SUCCESS)


class TestPost:
    def test_schedules_the_callback_with_a_zero_delay(self):
        application = app_with_deferred_event_loop()
        callback = RecordingCallback()

        application._post(callback)

        assert application.root.after_calls == [(0, callback)]

    def test_does_not_run_the_callback_itself(self):
        application = app_with_deferred_event_loop()
        callback = RecordingCallback()

        application._post(callback)

        assert callback.invocations == 0

    def test_the_scheduled_callback_runs_when_the_event_loop_turns(self):
        application = app_with_deferred_event_loop()
        callback = RecordingCallback()

        application._post(callback)
        application.root.run_pending()

        assert callback.invocations == 1

    def test_runs_the_callback_immediately_on_an_already_pumping_event_loop(self, app):
        callback = RecordingCallback()

        app._post(callback)

        assert callback.invocations == 1

    def test_returns_none(self, app):
        assert app._post(RecordingCallback()) is None

    def test_preserves_the_order_of_queued_callbacks(self):
        application = app_with_deferred_event_loop()
        journal = []
        first = RecordingCallback("first", journal)
        second = RecordingCallback("second", journal)

        application._post(first)
        application._post(second)
        application.root.run_pending()

        assert journal == ["first", "second"]

    @pytest.mark.parametrize(
        "error",
        [
            tkinter.TclError("application has been destroyed"),
            tkinter.TclError("invalid command name"),
            RuntimeError("main thread is not in main loop"),
        ],
    )
    def test_posting_never_touches_the_window(self, error):
        application = make_app(root=RootThatRefusesToSchedule(error), _ui_queue=queue.Queue())

        assert application._post(RecordingCallback()) is None
        assert application.root.schedule_attempts == 0

    def test_the_drain_runs_queued_callbacks_in_order(self):
        application = make_app(root=FakeRoot(run_immediate=False), _ui_queue=queue.Queue())
        journal = []

        application._post(RecordingCallback("first", journal))
        application._post(RecordingCallback("second", journal))
        application._drain_ui_queue()

        assert journal == ["first", "second"]

    def test_the_drain_reschedules_itself(self):
        application = make_app(root=FakeRoot(run_immediate=False), _ui_queue=queue.Queue())

        application._drain_ui_queue()

        assert application.root.after_calls == [(ui.UI_POLL_MS, application._drain_ui_queue)]

    def test_a_callback_for_a_destroyed_widget_is_dropped_and_the_drain_continues(self):
        application = make_app(root=FakeRoot(run_immediate=False), _ui_queue=queue.Queue())
        survivor = RecordingCallback()

        application._post(CallbackThatRaises(tkinter.TclError("invalid command name")))
        application._post(survivor)
        application._drain_ui_queue()

        assert survivor.invocations == 1

    def test_a_swallowed_failure_leaves_no_breadcrumb(self):
        application = make_app(root=FakeRoot(run_immediate=False), _ui_queue=queue.Queue())

        application._post(CallbackThatRaises(tkinter.TclError("application has been destroyed")))
        application._drain_ui_queue()

        assert application._session_log == []

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("bad value"),
            AttributeError("widget has no configure"),
            KeyboardInterrupt(),
        ],
    )
    def test_unexpected_callback_errors_still_surface(self, error):
        application = make_app(root=FakeRoot(run_immediate=False), _ui_queue=queue.Queue())
        application._post(CallbackThatRaises(error))

        with pytest.raises(type(error)):
            application._drain_ui_queue()

    def test_the_swallowed_tcl_error_is_the_real_tkinter_one(self):
        assert ui.tk.TclError is tkinter.TclError

    @pytest.mark.gui
    def test_schedules_onto_a_real_tk_event_loop(self, tk_root):
        window = tkinter.Toplevel(tk_root)
        window.withdraw()
        callback = RecordingCallback()
        application = make_app(root=window, _ui_queue=queue.Queue())

        try:
            application._drain_ui_queue()
            application._post(callback)
            assert callback.invocations == 0

            time.sleep(ui.UI_POLL_MS * 2 / 1000)
            window.update()

            assert callback.invocations == 1
        finally:
            window.destroy()


class TestDetectGpu:
    def test_stores_the_detected_adapter_name(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)

        app._detect_gpu()

        assert app._gpu_name == GPU_NAME

    def test_records_the_detected_adapter_name(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)

        app._detect_gpu()

        assert f"GPU detection: {GPU_NAME}" in session_log_text(app)

    def test_probes_the_hardware_exactly_once(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        detector = install_gpu_detector(monkeypatch, name=GPU_NAME)

        app._detect_gpu()

        assert detector.call_count == 1

    def test_refreshes_the_status_row_with_the_detected_adapter(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)

        app._detect_gpu()

        assert app.gpu_status_label.text == f"GPU: {GPU_NAME} — acceleration ready"

    def test_leaves_the_name_unset_when_no_adapter_is_present(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        install_gpu_detector(monkeypatch, name=None)

        app._detect_gpu()

        assert app._gpu_name is None

    def test_records_that_no_adapter_was_found(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        install_gpu_detector(monkeypatch, name=None)

        app._detect_gpu()

        assert "GPU detection: no NVIDIA GPU found" in session_log_text(app)

    def test_refreshes_the_status_row_to_the_cpu_message(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        install_gpu_detector(monkeypatch, name=None)

        app._detect_gpu()

        assert app.gpu_status_label.text == "GPU: no NVIDIA GPU detected — using CPU"

    def test_writes_exactly_one_breadcrumb(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)

        app._detect_gpu()

        assert len(app._session_log) == 1

    def test_says_nothing_in_the_visible_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)

        app._detect_gpu()

        assert app.log_text.value == ""

    def test_marshals_the_row_refresh_onto_the_tk_thread(self, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)
        application = app_with_deferred_event_loop()

        application._detect_gpu()

        assert application.root.after_calls == [(0, application._refresh_gpu_row)]
        assert application.gpu_status_label.text == "GPU: checking..."

    def test_the_marshalled_refresh_updates_the_row_when_the_loop_turns(self, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)
        application = app_with_deferred_event_loop()

        application._detect_gpu()
        application.root.run_pending()

        assert application.gpu_status_label.text == f"GPU: {GPU_NAME} — acceleration ready"

    def test_a_closed_window_does_not_break_the_detection_thread(self, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=GPU_NAME)
        application = make_app(
            root=RootThatRefusesToSchedule(tkinter.TclError("application has been destroyed")),
            _ui_queue=queue.Queue())

        application._detect_gpu()

        assert application._gpu_name == GPU_NAME

    def test_an_empty_adapter_name_is_logged_as_no_gpu_but_still_renders_a_gpu_row(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        install_gpu_detector(monkeypatch, name="")

        app._detect_gpu()

        assert "GPU detection: no NVIDIA GPU found" in session_log_text(app)
        assert app._gpu_name == ""
        assert app.gpu_status_label.text == "GPU:  — acceleration not installed"

    def test_overwrites_a_previously_detected_adapter(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, name=None)
        app._gpu_name = GPU_NAME

        app._detect_gpu()

        assert app._gpu_name is None

    def test_a_crashing_hardware_probe_is_not_guarded(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        install_gpu_detector(monkeypatch, error=RuntimeError("nvidia-smi exploded"))

        with pytest.raises(RuntimeError, match="nvidia-smi exploded"):
            app._detect_gpu()


class TestRefreshGpuRow:
    @pytest.mark.parametrize("available", [True, False])
    def test_mirrors_the_library_probe_into_the_ready_flag(self, app, monkeypatch, available):
        install_library_probe(monkeypatch, available=available)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app._gpu_ready is available

    def test_re_reads_the_library_probe_on_every_call(self, app, monkeypatch):
        probe = install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()
        probe.available = True
        app._refresh_gpu_row()

        assert probe.call_count == 2
        assert app._gpu_ready is True

    def test_a_stale_ready_flag_is_corrected_when_the_libraries_disappear(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME
        app._gpu_ready = True

        app._refresh_gpu_row()

        assert app._gpu_ready is False

    def test_no_adapter_shows_the_cpu_message(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = None

        app._refresh_gpu_row()

        assert app.gpu_status_label.text == "GPU: no NVIDIA GPU detected — using CPU"

    def test_no_adapter_hides_the_install_button(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = None

        app._refresh_gpu_row()

        assert app.gpu_button.pack_forget_count == 1
        assert app.gpu_button.packed is False
        assert app.gpu_button.pack_calls == []

    def test_no_adapter_wins_over_ready_libraries(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app._gpu_name = None

        app._refresh_gpu_row()

        assert app.gpu_status_label.text == "GPU: no NVIDIA GPU detected — using CPU"
        assert app._gpu_ready is True
        assert app.gpu_button.pack_forget_count == 1

    def test_an_adapter_with_ready_libraries_reports_acceleration_ready(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app.gpu_status_label.text == f"GPU: {GPU_NAME} — acceleration ready"

    def test_an_adapter_with_ready_libraries_hides_the_install_button(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app.gpu_button.pack_forget_count == 1
        assert app.gpu_button.packed is False
        assert app.gpu_button.pack_calls == []

    def test_an_adapter_without_libraries_reports_not_installed(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app.gpu_status_label.text == f"GPU: {GPU_NAME} — acceleration not installed"

    def test_an_adapter_without_libraries_packs_the_install_button_on_the_right(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app.gpu_button.pack_calls == [{"side": "right"}]
        assert app.gpu_button.packed is True
        assert app.gpu_button.pack_forget_count == 0

    def test_installing_the_libraries_hides_the_button_on_the_next_refresh(self, app, monkeypatch):
        probe = install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()
        probe.available = True
        app._refresh_gpu_row()

        assert app.gpu_button.packed is False
        assert app.gpu_button.pack_forget_count == 1
        assert app.gpu_status_label.text == f"GPU: {GPU_NAME} — acceleration ready"

    def test_detecting_an_adapter_later_reveals_the_button(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = None

        app._refresh_gpu_row()
        app._gpu_name = GPU_NAME
        app._refresh_gpu_row()

        assert app.gpu_button.pack_calls == [{"side": "right"}]
        assert app.gpu_button.packed is True

    def test_repeated_refreshes_keep_repacking_the_visible_button(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()
        app._refresh_gpu_row()

        assert app.gpu_button.pack_calls == [{"side": "right"}, {"side": "right"}]

    @pytest.mark.parametrize("available", [True, False])
    def test_only_the_label_text_is_reconfigured(self, app, monkeypatch, available):
        install_library_probe(monkeypatch, available=available)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app.gpu_status_label.config_history == [{"text": app.gpu_status_label.text}]

    def test_the_row_is_refreshed_without_touching_the_visible_log(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=False)
        app._gpu_name = GPU_NAME

        app._refresh_gpu_row()

        assert app.log_text.value == ""
        assert app._session_log == []

    def test_returns_none(self, app, monkeypatch):
        install_library_probe(monkeypatch, available=True)
        app._gpu_name = GPU_NAME

        assert app._refresh_gpu_row() is None

    def test_a_failing_library_probe_is_not_swallowed(self, app, monkeypatch):
        def explode():
            raise OSError("WinDLL probe crashed")

        monkeypatch.setattr(ui.gpu_pack, "libraries_available", explode)
        app._gpu_name = GPU_NAME

        with pytest.raises(OSError, match="WinDLL probe crashed"):
            app._refresh_gpu_row()
