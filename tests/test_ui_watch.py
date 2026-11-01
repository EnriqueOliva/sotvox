import pytest
from doubles import FakeVar, SystemShellDouble

import ui


class RecordedThread:
    def __init__(self, target, keyword_arguments):
        self.target = target
        self.keyword_arguments = keyword_arguments
        self.started = False

    def start(self):
        self.started = True


class ThreadingDouble:
    def __init__(self):
        self.threads = []

    def Thread(self, target=None, daemon=None, kwargs=None, **unused):
        thread = RecordedThread(target, kwargs or {})
        self.threads.append(thread)
        return thread


class WatcherDouble:
    def __init__(self, batches=(), error=None):
        self.batches = list(batches)
        self.error = error
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.error is not None:
            raise self.error
        return self.batches.pop(0) if self.batches else []


class SettingsDouble:
    def __init__(self, error=None):
        self.saved = []
        self.error = error

    def save_settings(self, values):
        if self.error is not None:
            raise self.error
        self.saved.append(dict(values))


class FolderWatchDouble:
    def __init__(self):
        self.created = []

    def FolderWatcher(self, folder, extensions):
        watcher = WatcherDouble()
        self.created.append({"folder": folder, "extensions": set(extensions), "watcher": watcher})
        return watcher


@pytest.fixture
def threads(monkeypatch):
    double = ThreadingDouble()
    monkeypatch.setattr(ui, "threading", double)
    return double


@pytest.fixture
def shell(monkeypatch):
    double = SystemShellDouble()
    monkeypatch.setattr(ui, "system_shell", double)
    return double


@pytest.fixture
def saved_settings(monkeypatch):
    double = SettingsDouble()
    monkeypatch.setattr(ui, "settings", double)
    return double


@pytest.fixture
def folder_watch(monkeypatch):
    double = FolderWatchDouble()
    monkeypatch.setattr(ui, "folder_watch", double)
    return double


def downloads(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"audio")
        paths.append(str(path))
    return paths


class TestTurningTheWatchOnAndOff:
    def test_turning_it_on_remembers_the_choice(self, app, saved_settings, folder_watch):
        app.watch_var = FakeVar(True)

        app._on_watch_toggled()

        assert saved_settings.saved == [{"watch_downloads": True}]

    def test_turning_it_on_watches_the_downloads_folder_for_media(self, app, saved_settings, folder_watch):
        app.watch_var = FakeVar(True)

        app._on_watch_toggled()

        assert folder_watch.created[0]["folder"] == app._watch_folder
        assert folder_watch.created[0]["extensions"] == set(ui.SUPPORTED_EXTENSIONS)
        assert app._watcher is folder_watch.created[0]["watcher"]

    def test_turning_it_on_tells_the_user_what_will_happen(self, app, saved_settings, folder_watch):
        app.watch_var = FakeVar(True)

        app._on_watch_toggled()

        assert f"Watching {app._watch_folder}." in app.log_text.value

    def test_turning_it_on_twice_keeps_the_same_watcher(self, app, saved_settings, folder_watch):
        app.watch_var = FakeVar(True)

        app._on_watch_toggled()
        app._on_watch_toggled()

        assert len(folder_watch.created) == 1

    def test_turning_it_off_forgets_the_watcher_and_anything_queued(self, app, saved_settings, folder_watch):
        app._watcher = WatcherDouble()
        app._watch_queue = ["queued.m4a"]
        app.watch_var = FakeVar(False)

        app._on_watch_toggled()

        assert app._watcher is None
        assert app._watch_queue == []
        assert saved_settings.saved == [{"watch_downloads": False}]
        assert "Stopped watching Downloads." in app.log_text.value

    def test_a_settings_file_that_cannot_be_written_does_not_stop_the_watch(self, app, monkeypatch,
                                                                          folder_watch):
        monkeypatch.setattr(ui, "settings", SettingsDouble(error=OSError("disk full")))
        app.watch_var = FakeVar(True)

        app._on_watch_toggled()

        assert app._watcher is not None
        assert any("disk full" in line for line in app._session_log)


class TestPollingTheWatch:
    def test_new_files_start_a_run_with_only_those_files(self, app, tmp_path, threads, shell):
        app.files = downloads(tmp_path, "older-manual.mp3")
        arrived = downloads(tmp_path, "audio_message.m4a", "WhatsApp Ptt.ogg")
        app._watcher = WatcherDouble([arrived])

        app._poll_watch()

        assert threads.threads[0].keyword_arguments == {"files": arrived, "keep_existing": True}
        assert threads.threads[0].started is True
        assert threads.threads[0].target == app._transcribe_worker

    def test_new_files_appear_in_the_list(self, app, tmp_path, threads, shell):
        arrived = downloads(tmp_path, "audio_message.m4a")
        app._watcher = WatcherDouble([arrived])

        app._poll_watch()

        assert app.files == arrived
        assert app.file_listbox.items == [" [AUD]  audio_message.m4a"]

    def test_the_log_names_what_arrived(self, app, tmp_path, threads, shell):
        arrived = downloads(tmp_path, "audio_message.m4a")
        app._watcher = WatcherDouble([arrived])

        app._poll_watch()

        assert "New in Downloads: audio_message.m4a" in app.log_text.value

    def test_files_arriving_during_a_run_wait_for_it_to_finish(self, app, tmp_path, threads, shell):
        arrived = downloads(tmp_path, "second.m4a")
        app._watcher = WatcherDouble([arrived])
        app._run_active = True
        app.is_transcribing = True

        app._poll_watch()

        assert threads.threads == []
        assert app._watch_queue == arrived

    def test_a_queued_batch_starts_once_the_run_has_finished(self, app, tmp_path, threads, shell):
        arrived = downloads(tmp_path, "second.m4a")
        app._watcher = WatcherDouble([arrived])
        app._run_active = True
        app._poll_watch()
        app._finish_run([], run_completed=True)

        app._poll_watch()

        assert [thread.keyword_arguments["files"] for thread in threads.threads] == [arrived]

    def test_a_run_whose_worker_ended_but_whose_ui_finish_is_pending_still_blocks(self, app, tmp_path,
                                                                                 threads, shell):
        app._watcher = WatcherDouble([downloads(tmp_path, "second.m4a")])
        app._run_active = True
        app.is_transcribing = False

        app._poll_watch()

        assert threads.threads == []

    def test_the_same_file_is_never_queued_twice(self, app, tmp_path, threads, shell):
        arrived = downloads(tmp_path, "second.m4a")
        app._watcher = WatcherDouble([arrived, arrived])
        app._run_active = True

        app._poll_watch()
        app._poll_watch()

        assert app._watch_queue == arrived

    def test_polling_keeps_going_after_the_watcher_fails(self, app, threads, shell):
        app._watcher = WatcherDouble(error=OSError("network drive gone"))

        app._poll_watch()

        assert app.root.pending[-1] == (ui.WATCH_POLL_MS, app._poll_watch)
        assert any("network drive gone" in line for line in app._session_log)

    def test_polling_without_a_watcher_does_nothing_but_keeps_polling(self, app, threads, shell):
        app._poll_watch()

        assert threads.threads == []
        assert app.root.pending[-1] == (ui.WATCH_POLL_MS, app._poll_watch)


class TestFinishingARun:
    def test_marks_the_run_as_over_and_restores_the_button(self, app, shell):
        app._run_active = True
        app.transcribe_btn.set_text("Cancel")

        app._finish_run([], run_completed=True)

        assert app._run_active is False
        assert app.transcribe_btn.texts[-1] == "Transcribe"
        assert app.transcribe_btn.enabled is True

    def test_hands_the_transcripts_over_and_copies_them_when_the_run_completed(self, app, shell, tmp_path):
        written = downloads(tmp_path, "clip.txt")

        app._finish_run(written, run_completed=True)

        assert app.last_transcripts == written
        assert shell.copied_paths == [written]


class TestManualRuns:
    def test_the_transcribe_button_runs_the_whole_list_and_overwrites_as_before(self, app, threads, shell):
        app.files = [r"C:\media\a.mp3", r"C:\media\b.mp3"]

        app._toggle_transcription()

        assert threads.threads[0].keyword_arguments == {"files": app.files, "keep_existing": False}
        assert app._run_active is True

    def test_the_run_gets_its_own_copy_of_the_list(self, app, threads, shell):
        app.files = [r"C:\media\a.mp3"]

        app._toggle_transcription()
        app.files.append(r"C:\media\later.mp3")

        assert threads.threads[0].keyword_arguments["files"] == [r"C:\media\a.mp3"]


class TestStartWithWindowsCheckbox:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_the_checkbox_sets_the_startup_entry(self, app, shell, enabled):
        app.startup_var = FakeVar(enabled)

        app._on_startup_toggled()

        assert shell.startup_calls == [enabled]

    def test_a_refused_change_is_reported(self, app, shell):
        shell.startup_error = PermissionError("access denied")
        app.startup_var = FakeVar(True)

        app._on_startup_toggled()

        assert "Could not change Start with Windows: access denied" in app.log_text.value


class TestStartingMinimized:
    def test_a_minimized_start_iconifies_the_window_after_it_is_shown(self, app, monkeypatch):
        monkeypatch.setattr(app, "_make_borderless", lambda: None)
        app._start_minimized = True

        app._reveal()

        assert (ui.MINIMIZE_DELAY_MS, app._minimize) in app.root.pending

    def test_a_normal_start_stays_on_screen(self, app, monkeypatch):
        monkeypatch.setattr(app, "_make_borderless", lambda: None)

        app._reveal()

        assert all(callback != app._minimize for _delay, callback in app.root.pending)
