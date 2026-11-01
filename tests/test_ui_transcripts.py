import os

import pytest
from doubles import FakeVar, SystemShellDouble

import ui


class StartFileRecorder:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args):
        self.paths.append(path)


class ThreadingDouble:
    def __init__(self):
        self.started = 0

    def Thread(self, target=None, daemon=None, **keyword_arguments):
        double = self

        class StartedThread:
            def start(self):
                double.started += 1

        return StartedThread()


@pytest.fixture
def shell(monkeypatch):
    double = SystemShellDouble()
    monkeypatch.setattr(ui, "system_shell", double)
    return double


@pytest.fixture
def start_file(monkeypatch):
    recorder = StartFileRecorder()
    monkeypatch.setattr(ui, "IS_WINDOWS", True)
    monkeypatch.setattr(ui.os, "startfile", recorder, raising=False)
    return recorder


@pytest.fixture
def output_folder(app, tmp_path):
    folder = tmp_path / "transcripts"
    folder.mkdir()
    app.output_var = FakeVar(str(folder))
    return folder


def write_transcripts(folder, *names):
    paths = []
    for name in names:
        path = folder / name
        path.write_text("texto", encoding="utf-8")
        paths.append(str(path))
    return paths


class TestOpenOutputFolderSelectsTheTranscripts:
    def test_selects_the_transcripts_of_the_last_run(self, app, shell, start_file, output_folder):
        app.last_transcripts = write_transcripts(output_folder, "one.txt", "two.txt")

        app._open_output()

        assert shell.reveal_calls == [{"folder": str(output_folder), "paths": app.last_transcripts}]
        assert start_file.paths == []

    def test_records_how_many_transcripts_were_selected(self, app, shell, start_file, output_folder):
        app.last_transcripts = write_transcripts(output_folder, "one.txt", "two.txt")

        app._open_output()

        assert any(line.endswith(f"Opened output folder: {output_folder} with 2 transcripts selected")
                   for line in app._session_log)

    def test_a_single_transcript_is_counted_in_the_singular(self, app, shell, start_file, output_folder):
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._open_output()

        assert any(line.endswith("with 1 transcript selected") for line in app._session_log)

    def test_opens_the_plain_folder_when_the_selection_fails(self, app, shell, start_file, output_folder):
        shell.reveal_result = False
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._open_output()

        assert start_file.paths == [str(output_folder)]
        assert any("opening the folder instead" in line for line in app._session_log)

    def test_opens_the_plain_folder_when_the_selection_raises(self, app, shell, start_file, output_folder):
        shell.reveal_error = OSError("shell unavailable")
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._open_output()

        assert start_file.paths == [str(output_folder)]
        assert any("shell unavailable" in line for line in app._session_log)

    def test_only_transcripts_inside_the_opened_folder_are_selected(self, app, shell, start_file,
                                                                    output_folder, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        inside = write_transcripts(output_folder, "inside.txt")
        app.last_transcripts = inside + write_transcripts(elsewhere, "outside.txt")

        app._open_output()

        assert shell.reveal_calls[0]["paths"] == inside

    def test_opens_the_plain_folder_when_every_transcript_lives_elsewhere(self, app, shell, start_file,
                                                                         output_folder, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        app.last_transcripts = write_transcripts(elsewhere, "outside.txt")

        app._open_output()

        assert shell.reveal_calls == []
        assert start_file.paths == [str(output_folder)]

    def test_transcripts_deleted_since_the_run_are_not_selected(self, app, shell, start_file, output_folder):
        kept, deleted = write_transcripts(output_folder, "kept.txt", "deleted.txt")
        os.remove(deleted)
        app.last_transcripts = [kept, deleted]

        app._open_output()

        assert shell.reveal_calls[0]["paths"] == [kept]

    def test_opens_the_plain_folder_before_anything_was_transcribed(self, app, shell, start_file,
                                                                    output_folder):
        app._open_output()

        assert shell.reveal_calls == []
        assert start_file.paths == [str(output_folder)]

    @pytest.mark.parametrize("suffix", [os.sep, os.sep + "." + os.sep])
    def test_the_folder_matches_however_its_path_is_spelled(self, app, shell, start_file,
                                                           output_folder, suffix):
        app.last_transcripts = write_transcripts(output_folder, "one.txt")
        app.output_var = FakeVar(str(output_folder) + suffix)

        app._open_output()

        assert shell.reveal_calls[0]["paths"] == app.last_transcripts

    @pytest.mark.skipif(os.name != "nt", reason="Windows paths ignore case")
    def test_the_folder_matches_regardless_of_letter_case_on_windows(self, app, shell, start_file,
                                                                    output_folder):
        app.last_transcripts = write_transcripts(output_folder, "one.txt")
        app.output_var = FakeVar(str(output_folder).upper())

        app._open_output()

        assert shell.reveal_calls[0]["paths"] == app.last_transcripts


class TestCopyTranscripts:
    def test_copies_every_transcript_of_the_last_run(self, app, shell, output_folder):
        app.last_transcripts = write_transcripts(output_folder, "one.txt", "two.txt")

        app._copy_transcripts()

        assert shell.copied_paths == [app.last_transcripts]

    def test_the_main_window_owns_the_clipboard(self, app, shell, output_folder):
        app._hwnd = 9001
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._copy_transcripts()

        assert shell.copy_calls[0]["owner_window"] == 9001

    def test_falls_back_to_the_tk_window_before_the_frame_handle_is_known(self, app, shell, output_folder):
        app._hwnd = None
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._copy_transcripts()

        assert shell.copy_calls[0]["owner_window"] == app.root.winfo_id()

    @pytest.mark.parametrize(("names", "expected"), [
        (("one.txt",), "Copied 1 transcript to the clipboard. Paste them anywhere."),
        (("one.txt", "two.txt", "three.txt"), "Copied 3 transcripts to the clipboard. Paste them anywhere."),
    ])
    def test_tells_the_user_how_many_were_copied(self, app, shell, output_folder, names, expected):
        app.last_transcripts = write_transcripts(output_folder, *names)

        app._copy_transcripts()

        assert app.log_text.value.splitlines()[-1] == expected

    def test_transcripts_deleted_since_the_run_are_not_copied(self, app, shell, output_folder):
        kept, deleted = write_transcripts(output_folder, "kept.txt", "deleted.txt")
        os.remove(deleted)
        app.last_transcripts = [kept, deleted]

        app._copy_transcripts()

        assert shell.copied_paths == [[kept]]

    def test_explains_when_there_is_nothing_to_copy(self, app, shell):
        app._copy_transcripts()

        assert shell.copy_calls == []
        assert "No transcripts to copy yet" in app.log_text.value

    def test_reports_a_refused_copy(self, app, shell, output_folder):
        shell.copy_result = False
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._copy_transcripts()

        assert "Could not copy the transcripts to the clipboard" in app.log_text.value

    def test_a_raising_copy_is_reported_instead_of_crashing(self, app, shell, output_folder):
        shell.copy_error = OSError("GlobalAlloc failed")
        app.last_transcripts = write_transcripts(output_folder, "one.txt")

        app._copy_transcripts()

        assert "Could not copy the transcripts to the clipboard" in app.log_text.value
        assert any("GlobalAlloc failed" in line for line in app._session_log)


class TestPublishTranscripts:
    def test_enables_the_copy_button_when_a_run_wrote_transcripts(self, app, shell, output_folder):
        app.copy_transcripts_button.set_enabled(False)

        app._publish_transcripts(write_transcripts(output_folder, "one.txt"), copy_now=False)

        assert app.copy_transcripts_button.enabled_history == [False, True]

    def test_disables_the_copy_button_when_a_run_wrote_nothing(self, app, shell):
        app._publish_transcripts([], copy_now=False)

        assert app.copy_transcripts_button.enabled_history == [False]

    def test_copies_right_away_when_asked(self, app, shell, output_folder):
        paths = write_transcripts(output_folder, "one.txt", "two.txt")

        app._publish_transcripts(paths, copy_now=True)

        assert shell.copied_paths == [paths]

    def test_leaves_the_clipboard_alone_when_not_asked(self, app, shell, output_folder):
        app._publish_transcripts(write_transcripts(output_folder, "one.txt"), copy_now=False)

        assert shell.copy_calls == []

    def test_an_empty_run_never_copies_even_when_asked(self, app, shell):
        app._publish_transcripts([], copy_now=True)

        assert shell.copy_calls == []
        assert "No transcripts to copy yet" not in app.log_text.value

    def test_replaces_the_transcripts_of_the_previous_run(self, app, shell, output_folder):
        app.last_transcripts = write_transcripts(output_folder, "old.txt")
        new = write_transcripts(output_folder, "new.txt")

        app._publish_transcripts(new, copy_now=False)

        assert app.last_transcripts == new

    def test_keeps_its_own_copy_of_the_list(self, app, shell, output_folder):
        paths = write_transcripts(output_folder, "one.txt")

        app._publish_transcripts(paths, copy_now=False)
        paths.append("later.txt")

        assert app.last_transcripts == [str(output_folder / "one.txt")]


class TestStartingARunForgetsThePreviousTranscripts:
    def test_the_previous_transcripts_are_cleared_and_the_button_disabled(self, app, shell, output_folder,
                                                                         monkeypatch):
        monkeypatch.setattr(ui, "threading", ThreadingDouble())
        app.last_transcripts = write_transcripts(output_folder, "old.txt")
        app.files = [str(output_folder / "clip.mp3")]

        app._toggle_transcription()

        assert app.last_transcripts == []
        assert app.copy_transcripts_button.enabled is False

    def test_cancelling_a_run_keeps_the_transcripts_already_listed(self, app, shell, output_folder,
                                                                  monkeypatch):
        monkeypatch.setattr(ui, "threading", ThreadingDouble())
        app.is_transcribing = True
        app.last_transcripts = write_transcripts(output_folder, "old.txt")

        app._toggle_transcription()

        assert app.last_transcripts == [str(output_folder / "old.txt")]
