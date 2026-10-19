"""File-queue behaviour of ui.SotvoxApp: drop parsing, queueing, removal and listbox refresh."""

import os

import pytest

import constants

SORTED_SUPPORTED_EXTENSIONS = sorted(constants.SUPPORTED_EXTENSIONS)
SORTED_VIDEO_EXTENSIONS = sorted(constants.VIDEO_EXTENSIONS)
SORTED_AUDIO_EXTENSIONS = sorted(constants.SUPPORTED_EXTENSIONS - constants.VIDEO_EXTENSIONS)

STALE_LISTBOX_ROW = " [AUD]  row-left-over-from-a-previous-refresh.mp3"
UNTOUCHED_STATUS_TEXT = "status text written before the call"
NO_SUPPORTED_FILES_MESSAGE = "No supported files found in the dropped items."


class DropEvent:
    def __init__(self, data):
        self.data = data


class RecordingAddFiles:
    def __init__(self):
        self.calls = []

    def __call__(self, paths):
        self.calls.append(list(paths))

    @property
    def last_paths(self):
        return self.calls[-1]


def session_messages(app):
    return [line.split("] ", 1)[1] for line in app._session_log]


def mark_listbox_as_untouched(app):
    app.file_listbox.items.append(STALE_LISTBOX_ROW)
    app.status_files.configure(text=UNTOUCHED_STATUS_TEXT)


def listbox_was_refreshed(app):
    return STALE_LISTBOX_ROW not in app.file_listbox.items


class TestOnDropPathParsing:
    @pytest.mark.parametrize(
        ("dropped_data", "expected_paths"),
        [
            (r"C:\media\take.mp3", [r"C:\media\take.mp3"]),
            (r"{C:\media\my long take.mp3}", [r"C:\media\my long take.mp3"]),
            (
                r"{C:\media\first take.mp3} {C:\media\second take.mp4}",
                [r"C:\media\first take.mp3", r"C:\media\second take.mp4"],
            ),
            (
                r"C:\media\first.mp3 C:\media\second.mp4",
                [r"C:\media\first.mp3", r"C:\media\second.mp4"],
            ),
            (
                r"{C:\media\first take.mp3} C:\media\second.mp4",
                [r"C:\media\first take.mp3", r"C:\media\second.mp4"],
            ),
            (
                r"C:\media\first.mp3 {C:\media\second take.mp4}",
                [r"C:\media\first.mp3", r"C:\media\second take.mp4"],
            ),
            (
                r"C:\a.mp3 {C:\b b.mp4} C:\c.wav {C:\d d.flac}",
                [r"C:\a.mp3", r"C:\b b.mp4", r"C:\c.wav", r"C:\d d.flac"],
            ),
            (
                r"{C:\a a.mp3}{C:\b b.mp4}",
                [r"C:\a a.mp3", r"C:\b b.mp4"],
            ),
            (
                "C:\\a.mp3     C:\\b.mp4",
                [r"C:\a.mp3", r"C:\b.mp4"],
            ),
            (
                "C:\\a.mp3\tC:\\b.mp4",
                [r"C:\a.mp3", r"C:\b.mp4"],
            ),
            (
                "C:\\a.mp3\nC:\\b.mp4",
                [r"C:\a.mp3", r"C:\b.mp4"],
            ),
            ("", []),
            ("    ", []),
            ("\t\n ", []),
        ],
    )
    def test_parses_the_tkdnd_payload_into_individual_paths(self, app, dropped_data, expected_paths):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent(dropped_data))

        assert recorder.last_paths == expected_paths

    @pytest.mark.parametrize(
        ("dropped_data", "expected_paths"),
        [
            (r"C:\media\Bob's take.mp3".replace(" ", ""), [r"C:\media\Bob'stake.mp3"]),
            (r"{C:\media\Bob's long take.mp3}", [r"C:\media\Bob's long take.mp3"]),
            (r"C:\media\take-01-final.mp4", [r"C:\media\take-01-final.mp4"]),
            (r"{C:\media\take - 01 - final.mp4}", [r"C:\media\take - 01 - final.mp4"]),
            (r"C:\media\entrevista_ñandú.mp3", [r"C:\media\entrevista_ñandú.mp3"]),
            (r"{C:\media\take (1).mp4}", [r"C:\media\take (1).mp4"]),
            (r"{C:\media\[raw] take #2.mp4}", [r"C:\media\[raw] take #2.mp4"]),
            (r"//server/share/take.mp3", [r"//server/share/take.mp3"]),
            ("C:/media/forward slashes".replace(" ", "-") + ".mp3", ["C:/media/forward-slashes.mp3"]),
        ],
    )
    def test_parses_paths_containing_punctuation(self, app, dropped_data, expected_paths):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent(dropped_data))

        assert recorder.last_paths == expected_paths

    def test_a_bare_path_containing_spaces_is_split_because_tkdnd_did_not_brace_it(self, app):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent(r"C:\media\take one.mp4"))

        assert recorder.last_paths == [r"C:\media\take", "one.mp4"]

    def test_an_unterminated_brace_falls_back_to_whitespace_splitting(self, app):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent(r"{C:\media\take one.mp4"))

        assert recorder.last_paths == [r"{C:\media\take", "one.mp4"]

    def test_an_empty_brace_pair_is_kept_as_a_literal_token(self, app):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent("{}"))

        assert recorder.last_paths == ["{}"]

    def test_braced_paths_keep_interior_whitespace_verbatim(self, app):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent("{C:\\media\\two  spaces.mp3}"))

        assert recorder.last_paths == ["C:\\media\\two  spaces.mp3"]


class TestOnDropLoggingAndForwarding:
    @pytest.mark.parametrize(
        ("dropped_data", "expected_count"),
        [
            ("", 0),
            ("   ", 0),
            (r"C:\media\take.mp3", 1),
            (r"{C:\media\take one.mp3}", 1),
            (r"C:\a.mp3 C:\b.mp3", 2),
            (r"{C:\a a.mp3} C:\b.mp3 {C:\c c.mp3}", 3),
        ],
    )
    def test_logs_how_many_items_were_dropped(self, app, dropped_data, expected_count):
        app._add_files = RecordingAddFiles()

        app._on_drop(DropEvent(dropped_data))

        assert session_messages(app)[0] == f"Files dropped: {expected_count} item(s)"

    def test_forwards_the_parsed_paths_to_add_files_exactly_once(self, app):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent(r"C:\a.mp3 C:\b.mp4"))

        assert recorder.calls == [[r"C:\a.mp3", r"C:\b.mp4"]]

    def test_still_calls_add_files_when_the_payload_parsed_to_nothing(self, app):
        recorder = RecordingAddFiles()
        app._add_files = recorder

        app._on_drop(DropEvent(""))

        assert recorder.calls == [[]]

    def test_logs_the_count_before_handing_the_paths_over(self, app):
        counts_seen_by_add_files = []
        app._add_files = lambda paths: counts_seen_by_add_files.append(list(session_messages(app)))

        app._on_drop(DropEvent(r"C:\a.mp3"))

        assert counts_seen_by_add_files == [["Files dropped: 1 item(s)"]]

    def test_returns_none(self, app):
        app._add_files = RecordingAddFiles()

        assert app._on_drop(DropEvent(r"C:\a.mp3")) is None

    def test_a_real_drop_queues_the_supported_files_and_refreshes_the_listbox(self, app):
        app._on_drop(DropEvent(r"{C:\Videos\my clip.mp4} C:\Audio\take.mp3"))

        assert app.files == [r"C:\Videos\my clip.mp4", r"C:\Audio\take.mp3"]
        assert app.file_listbox.items == [" [VID]  my clip.mp4", " [AUD]  take.mp3"]

    def test_a_drop_of_unsupported_items_tells_the_user_nothing_was_supported(self, app):
        app._on_drop(DropEvent(r"C:\docs\notes.txt C:\docs\report.pdf"))

        assert app.files == []
        assert app.log_text.value == NO_SUPPORTED_FILES_MESSAGE + "\n"

    def test_an_empty_drop_produces_only_the_count_line(self, app):
        app._on_drop(DropEvent(""))

        assert session_messages(app) == ["Files dropped: 0 item(s)"]
        assert app.log_text.value == ""


class TestAddFilesExtensionFiltering:
    @pytest.mark.parametrize("extension", SORTED_SUPPORTED_EXTENSIONS)
    def test_accepts_every_supported_extension(self, app, extension):
        path = f"C:\\media\\clip{extension}"

        app._add_files([path])

        assert app.files == [path]

    @pytest.mark.parametrize("extension", [".MP4", ".Mp3", ".WAV", ".FlAc", ".MOV", ".Ts", ".WebM"])
    def test_matches_the_extension_case_insensitively(self, app, extension):
        path = f"C:\\media\\clip{extension}"

        app._add_files([path])

        assert app.files == [path]

    def test_keeps_the_original_casing_of_an_accepted_path(self, app):
        app._add_files([r"C:\Media\CLIP.MP4"])

        assert app.files == [r"C:\Media\CLIP.MP4"]

    @pytest.mark.parametrize(
        "path",
        [
            r"C:\docs\notes.txt",
            r"C:\docs\report.pdf",
            r"C:\docs\sheet.xlsx",
            r"C:\media\clip.mp3.bak",
            r"C:\media\clip.mpeg",
            r"C:\media\clip.m4v",
            r"C:\media\clip.opus",
            r"C:\media\clip",
            r"C:\media\clip.",
            r"C:\media\.mp3",
            r"C:\music.mp3\notes.txt",
            "",
            "   ",
        ],
    )
    def test_rejects_paths_without_a_supported_extension(self, app, path):
        app._add_files([path])

        assert app.files == []

    def test_a_leading_dot_filename_has_no_extension_and_is_rejected(self, app):
        app._add_files([".mp3"])

        assert app.files == []

    def test_keeps_only_the_supported_paths_from_a_mixed_batch(self, app):
        app._add_files([r"C:\a.mp3", r"C:\b.txt", r"C:\c.mkv", r"C:\d.pdf"])

        assert app.files == [r"C:\a.mp3", r"C:\c.mkv"]

    def test_accepts_a_tuple_of_paths_as_delivered_by_the_file_dialog(self, app):
        app._add_files((r"C:\a.mp3", r"C:\b.mp4"))

        assert app.files == [r"C:\a.mp3", r"C:\b.mp4"]


class TestAddFilesQuoteStripping:
    @pytest.mark.parametrize(
        ("supplied_path", "expected_stored_path"),
        [
            ('"C:\\media\\clip.mp3"', r"C:\media\clip.mp3"),
            ("'C:\\media\\clip.mp3'", r"C:\media\clip.mp3"),
            ('""C:\\media\\clip.mp3""', r"C:\media\clip.mp3"),
            ("'''C:\\media\\clip.mp3'''", r"C:\media\clip.mp3"),
            ('"C:\\media\\clip.mp3', r"C:\media\clip.mp3"),
            ('C:\\media\\clip.mp3"', r"C:\media\clip.mp3"),
            ("'C:\\media\\clip.mp3", r"C:\media\clip.mp3"),
            ('"\'C:\\media\\clip.mp3\'"', r"C:\media\clip.mp3"),
        ],
    )
    def test_strips_surrounding_quotes_before_checking_the_extension(
        self, app, supplied_path, expected_stored_path
    ):
        app._add_files([supplied_path])

        assert app.files == [expected_stored_path]

    def test_single_quotes_wrapped_in_double_quotes_are_rejected_because_of_the_strip_order(self, app):
        app._add_files(["'\"C:\\media\\clip.mp3\"'"])

        assert app.files == []

    def test_does_not_strip_quotes_from_inside_the_path(self, app):
        app._add_files(['C:\\media\\say "hi".mp3'])

        assert app.files == ['C:\\media\\say "hi".mp3']

    def test_the_logged_path_line_shows_the_stripped_path(self, app):
        app._add_files(['"C:\\media\\clip.mp3"'])

        assert "  Path: C:\\media\\clip.mp3" in session_messages(app)


class TestAddFilesDuplicateRejection:
    def test_rejects_a_path_that_is_already_queued(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files([r"C:\media\clip.mp3"])

        assert app.files == [r"C:\media\clip.mp3"]

    def test_rejects_a_repeat_inside_the_same_batch(self, app):
        app._add_files([r"C:\media\clip.mp3", r"C:\media\clip.mp3"])

        assert app.files == [r"C:\media\clip.mp3"]

    def test_a_quoted_form_of_a_queued_path_is_recognised_as_a_duplicate(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files(['"C:\\media\\clip.mp3"'])

        assert app.files == [r"C:\media\clip.mp3"]

    def test_the_duplicate_is_not_logged_as_added(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files([r"C:\media\clip.mp3"])

        assert "File added: clip.mp3" not in session_messages(app)

    def test_the_same_file_spelled_with_a_different_case_is_queued_twice(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files([r"C:\media\CLIP.MP3"])

        assert app.files == [r"C:\media\clip.mp3", r"C:\media\CLIP.MP3"]

    def test_same_basename_in_different_folders_are_both_queued(self, app):
        app._add_files([r"C:\one\clip.mp3", r"C:\two\clip.mp3"])

        assert app.files == [r"C:\one\clip.mp3", r"C:\two\clip.mp3"]

    def test_a_new_file_alongside_a_duplicate_is_still_queued(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files([r"C:\media\clip.mp3", r"C:\media\other.mp4"])

        assert app.files == [r"C:\media\clip.mp3", r"C:\media\other.mp4"]


class TestAddFilesQueueOrdering:
    def test_preserves_the_order_of_a_single_batch(self, app):
        app._add_files([r"C:\c.mp3", r"C:\a.mp4", r"C:\b.wav"])

        assert app.files == [r"C:\c.mp3", r"C:\a.mp4", r"C:\b.wav"]

    def test_appends_later_batches_after_the_existing_queue(self, app):
        app._add_files([r"C:\a.mp3"])
        app._add_files([r"C:\b.mp4"])
        app._add_files([r"C:\c.wav"])

        assert app.files == [r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"]

    def test_mutates_the_existing_queue_list_in_place(self, app):
        original_queue = app.files

        app._add_files([r"C:\a.mp3"])

        assert app.files is original_queue


class TestAddFilesLoggingAndRefresh:
    def test_logs_the_basename_of_each_added_file(self, app):
        app._add_files([r"C:\media\first.mp3", r"C:\media\second.mp4"])

        messages = session_messages(app)
        assert "File added: first.mp3" in messages
        assert "File added: second.mp4" in messages

    def test_logs_the_full_path_of_each_added_file(self, app):
        app._add_files([r"C:\media\first.mp3"])

        assert "  Path: C:\\media\\first.mp3" in session_messages(app)

    def test_logs_the_running_queue_total_once_per_batch(self, app):
        app._add_files([r"C:\a.mp3", r"C:\b.mp4"])
        app._add_files([r"C:\c.wav"])

        totals = [line for line in session_messages(app) if line.startswith("Total files in queue")]
        assert totals == ["Total files in queue: 2", "Total files in queue: 3"]

    def test_refreshes_the_listbox_when_at_least_one_file_was_added(self, app):
        mark_listbox_as_untouched(app)

        app._add_files([r"C:\media\clip.mp4"])

        assert listbox_was_refreshed(app)
        assert app.file_listbox.items == [" [VID]  clip.mp4"]

    def test_updates_the_status_label_when_a_file_was_added(self, app):
        app._add_files([r"C:\media\clip.mp4"])

        assert app.status_files.cget("text") == "1 file"

    def test_refreshes_when_only_some_of_the_batch_was_accepted(self, app):
        mark_listbox_as_untouched(app)

        app._add_files([r"C:\media\clip.mp4", r"C:\docs\notes.txt"])

        assert listbox_was_refreshed(app)
        assert app.file_listbox.items == [" [VID]  clip.mp4"]

    def test_does_not_warn_the_user_when_at_least_one_file_was_added(self, app):
        app._add_files([r"C:\media\clip.mp4", r"C:\docs\notes.txt"])

        assert app.log_text.value == ""

    def test_warns_the_user_when_the_whole_batch_was_rejected(self, app):
        app._add_files([r"C:\docs\notes.txt", r"C:\docs\report.pdf"])

        assert app.log_text.value == NO_SUPPORTED_FILES_MESSAGE + "\n"

    def test_the_rejection_warning_also_lands_in_the_session_log(self, app):
        app._add_files([r"C:\docs\notes.txt", r"C:\docs\report.pdf"])

        assert session_messages(app) == [
            "Rejected 2 unsupported file(s)",
            NO_SUPPORTED_FILES_MESSAGE,
        ]

    def test_leaves_the_log_widget_disabled_after_warning_the_user(self, app):
        app._add_files([r"C:\docs\notes.txt"])

        assert app.log_text.cget("state") == "disabled"

    def test_does_not_refresh_the_listbox_when_nothing_was_added(self, app):
        mark_listbox_as_untouched(app)

        app._add_files([r"C:\docs\notes.txt"])

        assert not listbox_was_refreshed(app)
        assert app.status_files.cget("text") == UNTOUCHED_STATUS_TEXT

    def test_counts_every_supplied_path_as_unsupported_even_when_it_was_only_a_duplicate(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files([r"C:\media\clip.mp3"])

        assert "Rejected 1 unsupported file(s)" in session_messages(app)

    def test_warns_the_user_when_the_batch_held_only_duplicates(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._add_files([r"C:\media\clip.mp3"])

        assert app.log_text.value == NO_SUPPORTED_FILES_MESSAGE + "\n"

    def test_an_empty_path_list_logs_nothing(self, app):
        app._add_files([])

        assert session_messages(app) == []
        assert app.log_text.value == ""

    def test_an_empty_path_list_does_not_refresh_the_listbox(self, app):
        mark_listbox_as_untouched(app)

        app._add_files([])

        assert not listbox_was_refreshed(app)
        assert app.status_files.cget("text") == UNTOUCHED_STATUS_TEXT

    def test_an_empty_tuple_is_treated_like_an_empty_list(self, app):
        mark_listbox_as_untouched(app)

        app._add_files(())

        assert session_messages(app) == []
        assert not listbox_was_refreshed(app)

    def test_returns_none(self, app):
        assert app._add_files([r"C:\media\clip.mp4"]) is None


class TestAddFilesSizeReporting:
    def test_logs_the_formatted_size_of_a_real_file(self, app, tmp_path):
        media_file = tmp_path / "recording.mp3"
        media_file.write_bytes(b"x" * 12)

        app._add_files([str(media_file)])

        assert "  Size: 12.0 B" in session_messages(app)

    def test_reports_kilobytes_once_the_file_passes_one_kibibyte(self, app, tmp_path):
        media_file = tmp_path / "recording.wav"
        media_file.write_bytes(b"x" * 2048)

        app._add_files([str(media_file)])

        assert "  Size: 2.0 KB" in session_messages(app)

    def test_reports_an_empty_file_as_zero_bytes(self, app, tmp_path):
        media_file = tmp_path / "recording.flac"
        media_file.write_bytes(b"")

        app._add_files([str(media_file)])

        assert "  Size: 0.0 B" in session_messages(app)

    def test_a_missing_file_is_still_queued(self, app, tmp_path):
        missing_file = tmp_path / "never-written.mp3"

        app._add_files([str(missing_file)])

        assert app.files == [str(missing_file)]

    def test_a_missing_file_logs_no_size_line(self, app, tmp_path):
        missing_file = tmp_path / "never-written.mp3"

        app._add_files([str(missing_file)])

        assert not any(line.startswith("  Size:") for line in session_messages(app))

    def test_a_missing_file_still_counts_towards_the_queue_total(self, app, tmp_path):
        missing_file = tmp_path / "never-written.mp3"

        app._add_files([str(missing_file), r"C:\media\clip.mp4"])

        assert "Total files in queue: 2" in session_messages(app)

    def test_a_missing_file_still_refreshes_the_listbox(self, app, tmp_path):
        missing_file = tmp_path / "never-written.mp3"

        app._add_files([str(missing_file)])

        assert app.file_listbox.items == [" [AUD]  never-written.mp3"]

    def test_a_permission_error_while_sizing_is_swallowed(self, app, monkeypatch):
        def refuse(path):
            raise PermissionError("locked by another process")

        monkeypatch.setattr(os.path, "getsize", refuse)

        app._add_files([r"C:\media\clip.mp3"])

        assert app.files == [r"C:\media\clip.mp3"]
        assert "Total files in queue: 1" in session_messages(app)

    def test_a_non_oserror_while_sizing_is_not_swallowed(self, app, monkeypatch):
        def explode(path):
            raise ValueError("not an os level failure")

        monkeypatch.setattr(os.path, "getsize", explode)

        with pytest.raises(ValueError):
            app._add_files([r"C:\media\clip.mp3"])

    def test_a_non_oserror_leaves_the_file_queued_but_the_listbox_stale(self, app, monkeypatch):
        def explode(path):
            raise ValueError("not an os level failure")

        monkeypatch.setattr(os.path, "getsize", explode)
        mark_listbox_as_untouched(app)

        with pytest.raises(ValueError):
            app._add_files([r"C:\media\clip.mp3"])

        assert app.files == [r"C:\media\clip.mp3"]
        assert not listbox_was_refreshed(app)


class TestRemoveFile:
    def test_removes_a_queued_path(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])

        app._remove_file(r"C:\a.mp3")

        assert app.files == [r"C:\b.mp4"]

    def test_removing_from_the_middle_keeps_the_remaining_order(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])

        app._remove_file(r"C:\b.mp4")

        assert app.files == [r"C:\a.mp3", r"C:\c.wav"]

    def test_refreshes_the_listbox_after_removing(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        mark_listbox_as_untouched(app)

        app._remove_file(r"C:\a.mp3")

        assert listbox_was_refreshed(app)
        assert app.file_listbox.items == [" [VID]  b.mp4"]

    def test_logs_the_basename_and_the_new_total(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])

        app._remove_file(r"C:\a.mp3")

        assert session_messages(app) == ["File removed: a.mp3", "Total files in queue: 1"]

    def test_removing_the_last_file_leaves_an_empty_queue_and_status(self, app):
        app.files.append(r"C:\a.mp3")

        app._remove_file(r"C:\a.mp3")

        assert app.files == []
        assert app.file_listbox.items == []
        assert app.status_files.cget("text") == "0 files"

    def test_an_unqueued_path_leaves_the_queue_untouched(self, app):
        app.files.extend([r"C:\a.mp3"])

        app._remove_file(r"C:\zzz.mp3")

        assert app.files == [r"C:\a.mp3"]

    def test_an_unqueued_path_logs_nothing(self, app):
        app.files.extend([r"C:\a.mp3"])

        app._remove_file(r"C:\zzz.mp3")

        assert session_messages(app) == []

    def test_an_unqueued_path_does_not_refresh_the_listbox(self, app):
        app.files.extend([r"C:\a.mp3"])
        mark_listbox_as_untouched(app)

        app._remove_file(r"C:\zzz.mp3")

        assert not listbox_was_refreshed(app)
        assert app.status_files.cget("text") == UNTOUCHED_STATUS_TEXT

    def test_removing_from_an_empty_queue_is_a_no_op(self, app):
        mark_listbox_as_untouched(app)

        app._remove_file(r"C:\a.mp3")

        assert app.files == []
        assert session_messages(app) == []
        assert not listbox_was_refreshed(app)

    def test_the_path_must_match_exactly(self, app):
        app.files.append(r"C:\media\clip.mp3")

        app._remove_file('"C:\\media\\clip.mp3"')

        assert app.files == [r"C:\media\clip.mp3"]

    def test_returns_none(self, app):
        app.files.append(r"C:\a.mp3")

        assert app._remove_file(r"C:\a.mp3") is None


class TestRemoveSelected:
    def test_an_empty_selection_leaves_the_queue_untouched(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        app.file_listbox.selection = ()

        app._remove_selected()

        assert app.files == [r"C:\a.mp3", r"C:\b.mp4"]

    def test_an_empty_selection_logs_nothing_and_does_not_refresh(self, app):
        app.files.extend([r"C:\a.mp3"])
        app.file_listbox.selection = ()
        mark_listbox_as_untouched(app)

        app._remove_selected()

        assert session_messages(app) == []
        assert not listbox_was_refreshed(app)

    @pytest.mark.parametrize(
        ("selected_index", "expected_remaining"),
        [
            (0, [r"C:\b.mp4", r"C:\c.wav"]),
            (1, [r"C:\a.mp3", r"C:\c.wav"]),
            (2, [r"C:\a.mp3", r"C:\b.mp4"]),
        ],
    )
    def test_removes_the_file_at_the_selected_index(self, app, selected_index, expected_remaining):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (selected_index,)

        app._remove_selected()

        assert app.files == expected_remaining

    def test_removing_the_first_and_last_of_three_leaves_exactly_the_middle_file(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (0, 2)

        app._remove_selected()

        assert app.files == [r"C:\b.mp4"]

    def test_removing_the_first_two_of_three_leaves_the_last_file(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (0, 1)

        app._remove_selected()

        assert app.files == [r"C:\c.wav"]

    def test_removing_the_last_two_of_three_leaves_the_first_file(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (1, 2)

        app._remove_selected()

        assert app.files == [r"C:\a.mp3"]

    def test_a_selection_given_out_of_order_removes_the_same_files(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (2, 0)

        app._remove_selected()

        assert app.files == [r"C:\b.mp4"]

    def test_selecting_every_row_empties_the_queue(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (0, 1, 2)

        app._remove_selected()

        assert app.files == []
        assert app.status_files.cget("text") == "0 files"

    def test_removes_five_of_seven_scattered_rows_correctly(self, app):
        app.files.extend([f"C:\\clip{index}.mp3" for index in range(7)])
        app.file_listbox.selection = (0, 2, 3, 5, 6)

        app._remove_selected()

        assert app.files == [r"C:\clip1.mp3", r"C:\clip4.mp3"]

    def test_an_out_of_range_index_is_ignored(self, app):
        app.files.extend([r"C:\a.mp3"])
        app.file_listbox.selection = (7,)
        mark_listbox_as_untouched(app)

        app._remove_selected()

        assert app.files == [r"C:\a.mp3"]
        assert session_messages(app) == []
        assert not listbox_was_refreshed(app)

    def test_an_index_equal_to_the_queue_length_is_ignored(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        app.file_listbox.selection = (2,)

        app._remove_selected()

        assert app.files == [r"C:\a.mp3", r"C:\b.mp4"]

    def test_a_negative_index_is_ignored(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        app.file_listbox.selection = (-1,)

        app._remove_selected()

        assert app.files == [r"C:\a.mp3", r"C:\b.mp4"]

    def test_a_valid_index_alongside_an_out_of_range_one_still_removes_the_valid_file(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        app.file_listbox.selection = (0, 9)

        app._remove_selected()

        assert app.files == [r"C:\b.mp4"]

    def test_a_selection_against_an_empty_queue_is_ignored(self, app):
        app.file_listbox.selection = (0,)
        mark_listbox_as_untouched(app)

        app._remove_selected()

        assert app.files == []
        assert session_messages(app) == []
        assert not listbox_was_refreshed(app)

    def test_logs_the_removed_basenames_highest_index_first(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])
        app.file_listbox.selection = (0, 2)

        app._remove_selected()

        assert session_messages(app) == [
            "File removed: c.wav",
            "File removed: a.mp3",
            "Total files in queue: 1",
        ]

    def test_refreshes_the_listbox_after_removing(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        app.file_listbox.selection = (0,)
        mark_listbox_as_untouched(app)

        app._remove_selected()

        assert listbox_was_refreshed(app)
        assert app.file_listbox.items == [" [VID]  b.mp4"]

    def test_returns_none(self, app):
        app.files.append(r"C:\a.mp3")
        app.file_listbox.selection = (0,)

        assert app._remove_selected() is None


class TestClearFiles:
    def test_empties_the_queue(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])

        app._clear_files()

        assert app.files == []

    def test_logs_the_number_of_files_that_were_removed(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4", r"C:\c.wav"])

        app._clear_files()

        assert session_messages(app) == ["Cleared all files (3 removed)"]

    def test_clears_the_listbox_and_resets_the_status_label(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])
        app._refresh_file_list()

        app._clear_files()

        assert app.file_listbox.items == []
        assert app.status_files.cget("text") == "0 files"

    def test_logs_zero_removed_when_the_queue_was_already_empty(self, app):
        app._clear_files()

        assert session_messages(app) == ["Cleared all files (0 removed)"]

    def test_refreshes_the_listbox_even_when_the_queue_was_already_empty(self, app):
        mark_listbox_as_untouched(app)

        app._clear_files()

        assert listbox_was_refreshed(app)
        assert app.status_files.cget("text") == "0 files"

    def test_mutates_the_existing_queue_list_in_place(self, app):
        original_queue = app.files
        app.files.extend([r"C:\a.mp3"])

        app._clear_files()

        assert app.files is original_queue

    def test_a_cleared_queue_can_be_refilled(self, app):
        app.files.extend([r"C:\a.mp3"])

        app._clear_files()
        app._add_files([r"C:\b.mp4"])

        assert app.files == [r"C:\b.mp4"]

    def test_returns_none(self, app):
        assert app._clear_files() is None


class TestRefreshFileList:
    def test_rebuilds_the_listbox_from_scratch(self, app):
        app.file_listbox.items.extend(["stale one", "stale two", "stale three"])
        app.files.append(r"C:\media\clip.mp4")

        app._refresh_file_list()

        assert app.file_listbox.items == [" [VID]  clip.mp4"]

    def test_an_empty_queue_empties_the_listbox(self, app):
        app.file_listbox.items.extend(["stale one", "stale two"])

        app._refresh_file_list()

        assert app.file_listbox.items == []

    def test_calling_twice_does_not_duplicate_the_rows(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])

        app._refresh_file_list()
        app._refresh_file_list()

        assert app.file_listbox.size() == 2

    def test_rows_follow_the_queue_order(self, app):
        app.files.extend([r"C:\c.mp4", r"C:\a.mp3", r"C:\b.wav"])

        app._refresh_file_list()

        assert app.file_listbox.items == [
            " [VID]  c.mp4",
            " [AUD]  a.mp3",
            " [AUD]  b.wav",
        ]

    @pytest.mark.parametrize("extension", SORTED_VIDEO_EXTENSIONS)
    def test_tags_video_extensions_with_vid(self, app, extension):
        app.files.append(f"C:\\media\\clip{extension}")

        app._refresh_file_list()

        assert app.file_listbox.items == [f" [VID]  clip{extension}"]

    @pytest.mark.parametrize("extension", SORTED_AUDIO_EXTENSIONS)
    def test_tags_audio_extensions_with_aud(self, app, extension):
        app.files.append(f"C:\\media\\clip{extension}")

        app._refresh_file_list()

        assert app.file_listbox.items == [f" [AUD]  clip{extension}"]

    @pytest.mark.parametrize("extension", [".MP4", ".Mkv", ".MOV"])
    def test_tags_uppercase_video_extensions_with_vid(self, app, extension):
        app.files.append(f"C:\\media\\clip{extension}")

        app._refresh_file_list()

        assert app.file_listbox.items == [f" [VID]  clip{extension}"]

    @pytest.mark.parametrize("path", [r"C:\media\clip.txt", r"C:\media\clip", r"C:\media\clip."])
    def test_falls_back_to_the_audio_tag_for_anything_that_is_not_a_video(self, app, path):
        app.files.append(path)

        app._refresh_file_list()

        assert app.file_listbox.items[0].startswith(" [AUD]  ")

    def test_shows_the_basename_only(self, app):
        app.files.append(r"C:\Users\Someone\Videos\2026\clip.mp4")

        app._refresh_file_list()

        row = app.file_listbox.items[0]
        assert row == " [VID]  clip.mp4"
        assert "Users" not in row

    def test_handles_forward_slash_paths_as_delivered_by_tkdnd(self, app):
        app.files.append("C:/Users/Someone/Videos/clip.mp4")

        app._refresh_file_list()

        assert app.file_listbox.items == [" [VID]  clip.mp4"]

    def test_keeps_spaces_and_punctuation_in_the_displayed_name(self, app):
        app.files.append(r"C:\media\Bob's take - 01 (final).mp4")

        app._refresh_file_list()

        assert app.file_listbox.items == [" [VID]  Bob's take - 01 (final).mp4"]

    def test_a_bare_filename_without_a_folder_is_shown_as_is(self, app):
        app.files.append("clip.mp3")

        app._refresh_file_list()

        assert app.file_listbox.items == [" [AUD]  clip.mp3"]

    @pytest.mark.parametrize(
        ("file_count", "expected_status"),
        [
            (0, "0 files"),
            (1, "1 file"),
            (2, "2 files"),
            (3, "3 files"),
            (10, "10 files"),
        ],
    )
    def test_the_status_label_pluralises_the_queue_count(self, app, file_count, expected_status):
        app.files.extend([f"C:\\clip{index}.mp3" for index in range(file_count)])

        app._refresh_file_list()

        assert app.status_files.cget("text") == expected_status

    def test_writes_nothing_to_the_session_log(self, app):
        app.files.extend([r"C:\a.mp3", r"C:\b.mp4"])

        app._refresh_file_list()

        assert session_messages(app) == []

    def test_writes_nothing_to_the_visible_log(self, app):
        app.files.append(r"C:\a.mp3")

        app._refresh_file_list()

        assert app.log_text.value == ""

    def test_returns_none(self, app):
        assert app._refresh_file_list() is None
