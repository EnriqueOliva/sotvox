import ctypes
import os
import struct
import subprocess
import sys
import time
import types

import pytest

import system_shell

DROP_EFFECT_FORMAT_ID = 49999
FAILING_HRESULT = -2147467259
HEADER_SIZE = system_shell.DROPFILES_HEADER.size

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="needs the Windows shell")


def decode_file_list(payload):
    return payload[HEADER_SIZE:].decode("utf-16-le").split("\0")[:-2]


class FakeShellApi:
    def __init__(self, unresolvable=(), open_result=0):
        self.unresolvable = set(unresolvable)
        self.open_result = open_result
        self.created_paths = []
        self.freed = []
        self.open_calls = []
        self.co_initialize_calls = 0
        self.events = []
        self.paths_by_id = {}
        self._next_id = 1000

    def CoInitialize(self, reserved):
        self.co_initialize_calls += 1
        self.events.append("co_initialize")
        return 0

    def ILCreateFromPathW(self, path):
        self.created_paths.append(path)
        self.events.append("create")
        if path in self.unresolvable:
            return None
        self._next_id += 1
        self.paths_by_id[self._next_id] = path
        return self._next_id

    def ILFree(self, item_list):
        self.freed.append(item_list)

    def SHOpenFolderAndSelectItems(self, folder, count, selection, flags):
        self.events.append("open")
        self.open_calls.append({
            "folder": self.paths_by_id[folder],
            "items": [self.paths_by_id[selection[index]] for index in range(count)],
            "flags": flags,
        })
        return self.open_result


class FakeClipboardApi:
    def __init__(self, busy_attempts=0, refused_formats=(), allocation_error=False):
        self.busy_attempts = busy_attempts
        self.refused_formats = set(refused_formats)
        self.allocation_error = allocation_error
        self.open_attempts = 0
        self.owners = []
        self.events = []
        self.buffers = {}
        self.clipboard = {}
        self.freed = []
        self.registered_names = []
        self._next_handle = 500

    def OpenClipboard(self, owner_window):
        self.open_attempts += 1
        self.owners.append(owner_window)
        if self.open_attempts <= self.busy_attempts:
            return 0
        self.events.append("open")
        return 1

    def EmptyClipboard(self):
        self.events.append("empty")
        return 1

    def RegisterClipboardFormatW(self, name):
        self.registered_names.append(name)
        return DROP_EFFECT_FORMAT_ID

    def SetClipboardData(self, clipboard_format, handle):
        self.events.append(("set", clipboard_format))
        if clipboard_format in self.refused_formats:
            return None
        self.clipboard[clipboard_format] = self.buffers[handle].raw
        return handle

    def CloseClipboard(self):
        self.events.append("close")
        return 1

    def GlobalAlloc(self, flags, size):
        if self.allocation_error:
            return None
        self._next_handle += 1
        self.buffers[self._next_handle] = ctypes.create_string_buffer(size)
        return self._next_handle

    def GlobalLock(self, handle):
        return ctypes.addressof(self.buffers[handle])

    def GlobalUnlock(self, handle):
        return 1

    def GlobalFree(self, handle):
        self.freed.append(handle)
        return None


class SleepRecorder:
    def __init__(self):
        self.calls = []

    def sleep(self, seconds):
        self.calls.append(seconds)


class SubprocessRecorder:
    SubprocessError = subprocess.SubprocessError

    def __init__(self, returncode=0, error=None):
        self.returncode = returncode
        self.error = error
        self.commands = []

    def run(self, command, **keyword_arguments):
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(returncode=self.returncode)


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(system_shell, "IS_WINDOWS", True)
    monkeypatch.setattr(system_shell, "IS_MAC", False)


@pytest.fixture
def on_mac(monkeypatch):
    monkeypatch.setattr(system_shell, "IS_WINDOWS", False)
    monkeypatch.setattr(system_shell, "IS_MAC", True)


@pytest.fixture
def shell_api(monkeypatch, on_windows):
    def install(**options):
        api = FakeShellApi(**options)
        monkeypatch.setattr(system_shell, "_load_shell_api", lambda: (api, api))
        return api

    return install


@pytest.fixture
def clipboard_api(monkeypatch, on_windows):
    sleeper = SleepRecorder()
    monkeypatch.setattr(system_shell, "time", sleeper)

    def install(**options):
        api = FakeClipboardApi(**options)
        api.sleeper = sleeper
        monkeypatch.setattr(system_shell, "_load_clipboard_api", lambda: (api, api))
        return api

    return install


@pytest.fixture
def mac_subprocess(monkeypatch, on_mac):
    def install(**options):
        recorder = SubprocessRecorder(**options)
        monkeypatch.setattr(system_shell, "subprocess", recorder)
        return recorder

    return install


class TestDropFilesPayload:
    def test_the_file_list_starts_right_after_the_header(self, tmp_path):
        payload = system_shell.build_drop_files_payload([str(tmp_path / "a.txt")])

        offset, _x, _y, _non_client, _wide = system_shell.DROPFILES_HEADER.unpack_from(payload)

        assert offset == HEADER_SIZE

    def test_the_names_are_flagged_as_wide_characters_with_no_drop_point(self, tmp_path):
        payload = system_shell.build_drop_files_payload([str(tmp_path / "a.txt")])

        _offset, x, y, non_client, wide = system_shell.DROPFILES_HEADER.unpack_from(payload)

        assert (x, y, non_client, wide) == (0, 0, 0, 1)

    def test_every_name_is_null_terminated_and_the_list_ends_with_an_extra_null(self, tmp_path):
        first, second = str(tmp_path / "a.txt"), str(tmp_path / "b.txt")

        payload = system_shell.build_drop_files_payload([first, second])

        assert payload[HEADER_SIZE:].decode("utf-16-le") == f"{first}\0{second}\0\0"

    def test_relative_paths_are_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        payload = system_shell.build_drop_files_payload(["notes.txt"])

        assert decode_file_list(payload) == [os.path.join(str(tmp_path), "notes.txt")]

    def test_names_outside_ascii_survive(self, tmp_path):
        name = str(tmp_path / "reunión de mañana 日本.txt")

        payload = system_shell.build_drop_files_payload([name])

        assert decode_file_list(payload) == [name]


class TestCopyFilesToClipboardOnWindows:
    def test_puts_the_files_on_the_clipboard_as_a_file_drop_list(self, clipboard_api, tmp_path):
        api = clipboard_api()
        paths = [str(tmp_path / "one.txt"), str(tmp_path / "two.txt")]

        assert system_shell.copy_files_to_clipboard(paths, owner_window=77) is True

        assert decode_file_list(api.clipboard[system_shell.CF_HDROP]) == paths

    def test_marks_the_files_as_copied_rather_than_cut(self, clipboard_api, tmp_path):
        api = clipboard_api()

        system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77)

        assert api.registered_names == ["Preferred DropEffect"]
        assert struct.unpack("<I", api.clipboard[DROP_EFFECT_FORMAT_ID]) == (system_shell.DROPEFFECT_COPY,)

    def test_opens_with_the_owner_window_and_empties_before_writing(self, clipboard_api, tmp_path):
        api = clipboard_api()

        system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77)

        assert api.owners == [77]
        assert api.events == ["open", "empty", ("set", system_shell.CF_HDROP),
                              ("set", DROP_EFFECT_FORMAT_ID), "close"]

    def test_the_clipboard_keeps_the_memory_it_accepted(self, clipboard_api, tmp_path):
        api = clipboard_api()

        system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77)

        assert api.freed == []

    def test_retries_while_another_program_holds_the_clipboard(self, clipboard_api, tmp_path):
        api = clipboard_api(busy_attempts=3)

        assert system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77) is True

        assert api.open_attempts == 4
        assert api.sleeper.calls == [system_shell.OPEN_CLIPBOARD_RETRY_SECONDS] * 3

    def test_gives_up_after_the_attempt_limit_without_touching_the_contents(self, clipboard_api, tmp_path):
        api = clipboard_api(busy_attempts=system_shell.OPEN_CLIPBOARD_ATTEMPTS)

        assert system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77) is False

        assert api.open_attempts == system_shell.OPEN_CLIPBOARD_ATTEMPTS
        assert len(api.sleeper.calls) == system_shell.OPEN_CLIPBOARD_ATTEMPTS - 1
        assert api.events == []

    def test_frees_the_memory_and_reports_failure_when_the_clipboard_refuses_it(self, clipboard_api, tmp_path):
        api = clipboard_api(refused_formats={system_shell.CF_HDROP})

        assert system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77) is False

        assert len(api.freed) == 1
        assert api.events[-1] == "close"
        assert ("set", DROP_EFFECT_FORMAT_ID) not in api.events

    def test_closes_the_clipboard_even_when_the_memory_cannot_be_allocated(self, clipboard_api, tmp_path):
        api = clipboard_api(allocation_error=True)

        with pytest.raises(OSError):
            system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")], owner_window=77)

        assert api.events[-1] == "close"

    def test_an_empty_list_never_touches_the_clipboard(self, clipboard_api):
        api = clipboard_api()

        assert system_shell.copy_files_to_clipboard([], owner_window=77) is False

        assert api.open_attempts == 0


class TestRevealFilesOnWindows:
    def test_opens_the_folder_with_every_file_selected(self, shell_api, tmp_path):
        api = shell_api()
        paths = [str(tmp_path / "one.txt"), str(tmp_path / "two.txt")]

        assert system_shell.reveal_files(str(tmp_path), paths) is True

        assert api.open_calls == [{"folder": str(tmp_path), "items": paths, "flags": 0}]

    def test_initializes_com_before_talking_to_the_shell(self, shell_api, tmp_path):
        api = shell_api()

        system_shell.reveal_files(str(tmp_path), [str(tmp_path / "one.txt")])

        assert api.events[0] == "co_initialize"

    def test_frees_every_item_list_it_created(self, shell_api, tmp_path):
        api = shell_api()

        system_shell.reveal_files(str(tmp_path), [str(tmp_path / "one.txt"), str(tmp_path / "two.txt")])

        assert sorted(api.freed) == sorted(api.paths_by_id)

    def test_files_the_shell_cannot_resolve_are_left_out(self, shell_api, tmp_path):
        missing = str(tmp_path / "gone.txt")
        kept = str(tmp_path / "kept.txt")
        api = shell_api(unresolvable={missing})

        assert system_shell.reveal_files(str(tmp_path), [missing, kept]) is True

        assert api.open_calls[0]["items"] == [kept]

    def test_reports_failure_without_opening_anything_when_no_file_resolves(self, shell_api, tmp_path):
        missing = str(tmp_path / "gone.txt")
        api = shell_api(unresolvable={missing})

        assert system_shell.reveal_files(str(tmp_path), [missing]) is False

        assert api.open_calls == []
        assert sorted(api.freed) == sorted(api.paths_by_id)

    def test_reports_failure_when_the_shell_returns_an_error(self, shell_api, tmp_path):
        api = shell_api(open_result=FAILING_HRESULT)

        assert system_shell.reveal_files(str(tmp_path), [str(tmp_path / "one.txt")]) is False

        assert sorted(api.freed) == sorted(api.paths_by_id)

    def test_relative_paths_are_resolved_before_reaching_the_shell(self, shell_api, tmp_path, monkeypatch):
        api = shell_api()
        monkeypatch.chdir(tmp_path)

        system_shell.reveal_files(".", ["one.txt"])

        assert api.created_paths == [str(tmp_path), os.path.join(str(tmp_path), "one.txt")]

    def test_an_empty_list_never_reaches_the_shell(self, shell_api, tmp_path):
        api = shell_api()

        assert system_shell.reveal_files(str(tmp_path), []) is False

        assert api.events == []


class TestMacScripts:
    def test_reveal_selects_the_files_in_finder(self, mac_subprocess, tmp_path):
        recorder = mac_subprocess()
        paths = [str(tmp_path / "one.txt"), str(tmp_path / "two.txt")]

        assert system_shell.reveal_files(str(tmp_path), paths) is True

        assert recorder.commands == [
            ["osascript", "-l", "JavaScript", "-e", system_shell.MAC_REVEAL_SCRIPT, *paths]]

    def test_copy_puts_the_files_on_the_pasteboard(self, mac_subprocess, tmp_path):
        recorder = mac_subprocess()
        paths = [str(tmp_path / "one.txt")]

        assert system_shell.copy_files_to_clipboard(paths) is True

        assert recorder.commands == [
            ["osascript", "-l", "JavaScript", "-e", system_shell.MAC_COPY_SCRIPT, *paths]]

    def test_a_failing_script_is_reported(self, mac_subprocess, tmp_path):
        mac_subprocess(returncode=1)

        assert system_shell.copy_files_to_clipboard([str(tmp_path / "one.txt")]) is False

    @pytest.mark.parametrize("error", [FileNotFoundError("osascript"),
                                       subprocess.TimeoutExpired("osascript", 15)])
    def test_a_script_that_cannot_run_is_reported(self, mac_subprocess, tmp_path, error):
        mac_subprocess(error=error)

        assert system_shell.reveal_files(str(tmp_path), [str(tmp_path / "one.txt")]) is False


class TestOtherPlatforms:
    def test_nothing_is_attempted_on_an_unsupported_platform(self, monkeypatch, tmp_path):
        monkeypatch.setattr(system_shell, "IS_WINDOWS", False)
        monkeypatch.setattr(system_shell, "IS_MAC", False)
        paths = [str(tmp_path / "one.txt")]

        assert system_shell.reveal_files(str(tmp_path), paths) is False
        assert system_shell.copy_files_to_clipboard(paths) is False


def read_clipboard_file_list():
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    shell32 = ctypes.WinDLL("shell32")
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    query_count = 0xFFFFFFFF
    assert user32.OpenClipboard(None)
    try:
        handle = user32.GetClipboardData(system_shell.CF_HDROP)
        assert handle, "no file list on the clipboard"
        names = []
        for index in range(shell32.DragQueryFileW(handle, query_count, None, 0)):
            length = shell32.DragQueryFileW(handle, index, None, 0) + 1
            buffer = ctypes.create_unicode_buffer(length)
            shell32.DragQueryFileW(handle, index, buffer, length)
            names.append(buffer.value)
        return names
    finally:
        user32.CloseClipboard()


def selected_names_in_explorer(folder):
    script = (
        "$shell = New-Object -ComObject Shell.Application; "
        "foreach ($window in @($shell.Windows())) { try { "
        f"if ($window.Document.Folder.Self.Path -eq '{folder}') {{ "
        "foreach ($item in $window.Document.SelectedItems()) { $item.Name }; $window.Quit() } "
        "} catch {} }")
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                               capture_output=True, text=True, encoding="utf-8", timeout=60)
    return [line for line in completed.stdout.splitlines() if line.strip()]


@windows_only
@pytest.mark.gui
@pytest.mark.integration
@pytest.mark.allow_desktop_shell
def test_copied_transcripts_can_be_read_back_from_the_real_clipboard(tmp_path):
    paths = []
    for name in ("primera nota.txt", "reunión.txt"):
        path = tmp_path / name
        path.write_text("hola", encoding="utf-8")
        paths.append(str(path))

    assert system_shell.copy_files_to_clipboard(paths) is True

    assert read_clipboard_file_list() == paths


@windows_only
@pytest.mark.gui
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.allow_desktop_shell
def test_explorer_opens_with_exactly_the_transcripts_selected(tmp_path):
    folder = tmp_path / "salida"
    folder.mkdir()
    for name in ("one.txt", "two.txt", "other.txt"):
        (folder / name).write_text("x", encoding="utf-8")

    assert system_shell.reveal_files(str(folder), [str(folder / "one.txt"), str(folder / "two.txt")]) is True

    deadline = time.time() + 15
    selected = []
    while not selected and time.time() < deadline:
        time.sleep(0.5)
        selected = selected_names_in_explorer(str(folder))
    assert sorted(selected) == ["one.txt", "two.txt"]
