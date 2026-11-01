import json
import os
import time

import pytest

import engine
import settings
import system_shell

FIXED_MTIME = time.mktime((2026, 11, 1, 17, 20, 5, 0, 0, -1))


@pytest.fixture
def source(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    path = media / "audio_message.m4a"
    path.write_bytes(b"audio")
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return str(path)


@pytest.fixture
def output(tmp_path):
    folder = tmp_path / "transcripts"
    folder.mkdir()
    return folder


class TestSaveTranscriptKeepingExistingOnes:
    def test_a_free_name_is_used_as_is(self, source, output):
        assert engine.save_transcript(source, "nuevo", str(output), keep_existing=True) == "audio_message.txt"

    def test_an_older_transcript_with_the_same_name_is_left_untouched(self, source, output):
        (output / "audio_message.txt").write_text("viejo", encoding="utf-8")

        engine.save_transcript(source, "nuevo", str(output), keep_existing=True)

        assert (output / "audio_message.txt").read_text(encoding="utf-8") == "viejo"

    def test_the_new_transcript_is_named_after_the_audio_file_date(self, source, output):
        (output / "audio_message.txt").write_text("viejo", encoding="utf-8")

        saved = engine.save_transcript(source, "nuevo", str(output), keep_existing=True)

        assert saved == "audio_message 2026-11-01 17-20-05.txt"
        assert (output / saved).read_text(encoding="utf-8") == "nuevo"

    def test_a_taken_dated_name_gets_a_copy_number(self, source, output):
        (output / "audio_message.txt").write_text("uno", encoding="utf-8")
        (output / "audio_message 2026-11-01 17-20-05.txt").write_text("dos", encoding="utf-8")

        saved = engine.save_transcript(source, "tres", str(output), keep_existing=True)

        assert saved == "audio_message 2026-11-01 17-20-05 (2).txt"

    def test_a_missing_source_falls_back_to_the_current_time(self, tmp_path, output, monkeypatch):
        monkeypatch.setattr(engine.time, "time", lambda: FIXED_MTIME)
        (output / "gone.txt").write_text("viejo", encoding="utf-8")

        saved = engine.save_transcript(str(tmp_path / "gone.m4a"), "nuevo", str(output), keep_existing=True)

        assert saved == "gone 2026-11-01 17-20-05.txt"

    def test_without_the_flag_the_old_transcript_is_overwritten_as_before(self, source, output):
        (output / "audio_message.txt").write_text("viejo", encoding="utf-8")

        saved = engine.save_transcript(source, "nuevo", str(output))

        assert saved == "audio_message.txt"
        assert (output / saved).read_text(encoding="utf-8") == "nuevo"


class TestSettingsFile:
    def test_a_missing_file_loads_as_empty(self, tmp_path):
        assert settings.load_settings(str(tmp_path / "settings.json")) == {}

    def test_saved_values_load_back(self, tmp_path):
        path = str(tmp_path / "nested" / "settings.json")

        settings.save_settings({"watch_downloads": True}, path)

        assert settings.load_settings(path) == {"watch_downloads": True}

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
    def test_a_broken_file_loads_as_empty(self, tmp_path, content):
        path = tmp_path / "settings.json"
        path.write_text(content, encoding="utf-8")

        assert settings.load_settings(str(path)) == {}

    def test_saving_leaves_no_temporary_file_behind(self, tmp_path):
        path = tmp_path / "settings.json"

        settings.save_settings({"watch_downloads": False}, str(path))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["settings.json"]
        assert json.loads(path.read_text(encoding="utf-8")) == {"watch_downloads": False}


class FakeRunKey:
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_READ = 1
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self, values=None):
        self.values = dict(values or {})
        self.opened = []

    def OpenKey(self, root, path, reserved, access):
        self.opened.append((root, path, access))
        return FakeRunKey(self.values)

    def QueryValueEx(self, key, name):
        if name not in key.values:
            raise FileNotFoundError(name)
        return key.values[name], self.REG_SZ

    def SetValueEx(self, key, name, reserved, kind, value):
        key.values[name] = value

    def DeleteValue(self, key, name):
        if name not in key.values:
            raise FileNotFoundError(name)
        del key.values[name]


@pytest.fixture
def registry(monkeypatch):
    fake = FakeWinreg()
    monkeypatch.setattr(system_shell, "winreg", fake)
    return fake


class TestStartWithWindows:
    def test_reports_disabled_when_there_is_no_entry(self, registry):
        assert system_shell.is_start_with_windows_enabled() is False

    def test_enabling_writes_the_minimized_launch_command_under_the_users_run_key(self, registry):
        system_shell.set_start_with_windows(True)

        assert registry.values == {"Sotvox": system_shell.startup_command()}
        assert registry.opened[-1][:2] == ("HKCU", r"Software\Microsoft\Windows\CurrentVersion\Run")
        assert system_shell.is_start_with_windows_enabled() is True

    def test_disabling_removes_the_entry(self, registry):
        registry.values["Sotvox"] = "old"

        system_shell.set_start_with_windows(False)

        assert registry.values == {}

    def test_disabling_when_nothing_was_registered_is_harmless(self, registry):
        assert system_shell.set_start_with_windows(False) is True

    def test_other_programs_entries_are_left_alone(self, registry):
        registry.values["OneDrive"] = "onedrive.exe"

        system_shell.set_start_with_windows(True)
        system_shell.set_start_with_windows(False)

        assert registry.values == {"OneDrive": "onedrive.exe"}

    def test_the_installed_app_launches_itself_minimized(self, monkeypatch):
        monkeypatch.setattr(system_shell, "IS_FROZEN", True)
        monkeypatch.setattr(system_shell.sys, "executable", r"C:\Apps\Sotvox\Sotvox.exe")

        assert system_shell.startup_command() == r'"C:\Apps\Sotvox\Sotvox.exe" --minimized'

    def test_a_source_checkout_launches_main_py_minimized(self, monkeypatch):
        monkeypatch.setattr(system_shell, "IS_FROZEN", False)

        command = system_shell.startup_command()

        assert command.endswith('main.py" --minimized')

    def test_without_a_registry_nothing_is_enabled_or_written(self, monkeypatch):
        monkeypatch.setattr(system_shell, "winreg", None)

        assert system_shell.is_start_with_windows_enabled() is False
        assert system_shell.set_start_with_windows(True) is False


class TestSingleInstanceAndDownloadsElsewhere:
    def test_every_launch_owns_the_app_outside_windows(self, monkeypatch):
        monkeypatch.setattr(system_shell, "IS_WINDOWS", False)

        assert system_shell.claim_single_instance() is True
        assert system_shell.focus_running_instance() is False

    def test_downloads_falls_back_to_the_home_folder_outside_windows(self, monkeypatch, tmp_path):
        monkeypatch.setattr(system_shell, "IS_WINDOWS", False)
        monkeypatch.setattr(system_shell.os.path, "expanduser", lambda path: str(tmp_path))

        assert system_shell.downloads_folder() == os.path.join(str(tmp_path), "Downloads")

    def test_downloads_falls_back_when_windows_cannot_answer(self, monkeypatch, tmp_path):
        def refuse(folder_id):
            raise OSError("no shell")

        monkeypatch.setattr(system_shell, "IS_WINDOWS", True)
        monkeypatch.setattr(system_shell, "_known_folder_path", refuse)
        monkeypatch.setattr(system_shell.os.path, "expanduser", lambda path: str(tmp_path))

        assert system_shell.downloads_folder() == os.path.join(str(tmp_path), "Downloads")

    @pytest.mark.skipif(os.name != "nt", reason="needs the Windows shell")
    def test_the_real_downloads_folder_exists_on_this_machine(self):
        assert os.path.isdir(system_shell.downloads_folder())
