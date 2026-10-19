import os
import sys

import pytest

import constants


class TestResourceDirResolution:
    def test_uses_meipass_when_frozen(self, monkeypatch):
        monkeypatch.setattr(constants, "IS_FROZEN", True)
        monkeypatch.setattr(sys, "_MEIPASS", r"C:\Temp\_MEI123", raising=False)

        assert constants._resolve_resource_dir() == r"C:\Temp\_MEI123"

    def test_falls_back_to_executable_directory_when_frozen_without_meipass(self, monkeypatch):
        monkeypatch.setattr(constants, "IS_FROZEN", True)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        monkeypatch.setattr(sys, "executable", r"C:\Program Files\Sotvox\Sotvox.exe")

        assert constants._resolve_resource_dir() == r"C:\Program Files\Sotvox"

    def test_resolves_to_repository_root_when_running_from_source(self, monkeypatch):
        monkeypatch.setattr(constants, "IS_FROZEN", False)

        resolved = constants._resolve_resource_dir()

        assert os.path.isdir(os.path.join(resolved, "src"))
        assert os.path.basename(resolved) != "src"

    def test_source_layout_places_assets_beside_src(self, monkeypatch):
        monkeypatch.setattr(constants, "IS_FROZEN", False)

        resolved = constants._resolve_resource_dir()

        assert os.path.isdir(os.path.join(resolved, "assets"))


class TestAppDataDirResolution:
    def test_prefers_localappdata(self, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Someone\AppData\Local")

        assert constants._resolve_app_data_dir() == r"C:\Users\Someone\AppData\Local\Sotvox"

    def test_falls_back_to_home_when_localappdata_missing(self, monkeypatch):
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        expected = os.path.join(os.path.expanduser("~"), "Sotvox")
        assert constants._resolve_app_data_dir() == expected

    def test_falls_back_to_home_when_localappdata_is_empty(self, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", "")

        expected = os.path.join(os.path.expanduser("~"), "Sotvox")
        assert constants._resolve_app_data_dir() == expected

    def test_app_data_dir_is_namespaced_by_app_name(self, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Local")

        assert os.path.basename(constants._resolve_app_data_dir()) == constants.APP_NAME


class TestDerivedPaths:
    def test_log_dir_sits_under_app_data(self):
        assert constants.LOG_DIR == os.path.join(constants.APP_DATA_DIR, "logs")

    def test_cuda_dir_sits_under_app_data(self):
        assert constants.CUDA_DIR == os.path.join(constants.APP_DATA_DIR, "cuda")

    def test_assets_dir_sits_under_resource_dir(self):
        assert constants.ASSETS_DIR == os.path.join(constants.RESOURCE_DIR, "assets")

    def test_icon_path_points_at_the_ico_inside_assets(self):
        assert constants.ICON_PATH == os.path.join(constants.ASSETS_DIR, "sotvox.ico")

    @pytest.mark.skipif(sys.platform == "darwin", reason="Windows default")
    def test_default_output_dir_is_the_learning_transcripts_folder(self):
        assert constants.DEFAULT_OUTPUT_DIR == r"C:\dasbanq\dasbanq-learning\transcripts"

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS default")
    def test_default_output_dir_on_macos_is_documents_sotvox_transcripts(self):
        expected = os.path.join(os.path.expanduser("~"), "Documents", "sotvox-transcripts")
        assert constants.DEFAULT_OUTPUT_DIR == expected

    def test_log_and_cuda_dirs_are_distinct(self):
        assert constants.LOG_DIR != constants.CUDA_DIR

    def test_output_dir_is_outside_app_data(self):
        assert not constants.DEFAULT_OUTPUT_DIR.startswith(constants.APP_DATA_DIR)


class TestShippedAssetsExist:
    @pytest.mark.integration
    def test_icon_file_is_present_in_the_repository(self):
        assert os.path.isfile(constants.ICON_PATH)

    @pytest.mark.integration
    @pytest.mark.parametrize("size", [16, 32, 48])
    def test_taskbar_png_sizes_are_present(self, size):
        assert os.path.isfile(os.path.join(constants.ASSETS_DIR, f"sotvox_{size}.png"))


class TestSupportedExtensions:
    def test_every_extension_starts_with_a_dot(self):
        assert all(extension.startswith(".") for extension in constants.SUPPORTED_EXTENSIONS)

    def test_every_extension_is_lowercase(self):
        assert all(extension == extension.lower() for extension in constants.SUPPORTED_EXTENSIONS)

    def test_no_extension_contains_a_wildcard_or_space(self):
        assert all(
            "*" not in extension and " " not in extension
            for extension in constants.SUPPORTED_EXTENSIONS
        )

    def test_video_extensions_are_a_subset_of_supported(self):
        assert constants.VIDEO_EXTENSIONS <= constants.SUPPORTED_EXTENSIONS

    def test_audio_and_video_partition_the_supported_set(self):
        audio = constants.SUPPORTED_EXTENSIONS - constants.VIDEO_EXTENSIONS

        assert audio == {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac"}
        assert audio | constants.VIDEO_EXTENSIONS == constants.SUPPORTED_EXTENSIONS
        assert audio & constants.VIDEO_EXTENSIONS == set()

    def test_video_extensions_are_exactly_the_documented_set(self):
        assert constants.VIDEO_EXTENSIONS == {
            ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".ts", ".flv"}

    def test_supported_extension_count_is_pinned(self):
        assert len(constants.SUPPORTED_EXTENSIONS) == 15


class TestLanguageMap:
    def test_every_code_is_a_two_letter_iso_639_1_code(self):
        assert all(
            len(code) == 2 and code.isalpha() and code.islower()
            for code in constants.LANG_MAP.values()
        )

    def test_language_codes_are_unique(self):
        codes = list(constants.LANG_MAP.values())

        assert len(codes) == len(set(codes))

    def test_language_names_are_capitalised_english_names(self):
        assert all(name[0].isupper() for name in constants.LANG_MAP)

    def test_auto_detect_is_not_a_language_entry(self):
        assert "Auto-detect" not in constants.LANG_MAP

    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("Spanish", "es"),
            ("English", "en"),
            ("Portuguese", "pt"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Japanese", "ja"),
            ("Chinese", "zh"),
            ("Korean", "ko"),
        ],
    )
    def test_known_language_maps_to_expected_code(self, name, code):
        assert constants.LANG_MAP[name] == code

    def test_language_count_is_pinned(self):
        assert len(constants.LANG_MAP) == 9


class TestModuleIdentity:
    def test_app_name_is_sotvox(self):
        assert constants.APP_NAME == "Sotvox"

    def test_is_frozen_is_false_under_the_test_runner(self):
        assert constants.IS_FROZEN is False
