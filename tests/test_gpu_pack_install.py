"""Unit tests for gpu_pack._extract_libraries and gpu_pack.install."""

import os
import zipfile
from pathlib import Path

import doubles
import pytest

import gpu_pack

CUBLAS_LIBRARY = "cublas64_12.dll"
CUDNN_LIBRARY = "cudnn64_9.dll"
CUBLAS_PACKAGE = "nvidia-cublas-cu12"
CUDNN_PACKAGE = "nvidia-cudnn-cu12"
CUBLAS_WHEEL_FILENAME = "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"
CUDNN_WHEEL_FILENAME = "nvidia_cudnn_cu12-9.1.0.70-py3-none-win_amd64.whl"
CUBLAS_WHEEL_SIZE = 400
CUDNN_WHEEL_SIZE = 600
COMBINED_WHEEL_SIZE = CUBLAS_WHEEL_SIZE + CUDNN_WHEEL_SIZE
DOWNLOAD_DIRECTORY_NAME = "_download"
INCOMPLETE_MESSAGE = "GPU libraries were downloaded but are incomplete"
PAYLOAD_LARGER_THAN_ONE_CHUNK = (bytes(range(256)) * 2100)[:gpu_pack.DOWNLOAD_CHUNK_BYTES * 2 + 123]
PAYLOAD_EXACTLY_ONE_CHUNK = (bytes(range(256)) * 1025)[:gpu_pack.DOWNLOAD_CHUNK_BYTES]


def build_archive(directory, members, name="wheel.whl"):
    return str(doubles.build_wheel_archive(Path(directory) / name, members))


def default_wheels(cublas_size=CUBLAS_WHEEL_SIZE, cudnn_size=CUDNN_WHEEL_SIZE):
    return {
        CUBLAS_PACKAGE: doubles.wheel_file(CUBLAS_WHEEL_FILENAME, size=cublas_size),
        CUDNN_PACKAGE: doubles.wheel_file(CUDNN_WHEEL_FILENAME, size=cudnn_size),
    }


class ProgressRecorder:
    def __init__(self):
        self.events = []

    def __call__(self, current_bytes, total_bytes):
        self.events.append((current_bytes, total_bytes))

    @property
    def current_values(self):
        return [current_bytes for current_bytes, _total in self.events]


class RecordingWheelResolver:
    def __init__(self, wheels_by_package=None, error=None):
        self.wheels_by_package = wheels_by_package if wheels_by_package is not None else default_wheels()
        self.error = error
        self.calls = []

    def __call__(self, package, major, log=None):
        self.calls.append({"package": package, "major": major, "log": log})
        if self.error is not None:
            raise self.error
        return self.wheels_by_package[package]

    @property
    def requested_packages(self):
        return [call["package"] for call in self.calls]


class RecordingDownloader:
    def __init__(self, progress_by_filename=None, archive_members_by_filename=None,
                 payload=b"wheel-payload", error=None):
        self.progress_by_filename = progress_by_filename or {}
        self.archive_members_by_filename = archive_members_by_filename or {}
        self.payload = payload
        self.error = error
        self.calls = []

    def __call__(self, url, destination_path, on_progress=None, total_bytes=0, is_cancelled=None):
        filename = os.path.basename(destination_path)
        self.calls.append({
            "url": url,
            "destination_path": destination_path,
            "filename": filename,
            "total_bytes": total_bytes,
            "is_cancelled": is_cancelled,
            "download_directory": os.path.dirname(destination_path),
            "download_directory_existed": os.path.isdir(os.path.dirname(destination_path)),
        })
        members = self.archive_members_by_filename.get(filename)
        if members is None:
            written_bytes = len(self.payload)
            with open(destination_path, "wb") as handle:
                handle.write(self.payload)
        else:
            doubles.build_wheel_archive(destination_path, members)
            written_bytes = Path(destination_path).stat().st_size
        for cumulative_bytes in self.progress_by_filename.get(filename, (total_bytes,)):
            if on_progress:
                on_progress(cumulative_bytes, total_bytes)
        if self.error is not None:
            raise self.error
        return written_bytes

    @property
    def requested_filenames(self):
        return [call["filename"] for call in self.calls]


class RecordingExtractor:
    def __init__(self, cuda_directory, libraries_per_call=None, extra_directory_name=None,
                 error=None, return_value=None):
        self.cuda_directory = str(cuda_directory)
        if libraries_per_call is None:
            self.libraries_per_call = [[CUBLAS_LIBRARY], [CUDNN_LIBRARY]]
        else:
            self.libraries_per_call = [list(entry) for entry in libraries_per_call]
        self.extra_directory_name = extra_directory_name
        self.error = error
        self.return_value = return_value
        self.calls = []

    def __call__(self, wheel_path, log=None):
        index = len(self.calls)
        download_directory = os.path.dirname(wheel_path)
        self.calls.append({
            "wheel_path": wheel_path,
            "log": log,
            "wheel_existed": os.path.exists(wheel_path),
            "download_directory_contents": sorted(os.listdir(download_directory)),
        })
        created = self.libraries_per_call[index] if index < len(self.libraries_per_call) else []
        for library in created:
            with open(os.path.join(self.cuda_directory, library), "wb") as handle:
                handle.write(b"MZ")
        if self.extra_directory_name:
            (Path(download_directory) / self.extra_directory_name).mkdir(exist_ok=True)
        if self.error is not None:
            raise self.error
        if self.return_value is not None:
            return self.return_value
        return len(created)

    @property
    def extracted_wheel_names(self):
        return [os.path.basename(call["wheel_path"]) for call in self.calls]


class InstallDoubles:
    def __init__(self, cuda_directory, resolver, downloader, extractor):
        self.cuda_directory = str(cuda_directory)
        self.resolver = resolver
        self.downloader = downloader
        self.extractor = extractor

    @property
    def download_directory(self):
        return os.path.join(self.cuda_directory, DOWNLOAD_DIRECTORY_NAME)


def patch_wheel_source(monkeypatch, resolver=None, downloader=None):
    resolver = resolver if resolver is not None else RecordingWheelResolver()
    downloader = downloader if downloader is not None else RecordingDownloader()
    monkeypatch.setattr(gpu_pack, "_resolve_wheel", resolver)
    monkeypatch.setattr(gpu_pack, "_download", downloader)
    return resolver, downloader


def patch_install_dependencies(monkeypatch, cuda_directory, resolver=None, downloader=None, extractor=None):
    resolver, downloader = patch_wheel_source(monkeypatch, resolver, downloader)
    extractor = extractor if extractor is not None else RecordingExtractor(cuda_directory)
    monkeypatch.setattr(gpu_pack, "_extract_libraries", extractor)
    return InstallDoubles(cuda_directory, resolver, downloader, extractor)


@pytest.fixture
def install_doubles(cuda_dir, monkeypatch):
    return patch_install_dependencies(monkeypatch, cuda_dir)


class TestExtractLibrariesMemberSelection:
    def test_extracts_a_dll_that_lives_under_a_bin_directory(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b"cublas"})

        assert gpu_pack._extract_libraries(archive_path) == 1
        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == b"cublas"

    @pytest.mark.parametrize(
        "member_name",
        [
            "nvidia/cublas/bin/cublas64_12.dll",
            "nvidia/cublas/bin/cublas64_12.DLL",
            "nvidia/cublas/bin/cublas64_12.Dll",
            "nvidia/cublas/BIN/cublas64_12.dll",
            "nvidia/cublas/Bin/cublas64_12.dll",
            "NVIDIA/CUBLAS/BIN/CUBLAS64_12.DLL",
        ],
    )
    def test_matches_the_extension_and_the_bin_segment_case_insensitively(self, tmp_path, cuda_dir, member_name):
        archive_path = build_archive(tmp_path, {member_name: b"payload"})

        extracted_count = gpu_pack._extract_libraries(archive_path)

        assert extracted_count == 1
        assert os.listdir(cuda_dir) == [os.path.basename(member_name)]

    @pytest.mark.parametrize(
        "member_name",
        [
            "nvidia/cublas/lib/cublas64_12.dll",
            "nvidia/cublas/bin_extra/cublas64_12.dll",
            "nvidia/cublas/binaries/cublas64_12.dll",
            "nvidia/cublas/sbin/cublas64_12.dll",
            "bin/cublas64_12.dll",
            "cublas64_12.dll",
        ],
    )
    def test_skips_a_dll_that_is_not_inside_a_bin_directory(self, tmp_path, cuda_dir, member_name):
        archive_path = build_archive(tmp_path, {member_name: b"payload"})

        extracted_count = gpu_pack._extract_libraries(archive_path)

        assert extracted_count == 0
        assert os.listdir(cuda_dir) == []

    @pytest.mark.parametrize(
        "member_name",
        [
            "nvidia/cublas/bin/readme.txt",
            "nvidia/cublas/bin/cublas64_12.dll.txt",
            "nvidia/cublas/bin/cublas64_12.so",
            "nvidia/cublas/bin/cublas64_12.pdb",
            "nvidia/cublas/bin/cublas64_12.dl",
            "nvidia/cublas/bin/nvidia.dllx",
        ],
    )
    def test_skips_a_non_dll_member_inside_a_bin_directory(self, tmp_path, cuda_dir, member_name):
        archive_path = build_archive(tmp_path, {member_name: b"payload"})

        extracted_count = gpu_pack._extract_libraries(archive_path)

        assert extracted_count == 0
        assert os.listdir(cuda_dir) == []

    def test_extracts_only_the_matching_members_of_a_mixed_archive(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {
            "nvidia/cublas/bin/cublas64_12.dll": b"one",
            "nvidia/cublas/bin/cublasLt64_12.dll": b"two",
            "nvidia/cudnn/bin/cudnn64_9.dll": b"three",
            "nvidia/cublas/lib/cublas.lib": b"skip",
            "nvidia/cublas/bin/notes.txt": b"skip",
            "nvidia_cublas_cu12-12.4.5.8.dist-info/METADATA": b"skip",
        })

        extracted_count = gpu_pack._extract_libraries(archive_path)

        assert extracted_count == 3
        assert sorted(os.listdir(cuda_dir)) == [CUBLAS_LIBRARY, "cublasLt64_12.dll", CUDNN_LIBRARY]

    def test_returns_zero_for_an_archive_without_any_libraries(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/lib/cublas.lib": b"payload"})

        assert gpu_pack._extract_libraries(archive_path) == 0

    def test_returns_zero_for_an_empty_archive(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {})

        assert gpu_pack._extract_libraries(archive_path) == 0
        assert os.listdir(cuda_dir) == []


class TestExtractLibrariesFileWriting:
    def test_flattens_a_nested_member_path_onto_its_basename(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/inner/deeper/cublas64_12.dll": b"payload"})

        gpu_pack._extract_libraries(archive_path)

        assert os.listdir(cuda_dir) == [CUBLAS_LIBRARY]
        assert (cuda_dir / CUBLAS_LIBRARY).is_file()

    def test_does_not_recreate_the_directory_structure_of_the_archive(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b"payload"})

        gpu_pack._extract_libraries(archive_path)

        assert [entry.name for entry in cuda_dir.iterdir() if entry.is_dir()] == []

    def test_writes_bytes_identical_to_the_archived_payload(self, tmp_path, cuda_dir):
        payload = bytes([0, 1, 2, 255, 0, 128, 10, 13, 26])
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": payload})

        gpu_pack._extract_libraries(archive_path)

        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == payload

    def test_writes_a_payload_larger_than_one_download_chunk_completely(self, tmp_path, cuda_dir):
        archive_path = build_archive(
            tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": PAYLOAD_LARGER_THAN_ONE_CHUNK})

        gpu_pack._extract_libraries(archive_path)

        written = cuda_dir / CUBLAS_LIBRARY
        assert written.stat().st_size == len(PAYLOAD_LARGER_THAN_ONE_CHUNK)
        assert written.read_bytes() == PAYLOAD_LARGER_THAN_ONE_CHUNK

    def test_writes_a_payload_of_exactly_one_download_chunk_completely(self, tmp_path, cuda_dir):
        archive_path = build_archive(
            tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": PAYLOAD_EXACTLY_ONE_CHUNK})

        gpu_pack._extract_libraries(archive_path)

        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == PAYLOAD_EXACTLY_ONE_CHUNK

    def test_writes_an_empty_member_as_an_empty_file(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b""})

        assert gpu_pack._extract_libraries(archive_path) == 1
        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == b""

    def test_overwrites_a_library_that_is_already_present(self, tmp_path, cuda_dir):
        (cuda_dir / CUBLAS_LIBRARY).write_bytes(b"stale-and-longer")
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b"fresh"})

        gpu_pack._extract_libraries(archive_path)

        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == b"fresh"

    def test_counts_both_members_when_two_bin_paths_share_a_basename(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {
            "nvidia/first/bin/cublas64_12.dll": b"first",
            "nvidia/second/bin/cublas64_12.dll": b"second",
        })

        extracted_count = gpu_pack._extract_libraries(archive_path)

        assert extracted_count == 2
        assert os.listdir(cuda_dir) == [CUBLAS_LIBRARY]
        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == b"second"

    def test_makes_the_required_libraries_detectable_by_is_installed(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {
            "nvidia/cublas/bin/cublas64_12.dll": b"one",
            "nvidia/cudnn/bin/cudnn64_9.dll": b"two",
        })

        gpu_pack._extract_libraries(archive_path)

        assert gpu_pack.is_installed() is True


class TestExtractLibrariesLogging:
    def test_reports_the_extracted_count_to_the_log_callback(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {
            "nvidia/cublas/bin/cublas64_12.dll": b"one",
            "nvidia/cudnn/bin/cudnn64_9.dll": b"two",
        })
        log_lines = []

        gpu_pack._extract_libraries(archive_path, log=log_lines.append)

        assert log_lines == ["  Extracted 2 library file(s)"]

    def test_reports_a_zero_count_when_nothing_matched(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/lib/cublas.lib": b"payload"})
        log_lines = []

        gpu_pack._extract_libraries(archive_path, log=log_lines.append)

        assert log_lines == ["  Extracted 0 library file(s)"]

    def test_logs_once_regardless_of_how_many_members_matched(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {
            "nvidia/cublas/bin/one.dll": b"one",
            "nvidia/cublas/bin/two.dll": b"two",
            "nvidia/cublas/bin/three.dll": b"three",
        })
        log_lines = []

        gpu_pack._extract_libraries(archive_path, log=log_lines.append)

        assert log_lines == ["  Extracted 3 library file(s)"]

    def test_extracts_without_a_log_callback(self, tmp_path, cuda_dir):
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b"payload"})

        assert gpu_pack._extract_libraries(archive_path) == 1
        assert (cuda_dir / CUBLAS_LIBRARY).is_file()


class TestExtractLibrariesErrors:
    def test_raises_when_the_wheel_is_not_a_zip_archive(self, tmp_path, cuda_dir):
        not_an_archive = tmp_path / "broken.whl"
        not_an_archive.write_bytes(b"this is not a wheel")

        with pytest.raises(zipfile.BadZipFile):
            gpu_pack._extract_libraries(str(not_an_archive))

    def test_raises_when_the_wheel_file_is_missing(self, tmp_path, cuda_dir):
        with pytest.raises(FileNotFoundError):
            gpu_pack._extract_libraries(str(tmp_path / "absent.whl"))

    def test_raises_when_the_cuda_directory_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu_pack, "CUDA_DIR", str(tmp_path / "missing"))
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b"payload"})

        with pytest.raises(FileNotFoundError):
            gpu_pack._extract_libraries(archive_path)

    def test_does_not_log_when_extraction_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu_pack, "CUDA_DIR", str(tmp_path / "missing"))
        archive_path = build_archive(tmp_path, {"nvidia/cublas/bin/cublas64_12.dll": b"payload"})
        log_lines = []

        with pytest.raises(FileNotFoundError):
            gpu_pack._extract_libraries(archive_path, log=log_lines.append)

        assert log_lines == []


class TestInstallDirectorySetup:
    def test_creates_the_cuda_directory_when_it_is_missing(self, tmp_path, monkeypatch):
        cuda_directory = tmp_path / "appdata" / "cuda"
        monkeypatch.setattr(gpu_pack, "CUDA_DIR", str(cuda_directory))
        patch_install_dependencies(monkeypatch, cuda_directory)

        assert gpu_pack.install() is True
        assert cuda_directory.is_dir()

    def test_creates_the_download_directory_inside_the_cuda_directory(self, install_doubles):
        gpu_pack.install()

        assert [call["download_directory"] for call in install_doubles.downloader.calls] == [
            install_doubles.download_directory, install_doubles.download_directory]
        assert all(call["download_directory_existed"] for call in install_doubles.downloader.calls)

    def test_tolerates_a_download_directory_left_over_from_an_earlier_run(self, cuda_dir, monkeypatch):
        stale_directory = cuda_dir / DOWNLOAD_DIRECTORY_NAME
        stale_directory.mkdir()
        (stale_directory / "leftover.whl").write_bytes(b"stale")
        patch_install_dependencies(monkeypatch, cuda_dir)

        assert gpu_pack.install() is True
        assert not stale_directory.exists()

    def test_keeps_the_extracted_libraries_in_the_cuda_directory(self, install_doubles, cuda_dir):
        gpu_pack.install()

        assert sorted(os.listdir(cuda_dir)) == [CUBLAS_LIBRARY, CUDNN_LIBRARY]


class TestInstallWheelResolution:
    def test_resolves_one_wheel_for_every_required_package(self, install_doubles):
        gpu_pack.install()

        assert install_doubles.resolver.requested_packages == [
            entry["package"] for entry in gpu_pack.REQUIRED_PACKAGES]

    def test_resolves_each_package_with_its_declared_major_pin(self, install_doubles):
        gpu_pack.install()

        assert [(call["package"], call["major"]) for call in install_doubles.resolver.calls] == [
            (CUBLAS_PACKAGE, None), (CUDNN_PACKAGE, 9)]

    def test_passes_the_log_callback_to_the_resolver(self, install_doubles):
        log_lines = []

        gpu_pack.install(log=log_lines.append)

        assert [call["log"] for call in install_doubles.resolver.calls] == [
            log_lines.append, log_lines.append]

    def test_passes_no_log_callback_when_none_was_given(self, install_doubles):
        gpu_pack.install()

        assert [call["log"] for call in install_doubles.resolver.calls] == [None, None]

    def test_resolves_both_wheels_before_downloading_any_of_them(self, install_doubles):
        statuses = []

        gpu_pack.install(on_status=statuses.append)

        assert statuses.index(f"Locating {CUDNN_PACKAGE}...") < statuses.index(f"Downloading {CUBLAS_PACKAGE}...")

    def test_downloads_the_url_reported_by_the_resolver(self, install_doubles):
        gpu_pack.install()

        assert [call["url"] for call in install_doubles.downloader.calls] == [
            doubles.wheel_file(CUBLAS_WHEEL_FILENAME)["url"],
            doubles.wheel_file(CUDNN_WHEEL_FILENAME)["url"],
        ]

    def test_names_each_downloaded_file_after_the_resolved_wheel(self, install_doubles):
        gpu_pack.install()

        assert install_doubles.downloader.requested_filenames == [
            CUBLAS_WHEEL_FILENAME, CUDNN_WHEEL_FILENAME]

    def test_passes_the_declared_wheel_size_as_the_download_total(self, install_doubles):
        gpu_pack.install()

        assert [call["total_bytes"] for call in install_doubles.downloader.calls] == [
            CUBLAS_WHEEL_SIZE, CUDNN_WHEEL_SIZE]

    def test_passes_a_zero_download_total_when_the_wheel_size_is_unknown(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(default_wheels(cublas_size=None, cudnn_size=None))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)

        gpu_pack.install()

        assert [call["total_bytes"] for call in install_doubles.downloader.calls] == [0, 0]

    def test_forwards_the_cancellation_callback_to_every_download(self, install_doubles):
        def never_cancelled():
            return False

        gpu_pack.install(is_cancelled=never_cancelled)

        assert [call["is_cancelled"] for call in install_doubles.downloader.calls] == [
            never_cancelled, never_cancelled]


class TestInstallStatusReporting:
    def test_reports_the_full_status_sequence_for_a_successful_install(self, install_doubles):
        statuses = []

        gpu_pack.install(on_status=statuses.append)

        assert statuses == [
            f"Locating {CUBLAS_PACKAGE}...",
            f"Locating {CUDNN_PACKAGE}...",
            f"Downloading {CUBLAS_PACKAGE}...",
            f"Installing {CUBLAS_PACKAGE}...",
            f"Downloading {CUDNN_PACKAGE}...",
            f"Installing {CUDNN_PACKAGE}...",
            "GPU acceleration ready",
        ]

    def test_finishes_with_the_gpu_acceleration_ready_status(self, install_doubles):
        statuses = []

        gpu_pack.install(on_status=statuses.append)

        assert statuses[-1] == "GPU acceleration ready"

    def test_announces_installing_only_after_the_matching_download(self, install_doubles):
        statuses = []

        gpu_pack.install(on_status=statuses.append)

        assert statuses.index(f"Downloading {CUDNN_PACKAGE}...") < statuses.index(f"Installing {CUDNN_PACKAGE}...")

    def test_completes_when_no_status_callback_is_supplied(self, install_doubles):
        assert gpu_pack.install(on_status=None) is True

    def test_completes_when_every_callback_is_omitted(self, install_doubles):
        assert gpu_pack.install() is True


class TestInstallProgressReporting:
    def test_reports_progress_against_the_combined_size_of_both_wheels(self, install_doubles):
        progress = ProgressRecorder()

        gpu_pack.install(on_progress=progress)

        assert {total for _current, total in progress.events} == {COMBINED_WHEEL_SIZE}

    def test_reports_cumulative_bytes_across_both_wheels(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(progress_by_filename={
            CUBLAS_WHEEL_FILENAME: (100, CUBLAS_WHEEL_SIZE),
            CUDNN_WHEEL_FILENAME: (250, CUDNN_WHEEL_SIZE),
        })
        patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)
        progress = ProgressRecorder()

        gpu_pack.install(on_progress=progress)

        assert progress.events == [
            (100, COMBINED_WHEEL_SIZE),
            (CUBLAS_WHEEL_SIZE, COMBINED_WHEEL_SIZE),
            (CUBLAS_WHEEL_SIZE + 250, COMBINED_WHEEL_SIZE),
            (COMBINED_WHEEL_SIZE, COMBINED_WHEEL_SIZE),
        ]

    def test_offsets_the_second_wheel_progress_by_the_size_of_the_first(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(progress_by_filename={
            CUBLAS_WHEEL_FILENAME: (CUBLAS_WHEEL_SIZE,),
            CUDNN_WHEEL_FILENAME: (1, 2, 3),
        })
        patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)
        progress = ProgressRecorder()

        gpu_pack.install(on_progress=progress)

        assert progress.current_values == [
            CUBLAS_WHEEL_SIZE,
            CUBLAS_WHEEL_SIZE + 1,
            CUBLAS_WHEEL_SIZE + 2,
            CUBLAS_WHEEL_SIZE + 3,
        ]

    def test_never_reports_a_decreasing_byte_count(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(progress_by_filename={
            CUBLAS_WHEEL_FILENAME: (10, 200, CUBLAS_WHEEL_SIZE),
            CUDNN_WHEEL_FILENAME: (5, 300, CUDNN_WHEEL_SIZE),
        })
        patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)
        progress = ProgressRecorder()

        gpu_pack.install(on_progress=progress)

        reported = progress.current_values
        assert reported == sorted(reported)
        assert reported[-1] == COMBINED_WHEEL_SIZE

    def test_starts_the_first_wheel_progress_from_zero_offset(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(progress_by_filename={CUBLAS_WHEEL_FILENAME: (7,)})
        patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)
        progress = ProgressRecorder()

        gpu_pack.install(on_progress=progress)

        assert progress.events[0] == (7, COMBINED_WHEEL_SIZE)

    def test_reports_no_progress_when_both_wheel_sizes_are_unknown(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(default_wheels(cublas_size=None, cudnn_size=None))
        patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)
        progress = ProgressRecorder()

        assert gpu_pack.install(on_progress=progress) is True
        assert progress.events == []

    def test_reports_no_progress_when_both_wheel_sizes_are_zero(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(default_wheels(cublas_size=0, cudnn_size=0))
        patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)
        progress = ProgressRecorder()

        assert gpu_pack.install(on_progress=progress) is True
        assert progress.events == []

    def test_measures_progress_only_against_the_wheels_that_declare_a_size(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(default_wheels(cudnn_size=None))
        patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)
        progress = ProgressRecorder()

        gpu_pack.install(on_progress=progress)

        assert progress.events == [(CUBLAS_WHEEL_SIZE, CUBLAS_WHEEL_SIZE), (CUBLAS_WHEEL_SIZE, CUBLAS_WHEEL_SIZE)]

    def test_completes_when_no_progress_callback_is_supplied(self, install_doubles):
        assert gpu_pack.install(on_progress=None) is True


class TestInstallExtractionAndCleanup:
    def test_extracts_every_downloaded_wheel_exactly_once(self, install_doubles):
        gpu_pack.install()

        assert install_doubles.extractor.extracted_wheel_names == [
            CUBLAS_WHEEL_FILENAME, CUDNN_WHEEL_FILENAME]

    def test_extracts_from_the_file_the_downloader_wrote(self, install_doubles):
        gpu_pack.install()

        assert [call["wheel_path"] for call in install_doubles.extractor.calls] == [
            call["destination_path"] for call in install_doubles.downloader.calls]
        assert all(call["wheel_existed"] for call in install_doubles.extractor.calls)

    def test_passes_the_log_callback_to_every_extraction(self, install_doubles):
        log_lines = []

        gpu_pack.install(log=log_lines.append)

        assert [call["log"] for call in install_doubles.extractor.calls] == [
            log_lines.append, log_lines.append]

    def test_removes_each_wheel_before_the_next_one_is_extracted(self, install_doubles):
        gpu_pack.install()

        assert [call["download_directory_contents"] for call in install_doubles.extractor.calls] == [
            [CUBLAS_WHEEL_FILENAME], [CUDNN_WHEEL_FILENAME]]

    def test_removes_the_download_directory_after_a_successful_install(self, install_doubles):
        gpu_pack.install()

        assert not os.path.exists(install_doubles.download_directory)

    def test_ignores_the_number_of_files_reported_by_the_extractor(self, cuda_dir, monkeypatch):
        extractor = RecordingExtractor(cuda_dir, return_value=0)
        patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)

        assert gpu_pack.install() is True

    def test_returns_true_when_the_required_libraries_are_present(self, install_doubles):
        assert gpu_pack.install() is True


class TestInstallFailurePaths:
    def test_raises_when_no_required_library_was_extracted(self, cuda_dir, monkeypatch):
        extractor = RecordingExtractor(cuda_dir, libraries_per_call=[[], []])
        patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)

        with pytest.raises(RuntimeError, match=INCOMPLETE_MESSAGE):
            gpu_pack.install()

    @pytest.mark.parametrize(
        ("libraries_per_call", "present_library"),
        [
            ([[CUBLAS_LIBRARY], []], CUBLAS_LIBRARY),
            ([[], [CUDNN_LIBRARY]], CUDNN_LIBRARY),
        ],
    )
    def test_raises_when_only_one_required_library_was_extracted(
            self, cuda_dir, monkeypatch, libraries_per_call, present_library):
        extractor = RecordingExtractor(cuda_dir, libraries_per_call=libraries_per_call)
        patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)

        with pytest.raises(RuntimeError, match=INCOMPLETE_MESSAGE):
            gpu_pack.install()

        assert os.listdir(cuda_dir) == [present_library]

    def test_does_not_report_ready_when_the_libraries_are_incomplete(self, cuda_dir, monkeypatch):
        extractor = RecordingExtractor(cuda_dir, libraries_per_call=[[], []])
        patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)
        statuses = []

        with pytest.raises(RuntimeError):
            gpu_pack.install(on_status=statuses.append)

        assert "GPU acceleration ready" not in statuses

    def test_removes_the_download_directory_when_the_libraries_are_incomplete(self, cuda_dir, monkeypatch):
        extractor = RecordingExtractor(cuda_dir, libraries_per_call=[[], []])
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)

        with pytest.raises(RuntimeError):
            gpu_pack.install()

        assert not os.path.exists(install_doubles.download_directory)

    def test_propagates_a_wheel_resolution_failure(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(error=RuntimeError("No win_amd64 wheel found for nvidia-cublas-cu12"))
        patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)

        with pytest.raises(RuntimeError, match="No win_amd64 wheel found"):
            gpu_pack.install()

    def test_stops_at_the_first_package_when_resolution_fails(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(error=RuntimeError("no wheel"))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)
        statuses = []

        with pytest.raises(RuntimeError):
            gpu_pack.install(on_status=statuses.append)

        assert statuses == [f"Locating {CUBLAS_PACKAGE}..."]
        assert install_doubles.downloader.calls == []
        assert install_doubles.extractor.calls == []

    def test_removes_the_download_directory_when_resolution_fails(self, cuda_dir, monkeypatch):
        resolver = RecordingWheelResolver(error=RuntimeError("no wheel"))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, resolver=resolver)

        with pytest.raises(RuntimeError):
            gpu_pack.install()

        assert not os.path.exists(install_doubles.download_directory)

    def test_propagates_a_download_failure(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(error=OSError("connection reset"))
        patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)

        with pytest.raises(OSError, match="connection reset"):
            gpu_pack.install()

    def test_removes_the_partial_download_and_its_directory_when_the_download_fails(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(error=OSError("connection reset"))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)

        with pytest.raises(OSError):
            gpu_pack.install()

        assert not os.path.exists(install_doubles.download_directory)
        assert os.listdir(cuda_dir) == []

    def test_does_not_extract_or_report_ready_when_the_download_fails(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(error=OSError("connection reset"))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)
        statuses = []

        with pytest.raises(OSError):
            gpu_pack.install(on_status=statuses.append)

        assert install_doubles.extractor.calls == []
        assert statuses[-1] == f"Downloading {CUBLAS_PACKAGE}..."

    def test_propagates_a_cancelled_download(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(error=RuntimeError("Cancelled"))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)

        def always_cancelled():
            return True

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack.install(is_cancelled=always_cancelled)

        assert not os.path.exists(install_doubles.download_directory)

    def test_propagates_an_extraction_failure(self, cuda_dir, monkeypatch):
        extractor = RecordingExtractor(cuda_dir, error=zipfile.BadZipFile("not a zip"))
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)

        with pytest.raises(zipfile.BadZipFile):
            gpu_pack.install()

        assert not os.path.exists(install_doubles.download_directory)

    def test_propagates_a_failure_to_remove_the_downloaded_wheel(self, install_doubles, monkeypatch):
        def refuse_to_remove_file(path):
            raise PermissionError("the file is in use")

        monkeypatch.setattr(os, "remove", refuse_to_remove_file)

        with pytest.raises(PermissionError):
            gpu_pack.install()

        assert os.listdir(install_doubles.download_directory) == [CUBLAS_WHEEL_FILENAME]


class TestInstallCleanupFailures:
    def test_swallows_a_failure_to_remove_the_download_directory(self, install_doubles, monkeypatch):
        def refuse_to_remove_directory(path):
            raise OSError("the directory is locked")

        monkeypatch.setattr(os, "rmdir", refuse_to_remove_directory)

        assert gpu_pack.install() is True
        assert os.listdir(install_doubles.download_directory) == []

    def test_swallows_a_failure_caused_by_a_leftover_subdirectory(self, cuda_dir, monkeypatch):
        extractor = RecordingExtractor(cuda_dir, extra_directory_name="stray")
        install_doubles = patch_install_dependencies(monkeypatch, cuda_dir, extractor=extractor)

        assert gpu_pack.install() is True
        assert (Path(install_doubles.download_directory) / "stray").is_dir()

    def test_still_reports_ready_when_cleanup_fails(self, install_doubles, monkeypatch):
        def refuse_to_remove_directory(path):
            raise OSError("the directory is locked")

        monkeypatch.setattr(os, "rmdir", refuse_to_remove_directory)
        statuses = []

        gpu_pack.install(on_status=statuses.append)

        assert statuses[-1] == "GPU acceleration ready"

    def test_a_cleanup_failure_does_not_mask_a_download_failure(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(error=OSError("connection reset"))
        patch_install_dependencies(monkeypatch, cuda_dir, downloader=downloader)

        def refuse_to_remove_directory(path):
            raise OSError("the directory is locked")

        monkeypatch.setattr(os, "rmdir", refuse_to_remove_directory)

        with pytest.raises(OSError, match="connection reset"):
            gpu_pack.install()


class TestInstallWithRealExtraction:
    def test_installs_both_libraries_from_real_wheel_archives(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(archive_members_by_filename={
            CUBLAS_WHEEL_FILENAME: {
                "nvidia/cublas/bin/cublas64_12.dll": b"cublas-bytes",
                "nvidia/cublas/lib/cublas.lib": b"ignored",
            },
            CUDNN_WHEEL_FILENAME: {
                "nvidia/cudnn/bin/cudnn64_9.dll": b"cudnn-bytes",
                "nvidia/cudnn/bin/notes.txt": b"ignored",
            },
        })
        patch_wheel_source(monkeypatch, downloader=downloader)
        log_lines = []

        assert gpu_pack.install(log=log_lines.append) is True
        assert (cuda_dir / CUBLAS_LIBRARY).read_bytes() == b"cublas-bytes"
        assert (cuda_dir / CUDNN_LIBRARY).read_bytes() == b"cudnn-bytes"
        assert log_lines[-2:] == ["  Extracted 1 library file(s)", "  Extracted 1 library file(s)"]

    def test_leaves_only_the_libraries_behind_after_a_real_extraction(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(archive_members_by_filename={
            CUBLAS_WHEEL_FILENAME: {"nvidia/cublas/bin/cublas64_12.dll": b"cublas-bytes"},
            CUDNN_WHEEL_FILENAME: {"nvidia/cudnn/bin/cudnn64_9.dll": b"cudnn-bytes"},
        })
        patch_wheel_source(monkeypatch, downloader=downloader)

        gpu_pack.install()

        assert sorted(os.listdir(cuda_dir)) == [CUBLAS_LIBRARY, CUDNN_LIBRARY]

    def test_raises_when_the_real_wheels_carry_no_libraries(self, cuda_dir, monkeypatch):
        downloader = RecordingDownloader(archive_members_by_filename={
            CUBLAS_WHEEL_FILENAME: {"nvidia/cublas/lib/cublas.lib": b"ignored"},
            CUDNN_WHEEL_FILENAME: {"nvidia/cudnn/lib/cudnn.lib": b"ignored"},
        })
        patch_wheel_source(monkeypatch, downloader=downloader)

        with pytest.raises(RuntimeError, match=INCOMPLETE_MESSAGE):
            gpu_pack.install()

        assert os.listdir(cuda_dir) == []
