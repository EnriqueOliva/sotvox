"""Unit tests for gpu_pack wheel resolution and streamed downloading."""

import json

import pytest
from doubles import FakeHTTPResponse, json_response, pypi_metadata, wheel_file

import gpu_pack

PACKAGE_NAME = "nvidia-cublas-cu12"
CUDNN_PACKAGE_NAME = "nvidia-cudnn-cu12"
BYTES_PER_MEGABYTE = 1024 * 1024


def windows_wheel(version_text, package_name=PACKAGE_NAME, **overrides):
    normalised_name = package_name.replace("-", "_")
    return wheel_file(f"{normalised_name}-{version_text}-py3-none-win_amd64.whl", **overrides)


def wheel_without_size_key(filename):
    entry = wheel_file(filename)
    del entry["size"]
    return entry


def payload_of_length(length):
    pattern = b"sotvox-gpu-pack-payload-"
    repetitions = (length // len(pattern)) + 1
    return (pattern * repetitions)[:length]


class RecordingResponse(FakeHTTPResponse):
    def __init__(self, payload=b"", headers=None, read_error_after=None):
        super().__init__(payload, headers, read_error_after)
        self.read_sizes = []
        self.exit_count = 0

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)

    def __exit__(self, exception_type, exception, traceback):
        self.exit_count += 1
        return super().__exit__(exception_type, exception, traceback)


class UrlopenRecorder:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, url, timeout=None):
        self.calls.append({"url": url, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def last_call(self):
        return self.calls[-1]


class CancellationProbe:
    def __init__(self, cancel_after_calls=None, return_value=False):
        self.cancel_after_calls = cancel_after_calls
        self.return_value = return_value
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        if self.cancel_after_calls is None:
            return self.return_value
        return self.call_count > self.cancel_after_calls


class ProgressRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, downloaded_bytes, total_bytes):
        self.calls.append((downloaded_bytes, total_bytes))

    @property
    def downloaded_counts(self):
        return [downloaded for downloaded, _total in self.calls]

    @property
    def totals(self):
        return [total for _downloaded, total in self.calls]


@pytest.fixture
def patched_urlopen(monkeypatch):
    def install(response=None, error=None):
        recorder = UrlopenRecorder(response, error)
        monkeypatch.setattr(gpu_pack.urllib.request, "urlopen", recorder)
        return recorder

    return install


class TestResolveWheelRequest:
    def test_requests_the_pypi_json_url_for_the_requested_package(self, patched_urlopen):
        recorder = patched_urlopen(json_response(pypi_metadata({"1.0.0": [windows_wheel("1.0.0")]})))

        gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert recorder.last_call["url"] == "https://pypi.org/pypi/nvidia-cublas-cu12/json"

    def test_requests_the_url_of_the_package_it_was_asked_about(self, patched_urlopen):
        recorder = patched_urlopen(json_response(pypi_metadata(
            {"9.1.0.70": [windows_wheel("9.1.0.70", CUDNN_PACKAGE_NAME)]})))

        gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, 9)

        assert recorder.last_call["url"] == "https://pypi.org/pypi/nvidia-cudnn-cu12/json"

    def test_passes_a_sixty_second_timeout(self, patched_urlopen):
        recorder = patched_urlopen(json_response(pypi_metadata({"1.0.0": [windows_wheel("1.0.0")]})))

        gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert recorder.last_call["timeout"] == 60
        assert recorder.last_call["timeout"] == gpu_pack.REQUEST_TIMEOUT_SECONDS

    def test_closes_the_response_after_reading_the_metadata(self, patched_urlopen):
        response = RecordingResponse(json.dumps(pypi_metadata({"1.0.0": [windows_wheel("1.0.0")]})).encode("utf-8"))
        patched_urlopen(response)

        gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert response.exit_count == 1

    def test_closes_the_response_even_when_no_wheel_matches(self, patched_urlopen):
        response = RecordingResponse(json.dumps(pypi_metadata({})).encode("utf-8"))
        patched_urlopen(response)

        with pytest.raises(RuntimeError):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert response.exit_count == 1

    def test_decodes_the_metadata_body_as_utf8(self, patched_urlopen):
        accented_filename = "nvidia_cublás_cu12-1.0.0-py3-none-win_amd64.whl"
        payload = json.dumps(pypi_metadata({"1.0.0": [wheel_file(accented_filename)]})).encode("utf-8")
        patched_urlopen(FakeHTTPResponse(payload))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == accented_filename

    def test_propagates_transport_errors_from_urlopen(self, patched_urlopen):
        patched_urlopen(error=OSError("name resolution failed"))

        with pytest.raises(OSError, match="name resolution failed"):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

    def test_propagates_malformed_json_from_pypi(self, patched_urlopen):
        patched_urlopen(FakeHTTPResponse(b"<html>service unavailable</html>"))

        with pytest.raises(json.JSONDecodeError):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)


class TestResolveWheelVersionSelection:
    def test_returns_the_file_info_dictionary_of_the_matching_wheel(self, patched_urlopen):
        only_wheel = windows_wheel("12.4.5.8")
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [only_wheel]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved == only_wheel
        assert resolved["url"] == only_wheel["url"]
        assert resolved["size"] == only_wheel["size"]

    def test_picks_the_highest_version_that_has_a_windows_wheel(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.1.3.1": [windows_wheel("12.1.3.1")],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
            "12.3.4.1": [windows_wheel("12.3.4.1")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_picks_the_highest_version_even_when_it_is_listed_first(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.4.5.8": [windows_wheel("12.4.5.8")],
            "12.1.3.1": [windows_wheel("12.1.3.1")],
            "11.0.0": [windows_wheel("11.0.0")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_compares_version_numbers_numerically_rather_than_as_text(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "9.9.0": [windows_wheel("9.9.0")],
            "10.0.0": [windows_wheel("10.0.0")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-10.0.0-py3-none-win_amd64.whl"

    def test_a_longer_version_tuple_outranks_its_shorter_prefix(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.4": [windows_wheel("12.4")],
            "12.4.5": [windows_wheel("12.4.5")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5-py3-none-win_amd64.whl"

    def test_keeps_the_first_windows_wheel_of_the_winning_release(self, patched_urlopen):
        first_wheel = wheel_file("nvidia_cublas_cu12-12.4.5.8-cp311-cp311-win_amd64.whl")
        second_wheel = wheel_file("nvidia_cublas_cu12-12.4.5.8-cp312-cp312-win_amd64.whl")
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [first_wheel, second_wheel]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved == first_wheel

    def test_a_release_with_no_files_at_all_is_skipped(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    @pytest.mark.parametrize(
        "unparseable_version",
        ["", "nightly", "v1.0.0", "1!2.0", "latest", "dev"],
    )
    def test_skips_releases_whose_version_text_parses_to_nothing(self, patched_urlopen, unparseable_version):
        patched_urlopen(json_response(pypi_metadata({
            unparseable_version: [windows_wheel("999")],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_raises_when_every_release_version_is_unparseable(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "nightly": [windows_wheel("nightly")],
            "v2.0.0": [windows_wheel("v2.0.0")],
        })))

        with pytest.raises(RuntimeError):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

    def test_a_trailing_suffix_after_the_last_number_is_ignored_when_ranking(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.4.5.8.post1": [windows_wheel("12.4.5.8.post1")],
            "12.4.5.9": [windows_wheel("12.4.5.9")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.9-py3-none-win_amd64.whl"


class TestResolveWheelMajorPin:
    def test_a_major_nine_pin_ignores_a_newer_major_ten_release(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "9.1.0.70": [windows_wheel("9.1.0.70", CUDNN_PACKAGE_NAME)],
            "10.0.0.1": [windows_wheel("10.0.0.1", CUDNN_PACKAGE_NAME)],
        })))

        resolved = gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, 9)

        assert resolved["filename"] == "nvidia_cudnn_cu12-9.1.0.70-py3-none-win_amd64.whl"

    def test_a_major_pin_still_picks_the_highest_release_inside_that_major(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "9.1.0.70": [windows_wheel("9.1.0.70", CUDNN_PACKAGE_NAME)],
            "9.5.1.17": [windows_wheel("9.5.1.17", CUDNN_PACKAGE_NAME)],
            "10.9.9.9": [windows_wheel("10.9.9.9", CUDNN_PACKAGE_NAME)],
        })))

        resolved = gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, 9)

        assert resolved["filename"] == "nvidia_cudnn_cu12-9.5.1.17-py3-none-win_amd64.whl"

    def test_a_major_pin_also_ignores_older_majors(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "8.9.7.29": [windows_wheel("8.9.7.29", CUDNN_PACKAGE_NAME)],
            "9.0.0.1": [windows_wheel("9.0.0.1", CUDNN_PACKAGE_NAME)],
        })))

        resolved = gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, 9)

        assert resolved["filename"] == "nvidia_cudnn_cu12-9.0.0.1-py3-none-win_amd64.whl"

    def test_no_pin_lets_the_newest_major_win(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "9.1.0.70": [windows_wheel("9.1.0.70", CUDNN_PACKAGE_NAME)],
            "10.0.0.1": [windows_wheel("10.0.0.1", CUDNN_PACKAGE_NAME)],
        })))

        resolved = gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cudnn_cu12-10.0.0.1-py3-none-win_amd64.whl"

    def test_raises_when_the_pinned_major_has_no_release_at_all(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "10.0.0.1": [windows_wheel("10.0.0.1", CUDNN_PACKAGE_NAME)],
            "11.0.0.1": [windows_wheel("11.0.0.1", CUDNN_PACKAGE_NAME)],
        })))

        with pytest.raises(RuntimeError, match="nvidia-cudnn-cu12"):
            gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, 9)

    def test_a_zero_major_pin_is_honoured_rather_than_treated_as_absent(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "0.9.1": [windows_wheel("0.9.1")],
            "3.0.0": [windows_wheel("3.0.0")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, 0)

        assert resolved["filename"] == "nvidia_cublas_cu12-0.9.1-py3-none-win_amd64.whl"

    def test_a_bare_major_only_version_satisfies_the_pin(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({"9": [windows_wheel("9", CUDNN_PACKAGE_NAME)]})))

        resolved = gpu_pack._resolve_wheel(CUDNN_PACKAGE_NAME, 9)

        assert resolved["filename"] == "nvidia_cudnn_cu12-9-py3-none-win_amd64.whl"


class TestResolveWheelFileFiltering:
    @pytest.mark.parametrize(
        "rejected_filename",
        [
            "nvidia_cublas_cu12-12.9.0.tar.gz",
            "nvidia_cublas_cu12-12.9.0-py3-none-win_amd64.tar.gz",
            "nvidia_cublas_cu12-12.9.0-py3-none-manylinux2014_x86_64.whl",
            "nvidia_cublas_cu12-12.9.0-py3-none-manylinux1_x86_64.whl",
            "nvidia_cublas_cu12-12.9.0-py3-none-win32.whl",
            "nvidia_cublas_cu12-12.9.0-py3-none-any.whl",
            "nvidia_cublas_cu12-12.9.0-py3-none-macosx_11_0_arm64.whl",
            "nvidia_cublas_cu12-12.9.0-py3-none-win_arm64.whl",
            "nvidia_cublas_cu12-12.9.0-py3-none-win_amd64.whl.asc",
            "nvidia_cublas_cu12-12.9.0.zip",
        ],
    )
    def test_a_non_windows_wheel_file_cannot_win_a_release(self, patched_urlopen, rejected_filename):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [wheel_file(rejected_filename)],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_the_windows_wheel_is_chosen_from_among_its_siblings(self, patched_urlopen):
        source_distribution = wheel_file("nvidia_cublas_cu12-12.9.0.tar.gz")
        linux_wheel = wheel_file("nvidia_cublas_cu12-12.9.0-py3-none-manylinux2014_x86_64.whl")
        expected_wheel = windows_wheel("12.9.0")
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [source_distribution, linux_wheel, expected_wheel],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved == expected_wheel

    def test_a_file_entry_without_a_filename_key_is_skipped(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [{"url": "https://files.pythonhosted.org/anonymous", "size": 10}],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_a_yanked_wheel_cannot_win_a_release(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [windows_wheel("12.9.0", yanked=True)],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_a_yanked_wheel_is_skipped_in_favour_of_a_sibling_in_the_same_release(self, patched_urlopen):
        yanked_wheel = wheel_file("nvidia_cublas_cu12-12.9.0-cp311-cp311-win_amd64.whl", yanked=True)
        healthy_wheel = wheel_file("nvidia_cublas_cu12-12.9.0-cp312-cp312-win_amd64.whl")
        patched_urlopen(json_response(pypi_metadata({"12.9.0": [yanked_wheel, healthy_wheel]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved == healthy_wheel

    def test_a_yanked_reason_string_also_counts_as_yanked(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [windows_wheel("12.9.0", yanked="built against the wrong driver")],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_an_absent_yanked_key_is_treated_as_not_yanked(self, patched_urlopen):
        entry = {
            "filename": "nvidia_cublas_cu12-12.9.0-py3-none-win_amd64.whl",
            "url": "https://files.pythonhosted.org/nvidia_cublas_cu12-12.9.0-py3-none-win_amd64.whl",
            "size": 512,
        }
        patched_urlopen(json_response(pypi_metadata({"12.9.0": [entry]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved == entry


class TestResolveWheelFailure:
    def test_raises_runtime_error_naming_the_package_and_the_platform_tag(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.4.5.8": [wheel_file("nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl")],
        })))

        with pytest.raises(RuntimeError) as error:
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert PACKAGE_NAME in str(error.value)
        assert gpu_pack.WINDOWS_WHEEL_TAG in str(error.value)

    def test_raises_when_the_releases_key_is_absent_from_the_metadata(self, patched_urlopen):
        patched_urlopen(json_response({"info": {"name": PACKAGE_NAME}}))

        with pytest.raises(RuntimeError, match="nvidia-cublas-cu12"):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

    def test_raises_when_the_releases_mapping_is_empty(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({})))

        with pytest.raises(RuntimeError, match="nvidia-cublas-cu12"):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

    def test_raises_when_every_windows_wheel_is_yanked(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [windows_wheel("12.9.0", yanked=True)],
            "12.4.5.8": [windows_wheel("12.4.5.8", yanked=True)],
        })))

        with pytest.raises(RuntimeError, match="nvidia-cublas-cu12"):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None)

    def test_nothing_is_logged_when_resolution_fails(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({})))
        log_lines = []

        with pytest.raises(RuntimeError):
            gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=log_lines.append)

        assert log_lines == []


class TestResolveWheelLogging:
    def test_the_log_line_carries_the_package_the_filename_and_the_size(self, patched_urlopen):
        chosen_wheel = windows_wheel("12.4.5.8", size=3 * BYTES_PER_MEGABYTE)
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [chosen_wheel]})))
        log_lines = []

        gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=log_lines.append)

        assert len(log_lines) == 1
        assert PACKAGE_NAME in log_lines[0]
        assert chosen_wheel["filename"] in log_lines[0]
        assert "3 MB" in log_lines[0]

    def test_the_log_line_describes_the_winning_wheel_not_a_rejected_one(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.9.0": [windows_wheel("12.9.0", yanked=True)],
            "12.4.5.8": [windows_wheel("12.4.5.8")],
        })))
        log_lines = []

        gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=log_lines.append)

        assert "12.4.5.8" in log_lines[0]
        assert "12.9.0" not in log_lines[0]

    @pytest.mark.parametrize(
        ("size_bytes", "expected_fragment"),
        [
            (0, "(0 MB)"),
            (1, "(0 MB)"),
            (BYTES_PER_MEGABYTE - 1, "(1 MB)"),
            (BYTES_PER_MEGABYTE, "(1 MB)"),
            (BYTES_PER_MEGABYTE * 3 // 2, "(2 MB)"),
            (BYTES_PER_MEGABYTE * 5 // 2, "(2 MB)"),
            (BYTES_PER_MEGABYTE * 7 // 2, "(4 MB)"),
            (BYTES_PER_MEGABYTE * 600, "(600 MB)"),
        ],
    )
    def test_the_size_is_rendered_as_whole_megabytes(self, patched_urlopen, size_bytes, expected_fragment):
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [windows_wheel("12.4.5.8", size=size_bytes)]})))
        log_lines = []

        gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=log_lines.append)

        assert expected_fragment in log_lines[0]

    def test_a_missing_size_key_is_logged_as_zero_megabytes(self, patched_urlopen):
        entry = wheel_without_size_key("nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl")
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [entry]})))
        log_lines = []

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=log_lines.append)

        assert "(0 MB)" in log_lines[0]
        assert resolved == entry

    def test_a_null_size_is_logged_as_zero_megabytes(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({
            "12.4.5.8": [windows_wheel("12.4.5.8", size=None)],
        })))
        log_lines = []

        gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=log_lines.append)

        assert "(0 MB)" in log_lines[0]

    def test_resolution_succeeds_without_a_log_callback(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [windows_wheel("12.4.5.8")]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"

    def test_a_falsy_log_callback_is_not_invoked(self, patched_urlopen):
        patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [windows_wheel("12.4.5.8")]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None, log=None)

        assert resolved["filename"] == "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"


class TestDownloadWritesBytes:
    def test_writes_the_streamed_bytes_to_the_destination_exactly(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2 + 7)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "nvidia_cublas.whl"

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert destination_path.read_bytes() == payload

    def test_returns_the_number_of_bytes_written(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 123)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "nvidia_cublas.whl"

        written_bytes = gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert written_bytes == len(payload)
        assert written_bytes == destination_path.stat().st_size

    def test_writes_binary_bytes_without_newline_translation(self, patched_urlopen, tmp_path):
        payload = b"header\r\nbody\r\n\x00\x1a\xff\xfe\ntail"
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "binary.whl"

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert destination_path.read_bytes() == payload

    def test_writes_an_empty_file_when_the_body_is_empty(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(b""))
        destination_path = tmp_path / "empty.whl"

        written_bytes = gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert written_bytes == 0
        assert destination_path.read_bytes() == b""

    def test_overwrites_an_existing_destination_file(self, patched_urlopen, tmp_path):
        destination_path = tmp_path / "stale.whl"
        destination_path.write_bytes(b"leftover bytes from a previous attempt")
        payload = payload_of_length(64)
        patched_urlopen(FakeHTTPResponse(payload))

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert destination_path.read_bytes() == payload

    def test_accepts_a_path_object_as_the_destination(self, patched_urlopen, tmp_path):
        payload = payload_of_length(32)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "path_object.whl"

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", destination_path)

        assert destination_path.read_bytes() == payload


class TestDownloadRequest:
    def test_requests_the_given_url(self, patched_urlopen, tmp_path):
        recorder = patched_urlopen(FakeHTTPResponse(b"body"))

        gpu_pack._download("https://files.pythonhosted.org/a/b/wheel.whl", str(tmp_path / "wheel.whl"))

        assert recorder.last_call["url"] == "https://files.pythonhosted.org/a/b/wheel.whl"

    def test_passes_a_sixty_second_timeout(self, patched_urlopen, tmp_path):
        recorder = patched_urlopen(FakeHTTPResponse(b"body"))

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert recorder.last_call["timeout"] == gpu_pack.REQUEST_TIMEOUT_SECONDS

    def test_closes_the_response_when_the_download_completes(self, patched_urlopen, tmp_path):
        response = RecordingResponse(payload_of_length(100))
        patched_urlopen(response)

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert response.exit_count == 1

    def test_closes_the_response_when_the_stream_fails_midway(self, patched_urlopen, tmp_path):
        response = RecordingResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 3), read_error_after=1)
        patched_urlopen(response)

        with pytest.raises(OSError, match="connection reset"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert response.exit_count == 1

    def test_a_read_error_leaves_the_bytes_already_written_on_disk(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 3)
        patched_urlopen(FakeHTTPResponse(payload, read_error_after=1))
        destination_path = tmp_path / "partial.whl"

        with pytest.raises(OSError, match="connection reset"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert destination_path.read_bytes() == payload[:gpu_pack.DOWNLOAD_CHUNK_BYTES]

    def test_a_missing_destination_directory_raises_after_the_request_is_made(self, patched_urlopen, tmp_path):
        recorder = patched_urlopen(FakeHTTPResponse(b"body"))
        destination_path = tmp_path / "no-such-directory" / "wheel.whl"

        with pytest.raises(FileNotFoundError):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert len(recorder.calls) == 1


class TestDownloadChunking:
    def test_reads_in_download_chunk_bytes_sized_chunks(self, patched_urlopen, tmp_path):
        response = RecordingResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2 + 7))
        patched_urlopen(response)

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert response.read_sizes == [gpu_pack.DOWNLOAD_CHUNK_BYTES] * 4

    def test_a_body_shorter_than_one_chunk_takes_two_reads(self, patched_urlopen, tmp_path):
        response = RecordingResponse(payload_of_length(10))
        patched_urlopen(response)

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert response.read_sizes == [gpu_pack.DOWNLOAD_CHUNK_BYTES] * 2

    def test_a_body_of_exactly_one_chunk_still_needs_the_terminating_read(self, patched_urlopen, tmp_path):
        response = RecordingResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES))
        patched_urlopen(response)

        written_bytes = gpu_pack._download(
            "https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert response.read_sizes == [gpu_pack.DOWNLOAD_CHUNK_BYTES] * 2
        assert written_bytes == gpu_pack.DOWNLOAD_CHUNK_BYTES

    def test_an_empty_body_is_detected_on_the_first_read(self, patched_urlopen, tmp_path):
        response = RecordingResponse(b"")
        patched_urlopen(response)

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"))

        assert response.read_sizes == [gpu_pack.DOWNLOAD_CHUNK_BYTES]


class TestDownloadTotalBytes:
    def test_uses_content_length_when_total_bytes_is_zero(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 10)
        patched_urlopen(FakeHTTPResponse(payload, headers={"Content-Length": str(len(payload))}))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress, total_bytes=0)

        assert set(progress.totals) == {len(payload)}

    def test_prefers_the_caller_total_bytes_over_content_length(self, patched_urlopen, tmp_path):
        payload = payload_of_length(50)
        patched_urlopen(FakeHTTPResponse(payload, headers={"Content-Length": "50"}))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress, total_bytes=999)

        assert progress.totals == [999]

    def test_a_caller_total_that_disagrees_with_the_body_is_used_verbatim(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 1)
        patched_urlopen(FakeHTTPResponse(payload))
        progress = ProgressRecorder()

        written_bytes = gpu_pack._download(
            "https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
            on_progress=progress, total_bytes=7)

        assert progress.totals == [7, 7]
        assert written_bytes == len(payload)

    @pytest.mark.parametrize("headers", [{}, {"Content-Length": ""}, {"Content-Length": None}])
    def test_a_missing_or_blank_content_length_degrades_to_zero(self, patched_urlopen, tmp_path, headers):
        payload = payload_of_length(40)
        patched_urlopen(FakeHTTPResponse(payload, headers=headers))
        progress = ProgressRecorder()

        written_bytes = gpu_pack._download("https://files.pythonhosted.org/wheel.whl",
                                           str(tmp_path / "wheel.whl"), on_progress=progress)

        assert progress.totals == [0]
        assert written_bytes == len(payload)

    def test_a_content_length_of_zero_reports_a_zero_total(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(40), headers={"Content-Length": "0"}))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress)

        assert progress.totals == [0]

    def test_content_length_is_read_from_an_unrelated_header_set_without_failing(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(8), headers={"Content-Type": "application/zip"}))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress)

        assert progress.totals == [0]


class TestDownloadProgress:
    def test_reports_a_growing_downloaded_count_ending_at_the_body_length(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2 + 5)
        patched_urlopen(FakeHTTPResponse(payload, headers={"Content-Length": str(len(payload))}))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress)

        assert progress.downloaded_counts == [
            gpu_pack.DOWNLOAD_CHUNK_BYTES,
            gpu_pack.DOWNLOAD_CHUNK_BYTES * 2,
            len(payload),
        ]

    def test_reports_progress_once_per_written_chunk(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 3)
        patched_urlopen(FakeHTTPResponse(payload))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress)

        assert len(progress.calls) == 3

    def test_the_reported_total_never_changes_during_the_download(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2 + 1)
        patched_urlopen(FakeHTTPResponse(payload, headers={"Content-Length": str(len(payload))}))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress)

        assert len(set(progress.totals)) == 1

    def test_no_progress_is_reported_for_an_empty_body(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(b""))
        progress = ProgressRecorder()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           on_progress=progress)

        assert progress.calls == []

    def test_downloads_without_a_progress_callback(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 3)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "wheel.whl"

        written_bytes = gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path))

        assert written_bytes == len(payload)
        assert destination_path.read_bytes() == payload


class TestDownloadCancellation:
    def test_raises_cancelled_when_the_callback_reports_a_cancellation(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2)))

        with pytest.raises(RuntimeError) as error:
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                               is_cancelled=CancellationProbe(return_value=True))

        assert str(error.value) == "Cancelled"

    def test_cancellation_is_checked_before_the_first_read(self, patched_urlopen, tmp_path):
        response = RecordingResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES), read_error_after=0)
        patched_urlopen(response)
        destination_path = tmp_path / "wheel.whl"

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path),
                               is_cancelled=CancellationProbe(return_value=True))

        assert response.read_sizes == []

    def test_a_pre_cancelled_download_writes_no_bytes(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2)))
        destination_path = tmp_path / "wheel.whl"

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path),
                               is_cancelled=CancellationProbe(return_value=True))

        assert destination_path.read_bytes() == b""

    def test_a_pre_cancelled_download_still_leaves_an_empty_destination_file_behind(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(10)))
        destination_path = tmp_path / "wheel.whl"

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path),
                               is_cancelled=CancellationProbe(return_value=True))

        assert destination_path.exists()

    def test_a_pre_cancelled_download_reports_no_progress(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2)))
        progress = ProgressRecorder()

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                               on_progress=progress, is_cancelled=CancellationProbe(return_value=True))

        assert progress.calls == []

    def test_a_midway_cancellation_keeps_the_chunks_already_written(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 3)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "wheel.whl"

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path),
                               is_cancelled=CancellationProbe(cancel_after_calls=2))

        assert destination_path.read_bytes() == payload[:gpu_pack.DOWNLOAD_CHUNK_BYTES * 2]

    def test_a_midway_cancellation_discards_the_byte_count(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 3)))

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                               is_cancelled=CancellationProbe(cancel_after_calls=1))

    def test_cancellation_is_checked_once_per_chunk_plus_once_for_the_terminating_read(
            self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES * 2 + 1)))
        probe = CancellationProbe()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           is_cancelled=probe)

        assert probe.call_count == 4

    def test_cancellation_is_checked_once_for_an_empty_body(self, patched_urlopen, tmp_path):
        patched_urlopen(FakeHTTPResponse(b""))
        probe = CancellationProbe()

        gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                           is_cancelled=probe)

        assert probe.call_count == 1

    @pytest.mark.parametrize("falsy_value", [False, None, 0, "", []])
    def test_a_falsy_cancellation_answer_lets_the_download_finish(self, patched_urlopen, tmp_path, falsy_value):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 9)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "wheel.whl"

        written_bytes = gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path),
                                           is_cancelled=lambda: falsy_value)

        assert written_bytes == len(payload)
        assert destination_path.read_bytes() == payload

    @pytest.mark.parametrize("truthy_value", [True, 1, "yes", ["cancel"]])
    def test_any_truthy_cancellation_answer_stops_the_download(self, patched_urlopen, tmp_path, truthy_value):
        patched_urlopen(FakeHTTPResponse(payload_of_length(64)))

        with pytest.raises(RuntimeError, match="Cancelled"):
            gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(tmp_path / "wheel.whl"),
                               is_cancelled=lambda: truthy_value)

    def test_downloads_normally_when_no_cancellation_callback_is_given(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 2)
        patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / "wheel.whl"

        written_bytes = gpu_pack._download("https://files.pythonhosted.org/wheel.whl", str(destination_path),
                                           is_cancelled=None)

        assert written_bytes == len(payload)
        assert destination_path.read_bytes() == payload


class TestResolveThenDownload:
    def test_the_resolved_wheel_url_and_size_drive_a_matching_download(self, patched_urlopen, tmp_path):
        payload = payload_of_length(gpu_pack.DOWNLOAD_CHUNK_BYTES + 11)
        chosen_wheel = windows_wheel("12.4.5.8", size=len(payload))
        metadata_recorder = patched_urlopen(json_response(pypi_metadata({"12.4.5.8": [chosen_wheel]})))

        resolved = gpu_pack._resolve_wheel(PACKAGE_NAME, None)

        assert len(metadata_recorder.calls) == 1

        download_recorder = patched_urlopen(FakeHTTPResponse(payload))
        destination_path = tmp_path / resolved["filename"]
        progress = ProgressRecorder()

        written_bytes = gpu_pack._download(resolved["url"], str(destination_path),
                                           on_progress=progress, total_bytes=resolved["size"])

        assert download_recorder.last_call["url"] == chosen_wheel["url"]
        assert written_bytes == resolved["size"]
        assert progress.calls[-1] == (len(payload), len(payload))
        assert destination_path.read_bytes() == payload
