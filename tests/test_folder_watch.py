import os

import pytest

from folder_watch import FolderWatcher

MEDIA = {".m4a", ".mp3", ".mp4"}


class SteppingClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return SteppingClock()


@pytest.fixture
def folder(tmp_path):
    target = tmp_path / "Downloads"
    target.mkdir()
    return target


def write(path, size=10):
    path.write_bytes(b"x" * size)
    return str(path)


def watch(folder, clock):
    return FolderWatcher(str(folder), MEDIA, clock=clock, settle_seconds=2.0, quiet_seconds=3.0)


def poll_for(watcher, clock, seconds, step=1.0):
    batches = []
    elapsed = 0.0
    while elapsed < seconds:
        clock.advance(step)
        elapsed += step
        batch = watcher.poll()
        if batch:
            batches.append(batch)
    return batches


class TestFolderWatcher:
    def test_files_already_there_when_watching_starts_are_ignored(self, folder, clock):
        write(folder / "old.m4a")
        watcher = watch(folder, clock)

        assert poll_for(watcher, clock, 10) == []

    def test_a_new_file_is_reported_once_it_has_settled_and_the_folder_went_quiet(self, folder, clock):
        watcher = watch(folder, clock)
        path = write(folder / "audio_message.m4a")

        assert poll_for(watcher, clock, 10) == [[path]]

    def test_a_new_file_is_not_reported_before_it_settles(self, folder, clock):
        watcher = watch(folder, clock)
        write(folder / "audio_message.m4a")

        assert poll_for(watcher, clock, 2) == []

    def test_a_file_that_keeps_growing_is_held_back(self, folder, clock):
        watcher = watch(folder, clock)
        growing = folder / "long.mp4"
        for size in range(1, 8):
            write(growing, size)
            clock.advance(1.0)
            assert watcher.poll() == []

        assert poll_for(watcher, clock, 10) == [[str(growing)]]

    def test_files_downloaded_close_together_arrive_as_one_batch_in_arrival_order(self, folder, clock):
        watcher = watch(folder, clock)
        first = write(folder / "b-first.m4a")
        os.utime(first, ns=(1_000_000_000, 1_000_000_000))
        clock.advance(1.0)
        watcher.poll()
        second = write(folder / "a-second.m4a")
        os.utime(second, ns=(2_000_000_000, 2_000_000_000))

        assert poll_for(watcher, clock, 12) == [[first, second]]

    def test_a_file_must_stay_unchanged_for_the_settle_time_before_it_counts(self, folder, clock):
        watcher = FolderWatcher(str(folder), MEDIA, clock=clock, settle_seconds=2.0, quiet_seconds=0.0)
        path = write(folder / "audio_message.m4a")
        clock.advance(1.0)
        first_sighting = watcher.poll()
        clock.advance(1.0)
        one_second_later = watcher.poll()
        clock.advance(1.0)
        two_seconds_later = watcher.poll()

        assert (first_sighting, one_second_later, two_seconds_later) == ([], [], [path])

    def test_a_file_arriving_inside_the_quiet_window_joins_the_waiting_batch(self, folder, clock):
        watcher = FolderWatcher(str(folder), MEDIA, clock=clock, settle_seconds=1.0, quiet_seconds=3.0)
        first = write(folder / "first.m4a")
        os.utime(first, ns=(1_000_000_000, 1_000_000_000))
        clock.advance(1.0)
        watcher.poll()
        clock.advance(1.0)
        assert watcher.poll() == []
        second = write(folder / "second.m4a")
        os.utime(second, ns=(2_000_000_000, 2_000_000_000))

        assert poll_for(watcher, clock, 10) == [[first, second]]

    def test_files_far_apart_arrive_as_separate_batches(self, folder, clock):
        watcher = watch(folder, clock)
        first = write(folder / "first.m4a")
        batches = poll_for(watcher, clock, 10)
        second = write(folder / "second.m4a")
        batches += poll_for(watcher, clock, 10)

        assert batches == [[first], [second]]

    def test_unsupported_files_are_never_reported(self, folder, clock):
        watcher = watch(folder, clock)
        write(folder / "invoice.pdf")
        write(folder / "song.m4a.crdownload")

        assert poll_for(watcher, clock, 10) == []

    def test_extensions_match_regardless_of_case(self, folder, clock):
        watcher = watch(folder, clock)
        path = write(folder / "VOICE.M4A")

        assert poll_for(watcher, clock, 10) == [[path]]

    def test_a_file_replaced_under_the_same_name_is_reported_again(self, folder, clock):
        watcher = watch(folder, clock)
        path = write(folder / "audio_message.m4a", 10)
        poll_for(watcher, clock, 10)
        write(folder / "audio_message.m4a", 25)

        assert poll_for(watcher, clock, 10) == [[path]]

    def test_a_file_that_disappears_before_settling_is_forgotten(self, folder, clock):
        watcher = watch(folder, clock)
        path = folder / "temporary.mp3"
        write(path)
        clock.advance(1.0)
        watcher.poll()
        path.unlink()

        assert poll_for(watcher, clock, 10) == []

    def test_a_settled_file_deleted_before_the_batch_is_left_out(self, folder, clock):
        watcher = watch(folder, clock)
        kept = write(folder / "kept.m4a")
        gone = folder / "gone.m4a"
        write(gone)
        for _ in range(3):
            clock.advance(1.0)
            watcher.poll()
        gone.unlink()

        assert poll_for(watcher, clock, 10) == [[kept]]

    def test_a_missing_folder_reports_nothing_and_does_not_raise(self, tmp_path, clock):
        watcher = watch(tmp_path / "not-there", clock)

        assert poll_for(watcher, clock, 10) == []

    def test_a_folder_created_after_watching_started_is_picked_up(self, tmp_path, clock):
        target = tmp_path / "later"
        watcher = watch(target, clock)
        target.mkdir()
        path = write(target / "clip.mp3")

        assert poll_for(watcher, clock, 10) == [[path]]

    def test_subfolders_are_not_watched(self, folder, clock):
        watcher = watch(folder, clock)
        nested = folder / "nested"
        nested.mkdir()
        write(nested / "clip.mp3")

        assert poll_for(watcher, clock, 10) == []
