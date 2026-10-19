"""Window management behaviour of SotvoxApp.

Dragging, resizing, minimising, maximising, work-area probing and opening the
output folder, all exercised against the Tk doubles so no real window is needed.
"""

import ctypes
import os
import re
import types

import doubles
import pytest

import ui

POSITION_ONLY_GEOMETRY = re.compile(r"^\+(-?\d+)\+(-?\d+)$")
SIZE_ONLY_GEOMETRY = re.compile(r"^(\d+)x(\d+)$")
FULL_GEOMETRY = re.compile(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$")


class PointerEvent:
    def __init__(self, x_root, y_root):
        self.x_root = x_root
        self.y_root = y_root


class WidgetPath:
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return self.path


class MapEvent:
    def __init__(self, widget_path):
        self.widget = WidgetPath(widget_path)


class FakeSystemParametersInfo:
    def __init__(self, rectangle=(0, 0, 1920, 1040), error=None):
        self.rectangle = rectangle
        self.error = error
        self.calls = []

    def __call__(self, action, parameter, rect, window_ini):
        self.calls.append((action, parameter, window_ini))
        if self.error is not None:
            raise self.error
        rect.left, rect.top, rect.right, rect.bottom = self.rectangle
        return 1


class ExplodingNamespace:
    def __init__(self, error):
        self._error = error

    def __getattr__(self, name):
        raise self._error


class StartFileRecorder:
    def __init__(self, error=None):
        self.error = error
        self.paths = []

    def __call__(self, path, *args):
        self.paths.append(path)
        if self.error is not None:
            raise self.error


def build_fake_ctypes(windll):
    return types.SimpleNamespace(
        Structure=ctypes.Structure,
        c_long=ctypes.c_long,
        byref=lambda obj: obj,
        windll=windll,
    )


def build_windll(system_parameters_info):
    return types.SimpleNamespace(
        user32=types.SimpleNamespace(SystemParametersInfoW=system_parameters_info))


@pytest.fixture
def work_area(monkeypatch):
    def install(rectangle=(0, 0, 1920, 1040)):
        system_parameters_info = FakeSystemParametersInfo(rectangle)
        monkeypatch.setattr(ui, "ctypes", build_fake_ctypes(build_windll(system_parameters_info)))
        return system_parameters_info

    return install


@pytest.fixture
def start_file(monkeypatch):
    def install(error=None):
        recorder = StartFileRecorder(error)
        monkeypatch.setattr(ui.os, "startfile", recorder)
        return recorder

    return install


class TestMinimize:
    def test_minimize_iconifies_the_root_window(self, app):
        app._minimize()

        assert app.root.iconified is True

    def test_minimize_leaves_the_geometry_untouched(self, app):
        app._minimize()

        assert app.root.geometry_calls == []

    def test_minimize_schedules_nothing(self, app):
        app._minimize()

        assert app.root.after_calls == []

    def test_minimize_stays_iconified_when_invoked_repeatedly(self, app):
        app._minimize()
        app._minimize()

        assert app.root.iconified is True

    def test_minimize_does_not_change_the_maximized_flag(self, app):
        app._maximized = True

        app._minimize()

        assert app._maximized is True


class TestRemapReappliesTheBorderlessStyle:
    def test_toplevel_map_event_schedules_the_borderless_style(self, app):
        app._on_remap(MapEvent("."))

        assert app.root.after_calls == [(10, app._make_borderless)]

    def test_borderless_style_is_deferred_rather_than_applied_inline(self, app):
        app._on_remap(MapEvent("."))

        assert app.root.pending == [(10, app._make_borderless)]

    @pytest.mark.parametrize(
        "widget_path",
        [".!frame", ".!frame.!label", ".!toplevel", ".!frame.!canvas", "", ".."],
    )
    def test_child_widget_map_events_are_ignored(self, app, widget_path):
        app._on_remap(MapEvent(widget_path))

        assert app.root.after_calls == []

    def test_each_toplevel_map_schedules_its_own_reapplication(self, app):
        app._on_remap(MapEvent("."))
        app._on_remap(MapEvent("."))
        app._on_remap(MapEvent(".!frame"))

        assert len(app.root.after_calls) == 2

    def test_remap_does_not_touch_the_window_geometry(self, app):
        app._on_remap(MapEvent("."))

        assert app.root.geometry_calls == []


class TestDragStart:
    def test_records_the_pointer_offset_from_the_window_origin(self, app):
        app._drag_start(PointerEvent(160, 70))

        assert app._drag_offset == (60, 20)

    @pytest.mark.parametrize(
        ("pointer_x", "pointer_y", "expected_offset"),
        [
            (100, 50, (0, 0)),
            (101, 51, (1, 1)),
            (700, 740, (600, 690)),
            (40, 30, (-60, -20)),
            (0, 0, (-100, -50)),
        ],
    )
    def test_offset_is_the_pointer_position_relative_to_the_window(
            self, app, pointer_x, pointer_y, expected_offset):
        app._drag_start(PointerEvent(pointer_x, pointer_y))

        assert app._drag_offset == expected_offset

    def test_drag_start_does_not_move_the_window(self, app):
        app._drag_start(PointerEvent(160, 70))

        assert app.root.geometry_calls == []

    def test_a_second_press_replaces_the_recorded_offset(self, app):
        app._drag_start(PointerEvent(160, 70))
        app._drag_start(PointerEvent(120, 60))

        assert app._drag_offset == (20, 10)

    def test_offset_is_recorded_even_while_maximized(self, app):
        app._maximized = True

        app._drag_start(PointerEvent(160, 70))

        assert app._drag_offset == (60, 20)

    def test_offset_follows_the_window_origin(self):
        app = doubles.make_app(root=doubles.FakeRoot(x=300, y=200))

        app._drag_start(PointerEvent(360, 220))

        assert app._drag_offset == (60, 20)


class TestDragMove:
    def test_moves_the_window_to_the_pointer_minus_the_offset(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(200, 100))

        assert app.root.geometry_calls == ["+140+80"]

    def test_geometry_string_carries_a_position_only(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(200, 100))

        assert POSITION_ONLY_GEOMETRY.fullmatch(app.root.geometry_calls[-1])

    def test_geometry_string_never_carries_a_size(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(200, 100))

        assert "x" not in app.root.geometry_calls[-1]

    def test_does_nothing_while_maximized(self, app):
        app._drag_start(PointerEvent(160, 70))
        app._maximized = True

        app._drag_move(PointerEvent(200, 100))

        assert app.root.geometry_calls == []

    def test_uses_the_default_offset_when_no_press_was_recorded(self, app):
        app._drag_move(PointerEvent(300, 200))

        assert app.root.geometry_calls == ["+300+200"]

    def test_allows_the_window_to_be_dragged_to_negative_coordinates(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(10, 5))

        assert app.root.geometry_calls == ["+-50+-15"]

    def test_successive_moves_each_record_one_geometry_call(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(200, 100))
        app._drag_move(PointerEvent(210, 110))
        app._drag_move(PointerEvent(220, 120))

        assert app.root.geometry_calls == ["+140+80", "+150+90", "+160+100"]

    def test_moves_are_absolute_rather_than_cumulative(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(500, 400))
        app._drag_move(PointerEvent(200, 100))

        assert app.root.geometry_calls[-1] == "+140+80"

    def test_keeps_using_the_offset_captured_at_press_time(self, app):
        app._drag_start(PointerEvent(160, 70))
        app.root._x = 999
        app.root._y = 999

        app._drag_move(PointerEvent(200, 100))

        assert app.root.geometry_calls == ["+140+80"]

    def test_dragging_does_not_resize_the_window(self, app):
        app._drag_start(PointerEvent(160, 70))

        app._drag_move(PointerEvent(200, 100))

        assert app._resize is None


class TestResizeStart:
    def test_captures_the_pointer_and_the_current_window_size(self, app):
        app._resize_start(PointerEvent(400, 500))

        assert app._resize == (400, 500, 588, 690)

    def test_does_not_move_or_resize_the_window_on_press(self, app):
        app._resize_start(PointerEvent(400, 500))

        assert app.root.geometry_calls == []

    def test_leaves_a_normal_window_unmaximized(self, app):
        app._resize_start(PointerEvent(400, 500))

        assert app._maximized is False
        assert app._max_button.glyph_history == []

    def test_unmaximizes_first_when_the_window_is_maximized(self, app, work_area):
        work_area()
        app._maximized = True

        app._resize_start(PointerEvent(400, 500))

        assert app._maximized is False

    def test_unmaximizing_restores_the_stored_geometry(self, app):
        app._maximized = True
        app._normal_geometry = "640x480+12+34"

        app._resize_start(PointerEvent(400, 500))

        assert app.root.geometry_calls == ["640x480+12+34"]

    def test_unmaximizing_switches_the_button_back_to_the_max_glyph(self, app):
        app._maximized = True

        app._resize_start(PointerEvent(400, 500))

        assert app._max_button.glyph == "max"

    def test_the_captured_size_is_read_from_the_root_window(self):
        app = doubles.make_app(root=doubles.FakeRoot(width=1024, height=768))

        app._resize_start(PointerEvent(10, 20))

        assert app._resize == (10, 20, 1024, 768)

    def test_a_second_press_replaces_the_previous_capture(self, app):
        app._resize_start(PointerEvent(400, 500))

        app._resize_start(PointerEvent(410, 520))

        assert app._resize == (410, 520, 588, 690)


class TestResizeMove:
    def test_growing_the_window_records_the_new_size(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(350, 460))

        assert app.root.geometry_calls == ["638x750"]

    def test_shrinking_the_window_records_the_new_size(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(290, 380))

        assert app.root.geometry_calls == ["578x670"]

    def test_geometry_string_carries_a_size_only(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(350, 460))

        assert SIZE_ONLY_GEOMETRY.fullmatch(app.root.geometry_calls[-1])

    def test_geometry_string_never_carries_a_position(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(250, 350))

        assert "+" not in app.root.geometry_calls[-1]

    @pytest.mark.parametrize(
        ("horizontal_delta", "expected_width"),
        [(200, 788), (50, 638), (0, 588), (-27, 561), (-28, 560), (-29, 560), (-2000, 560)],
    )
    def test_width_is_clamped_to_the_minimum_width(self, app, horizontal_delta, expected_width):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(300 + horizontal_delta, 400))

        assert app.root.geometry_calls[-1].split("x")[0] == str(expected_width)

    @pytest.mark.parametrize(
        ("vertical_delta", "expected_height"),
        [(200, 890), (60, 750), (0, 690), (-169, 521), (-170, 520), (-171, 520), (-2000, 520)],
    )
    def test_height_is_clamped_to_the_minimum_height(self, app, vertical_delta, expected_height):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(300, 400 + vertical_delta))

        assert app.root.geometry_calls[-1].split("x")[1] == str(expected_height)

    def test_both_dimensions_clamp_when_dragged_far_up_and_left(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(-5000, -5000))

        assert app.root.geometry_calls == ["560x520"]

    def test_width_clamps_independently_of_height(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(-5000, 500))

        assert app.root.geometry_calls == ["560x790"]

    def test_height_clamps_independently_of_width(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(400, -5000))

        assert app.root.geometry_calls == ["688x520"]

    def test_the_floors_come_from_the_configured_minimums(self):
        app = doubles.make_app(_min_w=300, _min_h=200)

        app._resize_start(PointerEvent(300, 400))
        app._resize_move(PointerEvent(-5000, -5000))

        assert app.root.geometry_calls == ["300x200"]

    def test_is_a_no_op_when_no_resize_is_in_progress(self, app):
        app._resize_move(PointerEvent(350, 460))

        assert app.root.geometry_calls == []

    def test_is_a_no_op_when_the_resize_capture_was_cleared(self, app):
        app._resize_start(PointerEvent(300, 400))
        app._resize = None

        app._resize_move(PointerEvent(350, 460))

        assert app.root.geometry_calls == []

    def test_successive_moves_are_all_relative_to_the_press(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(310, 410))
        app._resize_move(PointerEvent(320, 420))

        assert app.root.geometry_calls == ["598x700", "608x710"]

    def test_resizing_does_not_move_the_window_origin(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(350, 460))

        assert all("+" not in spec for spec in app.root.geometry_calls)

    def test_resizing_leaves_the_capture_in_place_for_further_motion(self, app):
        app._resize_start(PointerEvent(300, 400))

        app._resize_move(PointerEvent(350, 460))

        assert app._resize == (300, 400, 588, 690)


class TestToggleMaximize:
    def test_stores_the_current_geometry_before_maximizing(self, app, work_area):
        work_area()

        app._toggle_max()

        assert app._normal_geometry == "588x690+100+50"

    def test_resizes_the_window_to_the_work_area(self, app, work_area):
        work_area((0, 0, 1920, 1040))

        app._toggle_max()

        assert app.root.geometry_calls == ["1920x1040+0+0"]

    def test_work_area_origin_is_honoured(self, app, work_area):
        work_area((10, 20, 1600, 900))

        app._toggle_max()

        assert app.root.geometry_calls == ["1590x880+10+20"]

    def test_maximizing_sets_the_maximized_flag(self, app, work_area):
        work_area()

        app._toggle_max()

        assert app._maximized is True

    def test_maximizing_switches_the_button_to_the_restore_glyph(self, app, work_area):
        work_area()

        app._toggle_max()

        assert app._max_button.glyph == "restore"

    def test_maximized_geometry_is_a_full_size_and_position(self, app, work_area):
        work_area()

        app._toggle_max()

        assert FULL_GEOMETRY.fullmatch(app.root.geometry_calls[-1])

    def test_restoring_writes_back_the_stored_geometry(self, app):
        app._maximized = True
        app._normal_geometry = "800x600+40+30"

        app._toggle_max()

        assert app.root.geometry_calls == ["800x600+40+30"]

    def test_restoring_clears_the_maximized_flag(self, app):
        app._maximized = True

        app._toggle_max()

        assert app._maximized is False

    def test_restoring_switches_the_button_back_to_the_max_glyph(self, app):
        app._maximized = True

        app._toggle_max()

        assert app._max_button.glyph == "max"

    def test_restoring_does_not_query_the_work_area(self, app, work_area):
        system_parameters_info = work_area()
        app._maximized = True

        app._toggle_max()

        assert system_parameters_info.calls == []

    def test_round_trip_restores_the_original_geometry_exactly(self, app, work_area):
        work_area()
        original_geometry = app.root.geometry()

        app._toggle_max()
        app._toggle_max()

        assert app.root.geometry_calls[-1] == original_geometry

    def test_round_trip_restores_the_original_glyph_and_flag(self, app, work_area):
        work_area()

        app._toggle_max()
        app._toggle_max()

        assert app._max_button.glyph_history == ["restore", "max"]
        assert app._maximized is False

    def test_two_round_trips_keep_the_same_stored_geometry(self, app, work_area):
        work_area()

        app._toggle_max()
        app._toggle_max()
        app._toggle_max()
        app._toggle_max()

        assert app.root.geometry_calls[-1] == "588x690+100+50"

    def test_maximizing_captures_the_live_geometry_rather_than_the_stored_one(self, work_area):
        app = doubles.make_app(root=doubles.FakeRoot(x=7, y=9, width=800, height=600),
                               _normal_geometry="1x1+0+0")
        work_area()

        app._toggle_max()

        assert app._normal_geometry == "800x600+7+9"


class TestWorkArea:
    def test_returns_the_rectangle_filled_in_by_the_system(self, app, work_area):
        work_area((0, 0, 1920, 1040))

        assert app._work_area() == (0, 0, 1920, 1040)

    def test_queries_the_desktop_work_area_parameter(self, app, work_area):
        system_parameters_info = work_area()

        app._work_area()

        assert system_parameters_info.calls == [(0x0030, 0, 0)]

    @pytest.mark.parametrize(
        "rectangle",
        [(0, 0, 1920, 1040), (10, 20, 1600, 900), (-1920, 0, 0, 1080), (0, 0, 1, 1)],
    )
    def test_the_reported_rectangle_is_passed_through_unchanged(self, app, work_area, rectangle):
        work_area(rectangle)

        assert app._work_area() == rectangle

    def test_falls_back_to_the_whole_screen_when_the_system_call_raises(self, app, monkeypatch):
        failing = FakeSystemParametersInfo(error=OSError("SystemParametersInfoW failed"))
        monkeypatch.setattr(ui, "ctypes", build_fake_ctypes(build_windll(failing)))

        assert app._work_area() == (0, 0, 1920, 1080)

    def test_falls_back_when_the_user32_library_is_unreachable(self, app, monkeypatch):
        monkeypatch.setattr(
            ui, "ctypes", build_fake_ctypes(ExplodingNamespace(OSError("no user32"))))

        assert app._work_area() == (0, 0, 1920, 1080)

    def test_falls_back_when_the_rect_structure_cannot_be_declared(self, app, monkeypatch):
        broken = types.SimpleNamespace(Structure=None, c_long=None, byref=lambda obj: obj,
                                       windll=None)
        monkeypatch.setattr(ui, "ctypes", broken)

        assert app._work_area() == (0, 0, 1920, 1080)

    def test_the_fallback_rectangle_starts_at_the_screen_origin(self, app, monkeypatch):
        failing = FakeSystemParametersInfo(error=RuntimeError("boom"))
        monkeypatch.setattr(ui, "ctypes", build_fake_ctypes(build_windll(failing)))

        left, top, _right, _bottom = app._work_area()

        assert (left, top) == (0, 0)

    @pytest.mark.parametrize(
        ("screen_width", "screen_height"),
        [(800, 600), (1366, 768), (3840, 2160)],
    )
    def test_the_fallback_uses_the_reported_screen_size(self, monkeypatch, screen_width,
                                                        screen_height):
        app = doubles.make_app(root=doubles.FakeRoot(screen_width=screen_width,
                                                     screen_height=screen_height))
        failing = FakeSystemParametersInfo(error=OSError("nope"))
        monkeypatch.setattr(ui, "ctypes", build_fake_ctypes(build_windll(failing)))

        assert app._work_area() == (0, 0, screen_width, screen_height)

    def test_a_failed_probe_does_not_move_the_window(self, app, monkeypatch):
        failing = FakeSystemParametersInfo(error=OSError("nope"))
        monkeypatch.setattr(ui, "ctypes", build_fake_ctypes(build_windll(failing)))

        app._work_area()

        assert app.root.geometry_calls == []

    def test_maximizing_uses_the_fallback_rectangle_when_the_probe_fails(self, app, monkeypatch):
        failing = FakeSystemParametersInfo(error=OSError("nope"))
        monkeypatch.setattr(ui, "ctypes", build_fake_ctypes(build_windll(failing)))

        app._toggle_max()

        assert app.root.geometry_calls == ["1920x1080+0+0"]


class TestOpenOutputFolder:
    def test_creates_the_output_directory_when_it_does_not_exist(self, app, tmp_path, start_file):
        start_file()
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        app._open_output()

        assert target.is_dir()

    def test_opens_the_directory_with_the_shell(self, app, tmp_path, start_file):
        recorder = start_file()
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        app._open_output()

        assert recorder.paths == [str(target)]

    def test_creates_missing_parent_directories(self, app, tmp_path, start_file):
        start_file()
        target = tmp_path / "one" / "two" / "three"
        app.output_var.set(str(target))

        app._open_output()

        assert target.is_dir()

    def test_an_existing_directory_is_opened_without_complaint(self, app, tmp_path, start_file):
        recorder = start_file()
        target = tmp_path / "already-there"
        target.mkdir()
        app.output_var.set(str(target))

        app._open_output()

        assert recorder.paths == [str(target)]

    def test_existing_content_is_left_alone(self, app, tmp_path, start_file):
        start_file()
        target = tmp_path / "already-there"
        target.mkdir()
        keeper = target / "keep.txt"
        keeper.write_text("kept", encoding="utf-8")
        app.output_var.set(str(target))

        app._open_output()

        assert keeper.read_text(encoding="utf-8") == "kept"

    def test_records_the_opened_folder_in_the_session_log(self, app, tmp_path, start_file):
        start_file()
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        app._open_output()

        assert any(line.endswith(f"Opened output folder: {target}")
                   for line in app._session_log)

    def test_opens_the_folder_currently_held_by_the_output_variable(self, app, tmp_path,
                                                                    start_file):
        recorder = start_file()
        first = tmp_path / "first"
        second = tmp_path / "second"
        app.output_var.set(str(first))
        app.output_var.set(str(second))

        app._open_output()

        assert recorder.paths == [str(second)]

    def test_opening_twice_opens_the_folder_twice(self, app, tmp_path, start_file):
        recorder = start_file()
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        app._open_output()
        app._open_output()

        assert recorder.paths == [str(target), str(target)]

    def test_a_path_blocked_by_a_file_raises_before_the_shell_is_called(self, app, tmp_path,
                                                                        start_file):
        recorder = start_file()
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        app.output_var.set(str(blocker))

        with pytest.raises(OSError):
            app._open_output()

        assert recorder.paths == []

    def test_a_blocked_path_is_never_logged_as_opened(self, app, tmp_path, start_file):
        start_file()
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        app.output_var.set(str(blocker))

        with pytest.raises(OSError):
            app._open_output()

        assert not any("Opened output folder" in line for line in app._session_log)

    def test_a_failing_shell_call_propagates_after_the_folder_was_logged(self, app, tmp_path,
                                                                        start_file):
        start_file(error=OSError("no association"))
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        with pytest.raises(OSError):
            app._open_output()

        assert any("Opened output folder" in line for line in app._session_log)

    def test_the_directory_survives_a_failing_shell_call(self, app, tmp_path, start_file):
        start_file(error=OSError("no association"))
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        with pytest.raises(OSError):
            app._open_output()

        assert target.is_dir()

    def test_opening_the_folder_does_not_schedule_ui_work(self, app, tmp_path, start_file):
        start_file()
        app.output_var.set(str(tmp_path / "transcripts"))

        app._open_output()

        assert app.root.after_calls == []

    def test_the_created_directory_starts_empty(self, app, tmp_path, start_file):
        start_file()
        target = tmp_path / "transcripts"
        app.output_var.set(str(target))

        app._open_output()

        assert os.listdir(target) == []
