import os
import struct
import subprocess
import sys
import time
import uuid

from constants import IS_FROZEN, IS_MAC, IS_WINDOWS

if IS_WINDOWS:
    import winreg
else:
    winreg = None

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE_NAME = "Sotvox"
MINIMIZED_FLAG = "--minimized"
DOWNLOADS_FOLDER_ID = "{374DE290-123F-4565-9164-39C4925E467B}"
SINGLE_INSTANCE_MUTEX = "Local\\SotvoxSingleInstance"
ERROR_ALREADY_EXISTS = 183
SW_RESTORE = 9
TK_WINDOW_CLASS = "TkTopLevel"
APP_WINDOW_TITLE = "Sotvox"
_held_instance_handles = []

CF_HDROP = 15
GMEM_MOVEABLE = 0x0002
DROPEFFECT_COPY = 1
PREFERRED_DROP_EFFECT_FORMAT = "Preferred DropEffect"
DROPFILES_HEADER = struct.Struct("<IiiII")
DROP_EFFECT_VALUE = struct.Struct("<I")
WIDE_CHARACTERS = 1
NULL_CHARACTER = "\0"
OPEN_CLIPBOARD_ATTEMPTS = 10
OPEN_CLIPBOARD_RETRY_SECONDS = 0.02
NO_SELECTION_FLAGS = 0
OSASCRIPT_TIMEOUT_SECONDS = 15

MAC_REVEAL_SCRIPT = """ObjC.import("AppKit");
function run(argv) {
    var urls = $.NSMutableArray.alloc.init;
    argv.forEach(function (path) { urls.addObject($.NSURL.fileURLWithPath(path)); });
    $.NSWorkspace.sharedWorkspace.activateFileViewerSelectingURLs(urls);
}"""

MAC_COPY_SCRIPT = """ObjC.import("AppKit");
function run(argv) {
    var urls = $.NSMutableArray.alloc.init;
    argv.forEach(function (path) { urls.addObject($.NSURL.fileURLWithPath(path)); });
    var pasteboard = $.NSPasteboard.generalPasteboard;
    pasteboard.clearContents;
    if (!pasteboard.writeObjects(urls)) {
        throw new Error("the pasteboard refused the files");
    }
}"""


def build_drop_files_payload(paths):
    header = DROPFILES_HEADER.pack(DROPFILES_HEADER.size, 0, 0, 0, WIDE_CHARACTERS)
    file_list = "".join(os.path.abspath(path) + NULL_CHARACTER for path in paths) + NULL_CHARACTER
    return header + file_list.encode("utf-16-le")


def reveal_files(folder, paths):
    if not paths:
        succeeded = False
    elif IS_WINDOWS:
        succeeded = _reveal_files_windows(folder, paths)
    elif IS_MAC:
        succeeded = _run_mac_script(MAC_REVEAL_SCRIPT, paths)
    else:
        succeeded = False
    return succeeded


def copy_files_to_clipboard(paths, owner_window=None):
    if not paths:
        succeeded = False
    elif IS_WINDOWS:
        succeeded = _copy_files_windows(paths, owner_window)
    elif IS_MAC:
        succeeded = _run_mac_script(MAC_COPY_SCRIPT, paths)
    else:
        succeeded = False
    return succeeded


def _load_shell_api():
    import ctypes
    from ctypes import wintypes

    shell32 = ctypes.WinDLL("shell32")
    ole32 = ctypes.WinDLL("ole32")
    shell32.ILCreateFromPathW.argtypes = [wintypes.LPCWSTR]
    shell32.ILCreateFromPathW.restype = ctypes.c_void_p
    shell32.ILFree.argtypes = [ctypes.c_void_p]
    shell32.ILFree.restype = None
    shell32.SHOpenFolderAndSelectItems.argtypes = [
        ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD]
    shell32.SHOpenFolderAndSelectItems.restype = ctypes.c_long
    ole32.CoInitialize.argtypes = [ctypes.c_void_p]
    ole32.CoInitialize.restype = ctypes.c_long
    return shell32, ole32


def _load_clipboard_api():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    kernel32 = ctypes.WinDLL("kernel32")
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL
    return user32, kernel32


def _reveal_files_windows(folder, paths):
    import ctypes

    shell32, ole32 = _load_shell_api()
    ole32.CoInitialize(None)
    folder_item_list = shell32.ILCreateFromPathW(os.path.abspath(folder))
    file_item_lists = [shell32.ILCreateFromPathW(os.path.abspath(path)) for path in paths]
    resolved_item_lists = [item_list for item_list in file_item_lists if item_list]
    try:
        if not folder_item_list or not resolved_item_lists:
            succeeded = False
        else:
            selection = (ctypes.c_void_p * len(resolved_item_lists))(*resolved_item_lists)
            result = shell32.SHOpenFolderAndSelectItems(
                folder_item_list, len(resolved_item_lists), selection, NO_SELECTION_FLAGS)
            succeeded = result >= 0
    finally:
        for item_list in [folder_item_list, *resolved_item_lists]:
            if item_list:
                shell32.ILFree(item_list)
    return succeeded


def _copy_to_global_memory(kernel32, data):
    import ctypes

    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not handle:
        raise OSError("GlobalAlloc failed")
    pointer = kernel32.GlobalLock(handle)
    if not pointer:
        kernel32.GlobalFree(handle)
        raise OSError("GlobalLock failed")
    ctypes.memmove(pointer, data, len(data))
    kernel32.GlobalUnlock(handle)
    return handle


def _open_clipboard(user32, owner_window):
    opened = False
    attempt = 0
    while not opened and attempt < OPEN_CLIPBOARD_ATTEMPTS:
        opened = bool(user32.OpenClipboard(owner_window))
        attempt += 1
        if not opened and attempt < OPEN_CLIPBOARD_ATTEMPTS:
            time.sleep(OPEN_CLIPBOARD_RETRY_SECONDS)
    return opened


def _fill_clipboard(user32, kernel32, paths):
    user32.EmptyClipboard()
    drop_effect_format = user32.RegisterClipboardFormatW(PREFERRED_DROP_EFFECT_FORMAT)
    payloads = [
        (CF_HDROP, build_drop_files_payload(paths)),
        (drop_effect_format, DROP_EFFECT_VALUE.pack(DROPEFFECT_COPY)),
    ]
    succeeded = True
    for clipboard_format, data in payloads:
        handle = _copy_to_global_memory(kernel32, data)
        if not user32.SetClipboardData(clipboard_format, handle):
            kernel32.GlobalFree(handle)
            succeeded = False
            break
    return succeeded


def _copy_files_windows(paths, owner_window):
    user32, kernel32 = _load_clipboard_api()
    if _open_clipboard(user32, owner_window):
        try:
            succeeded = _fill_clipboard(user32, kernel32, paths)
        finally:
            user32.CloseClipboard()
    else:
        succeeded = False
    return succeeded


def _run_mac_script(script, paths):
    absolute_paths = [os.path.abspath(path) for path in paths]
    command = ["osascript", "-l", "JavaScript", "-e", script, *absolute_paths]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=OSASCRIPT_TIMEOUT_SECONDS)
        succeeded = completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        succeeded = False
    return succeeded


def downloads_folder():
    found = None
    if IS_WINDOWS:
        try:
            found = _known_folder_path(DOWNLOADS_FOLDER_ID)
        except OSError:
            found = None
    return found or os.path.join(os.path.expanduser("~"), "Downloads")


def _known_folder_path(folder_id):
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    parsed = uuid.UUID(folder_id)
    folder_guid = GUID(parsed.time_low, parsed.time_mid, parsed.time_hi_version,
                       (ctypes.c_ubyte * 8)(*parsed.bytes[8:]))
    shell32 = ctypes.WinDLL("shell32")
    ole32 = ctypes.WinDLL("ole32")
    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None
    path_pointer = ctypes.c_void_p()
    result = shell32.SHGetKnownFolderPath(ctypes.byref(folder_guid), 0, None, ctypes.byref(path_pointer))
    try:
        found = ctypes.wstring_at(path_pointer.value) if result >= 0 and path_pointer.value else None
    finally:
        if path_pointer.value:
            ole32.CoTaskMemFree(path_pointer)
    return found


def startup_command():
    if IS_FROZEN:
        command = f'"{sys.executable}" {MINIMIZED_FLAG}'
    else:
        windowless_python = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        launcher = windowless_python if os.path.exists(windowless_python) else sys.executable
        entry_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        command = f'"{launcher}" "{entry_script}" {MINIMIZED_FLAG}'
    return command


def is_start_with_windows_enabled():
    if winreg is None:
        enabled = False
    else:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as key:
                winreg.QueryValueEx(key, RUN_VALUE_NAME)
            enabled = True
        except OSError:
            enabled = False
    return enabled


def set_start_with_windows(enabled):
    if winreg is None:
        return False
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, RUN_VALUE_NAME, 0, winreg.REG_SZ, startup_command())
        else:
            try:
                winreg.DeleteValue(key, RUN_VALUE_NAME)
            except FileNotFoundError:
                pass
    return True


def claim_single_instance():
    if not IS_WINDOWS:
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    handle = kernel32.CreateMutexW(None, False, SINGLE_INSTANCE_MUTEX)
    already_running = ctypes.get_last_error() == ERROR_ALREADY_EXISTS
    _held_instance_handles.append(handle)
    return not already_running


def focus_running_instance():
    if not IS_WINDOWS:
        return False
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    window = user32.FindWindowW(TK_WINDOW_CLASS, APP_WINDOW_TITLE)
    if window:
        user32.ShowWindow(window, SW_RESTORE)
        user32.SetForegroundWindow(window)
    return bool(window)
