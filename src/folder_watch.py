import os
import time

SETTLE_SECONDS = 2.0
QUIET_SECONDS = 3.0


class FolderWatcher:
    def __init__(self, folder, extensions, clock=time.monotonic,
                 settle_seconds=SETTLE_SECONDS, quiet_seconds=QUIET_SECONDS):
        self.folder = folder
        self._extensions = set(extensions)
        self._clock = clock
        self._settle_seconds = settle_seconds
        self._quiet_seconds = quiet_seconds
        self._known = self._scan()
        self._growing = {}
        self._settled = []
        self._last_change = None

    def _scan(self):
        found = {}
        try:
            entries = list(os.scandir(self.folder))
        except OSError:
            entries = []
        for entry in entries:
            extension = os.path.splitext(entry.name)[1].lower()
            if extension not in self._extensions:
                continue
            try:
                if entry.is_file():
                    status = entry.stat()
                    found[os.path.abspath(entry.path)] = (status.st_size, status.st_mtime_ns)
            except OSError:
                continue
        return found

    def poll(self):
        now = self._clock()
        current = self._scan()
        for path, signature in current.items():
            if self._known.get(path) == signature:
                continue
            observed = self._growing.get(path)
            if observed is None or observed[0] != signature:
                self._growing[path] = (signature, now)
                self._last_change = now
            elif now - observed[1] >= self._settle_seconds:
                del self._growing[path]
                self._known[path] = signature
                self._settled.append(path)
        for vanished in [path for path in self._growing if path not in current]:
            del self._growing[vanished]
        batch_is_ready = (self._settled and not self._growing
                          and now - self._last_change >= self._quiet_seconds)
        if batch_is_ready:
            still_present = [path for path in self._settled if path in current]
            batch = sorted(still_present, key=lambda path: current[path][1])
            self._settled = []
        else:
            batch = []
        return batch
