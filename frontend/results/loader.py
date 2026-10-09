"""Fetches run listings, manifests and metrics from the backend, and caches them."""

from __future__ import annotations

from urllib.parse import quote, urlencode

from PyQt6.QtCore import QObject, pyqtSignal

from .data import Manifest, MetricTable, Source

FILE_CACHE_SIZE = 40  # artifact previews kept in memory


def _seg(text: str) -> str:
    return quote(text, safe="")


class ResultsLoader(QObject):
    """Talks to the backend for the Results panel.

    `backend` needs get(path, callback(data, error)) for JSON and get_bytes(path, callback) for files.
    """

    runs_listed = pyqtSignal(list)  # GET /api/runs items
    manifest_loaded = pyqtSignal(object)  # Manifest
    table_loaded = pyqtSignal(object)  # Source whose MetricTable is now in self.tables
    file_loaded = pyqtSignal(str, str)  # run key, artifact path now in self.files
    failed = pyqtSignal(str)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.manifests: dict[str, Manifest] = {}
        self.tables: dict[Source, MetricTable] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self._pending: set = set()

    def list_runs(self):
        self.backend.get("/api/runs", self._runs_received)

    def _runs_received(self, data, error):
        if error:
            self.failed.emit(error)
            return
        self.runs_listed.emit(list((data or {}).get("runs") or []))

    def load_manifest(self, group: str, experiment_id: str):
        key = f"{group}/{experiment_id}"
        if key in self.manifests or ("manifest", key) in self._pending:
            return
        self._pending.add(("manifest", key))
        path = f"/api/runs/{_seg(group)}/{_seg(experiment_id)}/manifest"
        self.backend.get(path, lambda data, error: self._manifest_received(key, data, error))

    def _manifest_received(self, key, data, error):
        self._pending.discard(("manifest", key))
        if error:
            self.failed.emit(f"{key}: {error}")
            return
        manifest = Manifest.from_api(data)
        self.manifests[key] = manifest
        self.manifest_loaded.emit(manifest)

    def load_table(self, source: Source):
        """Fetch a source's metrics unless cached or already on the way; table_loaded fires when ready."""
        if source in self.tables:
            self.table_loaded.emit(source)
            return
        if source in self._pending:
            return
        self._pending.add(source)
        query = f"?{urlencode({'version': source.version})}" if source.version else ""
        path = f"/api/runs/{_seg(source.group)}/{_seg(source.experiment_id)}/{_seg(source.agent)}/metrics{query}"
        self.backend.get(path, lambda data, error: self._table_received(source, data, error))

    def _table_received(self, source, data, error):
        self._pending.discard(source)
        if error:
            self.failed.emit(f"{source.label}: {error}")
            return
        self.tables[source] = MetricTable.from_rows(data.get("metrics") or [], data.get("columns"))
        self.table_loaded.emit(source)

    def load_file(self, run_key: str, path: str):
        """Fetch one artifact's bytes (`path` as its manifest lists it); file_loaded fires when ready."""
        key = (run_key, path)
        if key in self.files:
            self.file_loaded.emit(run_key, path)
            return
        if key in self._pending:
            return
        self._pending.add(key)
        group, experiment_id = run_key.split("/", 1)
        url = f"/api/runs/{_seg(group)}/{_seg(experiment_id)}/files/{quote(path, safe='/')}"
        self.backend.get_bytes(url, lambda data, error: self._file_received(key, data, error))

    def _file_received(self, key, data, error):
        self._pending.discard(key)
        if error:
            self.failed.emit(f"{key[1]}: {error}")
            return
        self.files[key] = data
        while len(self.files) > FILE_CACHE_SIZE:  # keep memory bounded: drop the oldest preview
            self.files.pop(next(iter(self.files)))
        self.file_loaded.emit(*key)

    def clear(self):
        """Forget cached data, so the next requests re-read results/ (e.g. after Refresh)."""
        self.manifests.clear()
        self.tables.clear()
        self.files.clear()
