"""Non-blocking JSON client for the Theta-IDE backend API (src/app/api/app.py)."""
import json

from PyQt6.QtCore import QByteArray, QObject, QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

DEFAULT_URL = "http://127.0.0.1:8000"


class Backend(QObject):
    """Callbacks receive (data, error): parsed JSON on success, or None and a message."""

    def __init__(self, base_url=DEFAULT_URL, parent=None):
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.manager = QNetworkAccessManager(self)

    def get(self, path, callback):
        self._track(self.manager.get(self._request(path)), callback)

    def post(self, path, payload, callback):
        body = QByteArray(json.dumps(payload).encode("utf-8"))
        self._track(self.manager.post(self._request(path), body), callback)

    def _request(self, path):
        request = QNetworkRequest(QUrl(self.base_url + path))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        request.setTransferTimeout(10000)
        return request

    def _track(self, reply, callback):
        reply.finished.connect(lambda: self._finished(reply, callback))

    def _finished(self, reply, callback):
        data, error = None, None
        body = bytes(reply.readAll()).decode("utf-8", errors="replace")
        if reply.error() != QNetworkReply.NetworkError.NoError:
            error = reply.errorString()
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if status is not None:
                try:
                    error = f"HTTP {status}: {json.loads(body)['detail']}"  # FastAPI error message
                except (ValueError, KeyError, TypeError):
                    pass
        else:
            try:
                data = json.loads(body)
            except ValueError as exc:
                error = f"Invalid response from backend: {exc}"
        reply.deleteLater()
        callback(data, error)
