"""Application icon setup (frontend/app_icon.py): Windows taskbar ID and the Linux desktop entry."""
import sys
from unittest import mock

import pytest

from frontend import app_icon


def test_logo_file_exists():
    assert app_icon.ICON_PATH.is_file() and app_icon.ICON_PATH.suffix == ".svg"


def test_windows_app_id_is_skipped_elsewhere():
    with mock.patch.object(app_icon.sys, "platform", "linux"):
        assert app_icon.set_windows_app_id() is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_app_id_is_set():
    with mock.patch("ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID", return_value=0) as call:
        assert app_icon.set_windows_app_id() is True
    assert call.call_args.args[0].value == app_icon.WINDOWS_APP_ID


def test_desktop_entry_matches_the_app():
    text = app_icon.desktop_entry("/usr/bin/python3", "/home/me/Offline-BlendRL")
    lines = dict(line.split("=", 1) for line in text.splitlines()[1:])
    assert text.startswith("[Desktop Entry]\n")
    assert lines["Exec"] == '"/usr/bin/python3" -m frontend' and lines["Path"] == "/home/me/Offline-BlendRL"
    assert lines["Icon"] == app_icon.ICON_NAME
    assert lines["StartupWMClass"] == app_icon.DESKTOP_FILE_NAME


def test_install_desktop_entry(tmp_path):
    entry, icon = app_icon.install_desktop_entry(data_home=tmp_path, python="/opt/venv/bin/python")
    # The file name must equal QApplication.setDesktopFileName so Wayland compositors find the icon.
    assert entry == tmp_path / "applications" / f"{app_icon.DESKTOP_FILE_NAME}.desktop"
    assert icon == tmp_path / "icons" / "hicolor" / "scalable" / "apps" / f"{app_icon.ICON_NAME}.svg"
    assert icon.read_bytes() == app_icon.ICON_PATH.read_bytes()
    assert 'Exec="/opt/venv/bin/python" -m frontend' in entry.read_text(encoding="utf-8")
    assert f"Path={app_icon.ICON_PATH.parents[2]}" in entry.read_text(encoding="utf-8")
