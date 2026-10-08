"""The ThetaIDE logo as the application icon: title bars, the Windows taskbar, the macOS Dock and Linux launchers.

- Windows groups taskbar buttons by AppUserModelID. Without one, ThetaIDE is grouped under python.exe and
  shows Python's icon; with its own ID the taskbar shows the window icon (the logo).
- macOS: QApplication.setWindowIcon also sets the Dock icon while the app runs.
- Linux: X11 window managers read the window icon. Wayland compositors (and launchers) look up the icon
  through a .desktop file named after the app's desktop-file name, which install_desktop_entry() writes.
"""
import os
import sys
from pathlib import Path

ICON_PATH = Path(__file__).resolve().parent / "icons" / "theta_bracket_icon.svg"
DESKTOP_FILE_NAME = "ThetaIDE"  # QApplication.setDesktopFileName; the Linux .desktop file must match it
WINDOWS_APP_ID = "ThetaIDE.ResearchWorkspace"
ICON_NAME = "thetaide"  # icon-theme name used by the .desktop entry


def set_windows_app_id(app_id=WINDOWS_APP_ID):
    """Give the process its own taskbar identity. Must run before the first window is shown."""
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        return ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id)) == 0
    except (AttributeError, OSError):
        return False


def app_icon():
    """The logo as a QIcon (SVG, so every size the platform asks for is sharp)."""
    from PyQt6.QtGui import QIcon

    return QIcon(str(ICON_PATH))


def desktop_entry(python, repo_root):
    """Text of the Linux .desktop entry that launches ThetaIDE from `repo_root` with `python`."""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=ThetaIDE\n"
        "Comment=Research workspace for NeSyRL experiments\n"
        f'Exec="{python}" -m frontend\n'
        f"Path={repo_root}\n"
        f"Icon={ICON_NAME}\n"
        f"StartupWMClass={DESKTOP_FILE_NAME}\n"
        "Terminal=false\n"
        "Categories=Development;Science;\n"
    )


def install_desktop_entry(data_home=None, python=sys.executable):
    """Install the .desktop entry and the logo for the current user (Linux, freedesktop.org layout).

    Writes $XDG_DATA_HOME/applications/ThetaIDE.desktop and
    $XDG_DATA_HOME/icons/hicolor/scalable/apps/thetaide.svg (XDG_DATA_HOME defaults to ~/.local/share).
    Returns the two paths.
    """
    data_home = Path(data_home or os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    icon = data_home / "icons" / "hicolor" / "scalable" / "apps" / f"{ICON_NAME}.svg"
    entry = data_home / "applications" / f"{DESKTOP_FILE_NAME}.desktop"
    icon.parent.mkdir(parents=True, exist_ok=True)
    entry.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(ICON_PATH.read_bytes())
    entry.write_text(desktop_entry(python, ICON_PATH.parents[2]), encoding="utf-8")
    return entry, icon
