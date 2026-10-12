"""Community Hub cards take their text colors from the active theme, so they stay readable on light themes."""
import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.hub.dialog import HubComponentCard
    from frontend.hub.models import AuthorInfo, HubComponent
    from frontend.theme import BUILTINS
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

DESCRIPTION = "Link experiments to an Obsidian vault."


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHubCardFollowsTheme(unittest.TestCase):
    def setUp(self):
        self.previous = getattr(app, "theta_theme", None)

    def tearDown(self):
        if self.previous is None:
            if hasattr(app, "theta_theme"):
                del app.theta_theme
        else:
            app.theta_theme = self.previous

    def card(self, theme_name):
        app.theta_theme = BUILTINS[theme_name]
        component = HubComponent(id="obsidian", name="Obsidian Notes", kind="plugin", version="1.0",
                                 description=DESCRIPTION, author=AuthorInfo(name="Team"), tags=["notes"])
        card = HubComponentCard(component, None)
        self.addCleanup(card.deleteLater)
        return card

    def description_style(self, card):
        return next(lbl for lbl in card.findChildren(QLabel) if lbl.text() == DESCRIPTION).styleSheet()

    def test_description_uses_the_light_theme_text_color(self):
        style = self.description_style(self.card("Gruvbox Light"))
        self.assertIn(BUILTINS["Gruvbox Light"]["colors"]["text"], style)
        self.assertNotIn("#ebdbb2", style)  # Gruvbox Dark's cream, unreadable on a light card

    def test_dark_theme_keeps_its_colors(self):
        style = self.description_style(self.card("Gruvbox Dark"))
        self.assertIn(BUILTINS["Gruvbox Dark"]["colors"]["text"], style)

    def test_install_button_uses_the_theme_primary(self):
        card = self.card("Gruvbox Light")
        self.assertIn(BUILTINS["Gruvbox Light"]["colors"]["primary"], card.btn_action.styleSheet())


if __name__ == "__main__":
    unittest.main()
