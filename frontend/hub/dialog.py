"""Theta Hub Browser dialog for discovering and installing community components."""
from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .client import HubClient
from .models import HubComponent


class HubComponentCard(QFrame):
    """Card widget representing an individual component in the Hub."""

    def __init__(self, component: HubComponent, client: HubClient, parent=None):
        super().__init__(parent)
        self.component = component
        self.client = client
        self.setObjectName("card")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # ── Top Row: Badge, Title, Version, Author & Action Button ───────────
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Kind badge
        kind_colors = {
            "plugin": ("#b16286", "#d3869b"),   # Purple
            "method": ("#98971a", "#b8bb26"),   # Green
            "model": ("#458588", "#83a598"),    # Blue
            "env": ("#d79921", "#fabd2f"),      # Yellow/Orange
            "experiment": ("#cc241d", "#fb4934")# Red
        }
        badge_bg, badge_fg = kind_colors.get(self.component.kind, ("#665c54", "#ebdbb2"))
        kind_badge = QLabel(self.component.kind.upper())
        kind_badge.setStyleSheet(
            f"background-color: {badge_bg}; color: #ebdbb2; font-size: 10px; "
            f"font-weight: 700; padding: 2px 6px; border-radius: 4px;"
        )
        top_row.addWidget(kind_badge)

        # Title
        title_label = QLabel(self.component.name)
        title_label.setStyleSheet("font-size: 14px; font-weight: 700;")
        top_row.addWidget(title_label)

        # Version
        version_label = QLabel(f"v{self.component.version}")
        version_label.setStyleSheet("color: #928374; font-size: 12px;")
        top_row.addWidget(version_label)

        # Author
        author_text = f"by {self.component.author.name}"
        author_label = QLabel(author_text)
        author_label.setStyleSheet("color: #a89984; font-size: 11px;")
        top_row.addWidget(author_label)

        top_row.addStretch()

        # Action Button
        self.btn_action = QPushButton()
        self.btn_action.setFixedHeight(28)
        self.btn_action.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_action.clicked.connect(self._on_action_clicked)
        top_row.addWidget(self.btn_action)

        layout.addLayout(top_row)

        # ── Description ──────────────────────────────────────────────────────
        desc_label = QLabel(self.component.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #ebdbb2; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc_label)

        # ── Bottom Row: Tags & Links ─────────────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        for tag in self.component.tags[:5]:
            tag_label = QLabel(f"#{tag}")
            tag_label.setStyleSheet("color: #83a598; font-size: 11px;")
            bottom_row.addWidget(tag_label)

        bottom_row.addStretch()

        if self.component.repository:
            btn_repo = QToolButton()
            btn_repo.setText("GitHub")
            btn_repo.setStyleSheet("color: #8ec07c; font-size: 11px; border: none; background: transparent;")
            btn_repo.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.component.repository)))
            bottom_row.addWidget(btn_repo)

        layout.addLayout(bottom_row)

        # ── Download Progress Bar (hidden by default) ────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.update_action_state()

    def update_action_state(self):
        """Update button text and style according to local installation state."""
        if self.component.is_installed:
            if self.component.has_update:
                self.btn_action.setText(f"Update to v{self.component.version}")
                self.btn_action.setObjectName("actionUpdate")
                self.btn_action.setStyleSheet(
                    "QPushButton {"
                    "  background-color: #458588;"
                    "  color: #ebdbb2;"
                    "  border: 1px solid #83a598;"
                    "  border-radius: 4px;"
                    "  font-weight: 700;"
                    "  padding: 4px 14px;"
                    "}"
                    "QPushButton:hover {"
                    "  background-color: #83a598;"
                    "  color: #1d2021;"
                    "  border-color: #83a598;"
                    "}"
                    "QPushButton:disabled {"
                    "  background-color: #32302f;"
                    "  color: #665c54;"
                    "  border-color: #3c3836;"
                    "}"
                )
                self.btn_action.setEnabled(True)
            else:
                self.btn_action.setText("Uninstall")
                self.btn_action.setObjectName("actionUninstall")
                self.btn_action.setStyleSheet(
                    "QPushButton {"
                    "  background-color: #cc241d;"
                    "  color: #ebdbb2;"
                    "  border: 1px solid #cc241d;"
                    "  border-radius: 4px;"
                    "  font-weight: 700;"
                    "  padding: 4px 14px;"
                    "}"
                    "QPushButton:hover {"
                    "  background-color: #fb4934;"
                    "  color: #1d2021;"
                    "  border-color: #fb4934;"
                    "}"
                    "QPushButton:disabled {"
                    "  background-color: #32302f;"
                    "  color: #665c54;"
                    "  border-color: #3c3836;"
                    "}"
                )
                self.btn_action.setEnabled(True)
        else:
            self.btn_action.setText("Install")
            self.btn_action.setObjectName("actionInstall")
            self.btn_action.setStyleSheet(
                "QPushButton {"
                "  background-color: #b8bb26;"
                "  color: #1d2021;"
                "  border: 1px solid #b8bb26;"
                "  border-radius: 4px;"
                "  font-weight: 700;"
                "  padding: 4px 14px;"
                "}"
                "QPushButton:hover {"
                "  background-color: #c7c94b;"
                "  border-color: #c7c94b;"
                "}"
                "QPushButton:disabled {"
                "  background-color: #32302f;"
                "  color: #665c54;"
                "  border-color: #3c3836;"
                "}"
            )
            self.btn_action.setEnabled(True)
        self.btn_action.style().unpolish(self.btn_action)
        self.btn_action.style().polish(self.btn_action)

    def _on_action_clicked(self):
        self.btn_action.setEnabled(False)
        if self.component.is_installed and not self.component.has_update:
            self._prompt_and_uninstall()
        else:
            self.btn_action.setText("Installing…")
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.client.install_async(self.component)

    def _prompt_and_uninstall(self):
        has_configs = self.client.installer.has_configs(self.component)
        if not has_configs:
            self.btn_action.setText("Uninstalling…")
            self.client.uninstall_async(self.component, remove_configs=False)
            return

        pref = self.client.get_uninstall_configs_pref()
        if pref == "keep":
            self.btn_action.setText("Uninstalling…")
            self.client.uninstall_async(self.component, remove_configs=False)
            return
        elif pref == "remove":
            self.btn_action.setText("Uninstalling…")
            self.client.uninstall_async(self.component, remove_configs=True)
            return

        dlg = UninstallComponentDialog(self.component.name, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.choice == "cancel":
            self.btn_action.setEnabled(True)
            self.btn_action.setText("Uninstall")
            return

        if dlg.remember:
            self.client.set_uninstall_configs_pref(dlg.choice)

        remove_configs = (dlg.choice == "remove")
        self.btn_action.setText("Uninstalling…")
        self.client.uninstall_async(self.component, remove_configs=remove_configs)

    def set_progress(self, percent: int):
        self.progress_bar.setValue(percent)


class UninstallComponentDialog(QDialog):
    """Prompt asking whether to keep or remove configuration files on component uninstall."""

    def __init__(self, component_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Uninstall Component")
        self.setFixedWidth(430)
        self.choice = "cancel"
        self.remember = False

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Uninstall {component_name}?")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ebdbb2;")
        layout.addWidget(title)

        desc = QLabel(
            "This component has associated configuration files in <b>in/config/</b>.<br><br>"
            "Would you like to keep your configuration files or delete them as well?"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a89984; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        self.chk_remember = QCheckBox("Don't ask again (remember my choice)")
        self.chk_remember.setStyleSheet("color: #ebdbb2; font-size: 12px; margin-top: 4px;")
        layout.addWidget(self.chk_remember)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch()

        btn_keep = QPushButton("Keep Configs")
        btn_keep.setToolTip("Uninstall component code but preserve your YAML configuration files")
        btn_keep.clicked.connect(self._on_keep)
        btn_row.addWidget(btn_keep)

        btn_remove = QPushButton("Remove All")
        btn_remove.setStyleSheet("background-color: #cc241d; color: #ebdbb2; font-weight: bold;")
        btn_remove.setToolTip("Uninstall component code and delete all associated YAML configurations")
        btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(btn_remove)

        layout.addLayout(btn_row)

    def _on_cancel(self):
        self.choice = "cancel"
        self.reject()

    def _on_keep(self):
        self.choice = "keep"
        self.remember = self.chk_remember.isChecked()
        self.accept()

    def _on_remove(self):
        self.choice = "remove"
        self.remember = self.chk_remember.isChecked()
        self.accept()


class HubView(QWidget):
    """Full-featured marketplace view for exploring and installing components."""
    close_requested = pyqtSignal()

    def __init__(
        self,
        client: HubClient,
        initial_kind: str | None = None,
        close_button_text: str = "Back to Parameters",
        parent=None,
    ):
        super().__init__(parent)
        self.client = client
        self.active_kind: str | None = initial_kind
        self.close_button_text = close_button_text
        self.cards: Dict[str, HubComponentCard] = {}

        self._build_ui()
        self._wire_client()

        # Load components
        self.client.fetch_index_async()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 14)
        main_layout.setSpacing(12)

        # ── Header ───────────────────────────────────────────────────────────
        header = QVBoxLayout()
        header.setSpacing(3)

        eyebrow = QLabel("THETA COMMUNITY HUB")
        eyebrow.setObjectName("eyebrow")
        eyebrow.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #83a598;")
        header.addWidget(eyebrow)

        title = QLabel("Explore & Install Components")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        header.addWidget(title)

        desc = QLabel(
            "Discover community plugins, reinforcement learning algorithms, "
            "symbolic models, and benchmark environments."
        )
        desc.setStyleSheet("color: #a89984; font-size: 12px;")
        header.addWidget(desc)
        main_layout.addLayout(header)

        # ── Search & Filter Bar ──────────────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search components by name, author, tag, or description…")
        self.search_input.textChanged.connect(self._apply_filters)
        search_row.addWidget(self.search_input, 1)

        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("color: #a89984; font-size: 11px;")
        search_row.addWidget(sort_lbl)

        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Default", "default")
        self.sort_combo.addItem("Uninstalled first", "uninstalled_first")
        self.sort_combo.addItem("Installed first", "installed_first")
        self.sort_combo.addItem("Name (A–Z)", "name_asc")
        self.sort_combo.setToolTip("Sort components list")
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        search_row.addWidget(self.sort_combo)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setToolTip("Reload registry from GitHub")
        self.btn_refresh.clicked.connect(lambda: self.client.fetch_index_async(force=True))
        search_row.addWidget(self.btn_refresh)

        main_layout.addLayout(search_row)

        # ── Kind Filter Pills ────────────────────────────────────────────────
        pill_row = QHBoxLayout()
        pill_row.setSpacing(6)

        self.pills = {}
        kind_options = [
            ("all", "All Components"),
            ("plugin", "Plugins"),
            ("method", "RL Methods"),
            ("model", "Models"),
            ("env", "Environments"),
        ]

        for kind_id, label_text in kind_options:
            btn = QPushButton(label_text)
            btn.setCheckable(True)
            if (self.active_kind or "all") == kind_id:
                btn.setChecked(True)
            btn.clicked.connect(lambda chk, k=kind_id: self._select_kind(k))
            self.pills[kind_id] = btn
            pill_row.addWidget(btn)

        pill_row.addStretch()
        main_layout.addLayout(pill_row)

        # ── Components Scroll Area ───────────────────────────────────────────
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(0, 4, 6, 4)
        self.card_layout.setSpacing(10)
        self.card_layout.addStretch()

        self.scroll_area.setWidget(self.card_container)
        main_layout.addWidget(self.scroll_area, 1)

        # ── Footer / Status ──────────────────────────────────────────────────
        footer = QHBoxLayout()
        self.status_label = QLabel("Loading community registry…")
        self.status_label.setStyleSheet("color: #928374; font-size: 11px;")
        footer.addWidget(self.status_label, 1)

        self.btn_close = QPushButton(self.close_button_text)
        self.btn_close.clicked.connect(self.close_requested.emit)
        footer.addWidget(self.btn_close)

        main_layout.addLayout(footer)

    def _wire_client(self):
        self.client.indexLoaded.connect(self._on_index_loaded)
        self.client.fetchFailed.connect(self._on_fetch_failed)
        self.client.installProgress.connect(self._on_install_progress)
        self.client.installFinished.connect(self._on_install_finished)
        self.client.uninstallFinished.connect(self._on_uninstall_finished)

    def _select_kind(self, kind_id: str):
        self.active_kind = None if kind_id == "all" else kind_id
        for kid, btn in self.pills.items():
            btn.blockSignals(True)
            btn.setChecked(kid == kind_id)
            btn.blockSignals(False)
        self._apply_filters()

    def _on_index_loaded(self, components: List[HubComponent]):
        self.status_label.setText(f"Registry ready · {len(components)} items available")
        self._populate_cards(components)

    def _on_fetch_failed(self, error: str):
        self.status_label.setText(f"Using local cache (Network notice: {error})")
        self._populate_cards(self.client.components)

    def _populate_cards(self, components: List[HubComponent]):
        # Remove existing cards
        for card in self.cards.values():
            self.card_layout.removeWidget(card)
            card.deleteLater()
        self.cards.clear()

        # Build cards
        for comp in components:
            card = HubComponentCard(comp, self.client, self.card_container)
            self.cards[comp.id] = card
            # Insert before the trailing stretch spacer
            self.card_layout.insertWidget(self.card_layout.count() - 1, card)

        self._apply_filters()

    def _apply_filters(self):
        query = self.search_input.text()
        matched = list(self.client.search(query=query, kind=self.active_kind))

        sort_mode = self.sort_combo.currentData() if hasattr(self, "sort_combo") else "default"
        if sort_mode == "uninstalled_first":
            # False (uninstalled) comes before True (installed)
            matched.sort(key=lambda c: (c.is_installed, c.name.lower()))
        elif sort_mode == "installed_first":
            matched.sort(key=lambda c: (not c.is_installed, c.name.lower()))
        elif sort_mode == "name_asc":
            matched.sort(key=lambda c: c.name.lower())

        visible_count = 0
        matched_ids = {c.id for c in matched}

        # Re-pack layout in sorted order
        for comp in matched:
            card = self.cards.get(comp.id)
            if card:
                self.card_layout.removeWidget(card)
                self.card_layout.insertWidget(visible_count, card)
                card.setVisible(True)
                visible_count += 1

        for comp_id, card in self.cards.items():
            if comp_id not in matched_ids:
                card.setVisible(False)

        self.status_label.setText(
            f"Showing {visible_count} of {len(self.client.components)} components"
        )

    def _on_install_progress(self, comp_id: str, percent: int):
        if comp_id in self.cards:
            self.cards[comp_id].set_progress(percent)

    def _on_install_finished(self, comp_id: str, success: bool, message: str):
        if comp_id in self.cards:
            card = self.cards[comp_id]
            card.progress_bar.hide()
            card.update_action_state()
            if success:
                self.status_label.setText(f"Installed {card.component.name} successfully.")
            else:
                self.status_label.setText(f"Installation failed: {message}")
        self._apply_filters()

    def _on_uninstall_finished(self, comp_id: str, success: bool, message: str):
        if comp_id in self.cards:
            card = self.cards[comp_id]
            card.update_action_state()
            if success:
                self.status_label.setText(f"Uninstalled {card.component.name}.")
            else:
                self.status_label.setText(f"Uninstall failed: {message}")
        self._apply_filters()


class HubDialog(QDialog):
    """Full-featured marketplace dialog for exploring and installing components."""

    def __init__(self, client: HubClient, initial_kind: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theta Community Hub")
        self.resize(860, 680)
        self.setMinimumSize(700, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.hub_view = HubView(
            client=client,
            initial_kind=initial_kind,
            close_button_text="Close",
            parent=self,
        )
        self.hub_view.close_requested.connect(self.accept)
        layout.addWidget(self.hub_view)

    @property
    def client(self):
        return self.hub_view.client

    @property
    def cards(self):
        return self.hub_view.cards

    @property
    def pills(self):
        return self.hub_view.pills

    @property
    def sort_combo(self):
        return self.hub_view.sort_combo

    @property
    def card_layout(self):
        return self.hub_view.card_layout

    @property
    def search_input(self):
        return self.hub_view.search_input

    def _populate_cards(self, components: List[HubComponent]):
        return self.hub_view._populate_cards(components)

    def _apply_filters(self):
        return self.hub_view._apply_filters()

    def __getattr__(self, name):
        return getattr(self.hub_view, name)
