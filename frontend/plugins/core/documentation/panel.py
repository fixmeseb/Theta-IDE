"""Interactive Documentation panel widget for Theta-IDE."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt, QUrl
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from frontend.sidetabs import svg_icon
from frontend.theme import current_theme, theme_color
from frontend.widgets import label

from .content import ARTICLES, DocArticle, search_articles

if TYPE_CHECKING:
    from frontend.plugins.context import PluginContext

_BUTTON_ICON_COLORS = {(QIcon.Mode.Normal, QIcon.State.Off): "text"}
_SEARCH_ICON_COLORS = {(QIcon.Mode.Normal, QIcon.State.Off): "muted"}
_CODE_BLOCK = re.compile(r"<pre><code>(.*?)</code></pre>", re.DOTALL)


def _theme_key():
    return tuple(sorted(current_theme()["colors"].items()))


def _boxed_code_blocks(article_html: str) -> str:
    """Wrap each <pre><code> block in a one-cell table.

    Qt's rich text paints a <pre> background line by line (every line is its own block), so code
    showed as separated stripes, and it ignores border-radius. A table cell is one solid box with
    padding and a border, which is what a code block should look like.
    """
    border, fill = theme_color("border"), theme_color("base")
    return _CODE_BLOCK.sub(
        lambda m: (f'<table class="code" width="100%" cellspacing="0" cellpadding="12" border="1" '
                   f'style="border-color: {border}; border-style: solid; background-color: {fill};">'
                   f'<tr><td><pre>{m.group(1)}</pre></td></tr></table>'),
        article_html,
    )


class DocumentationPanel(QWidget):
    """Main documentation browser panel in Theta-IDE."""

    def __init__(self, context: PluginContext | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.context = context
        self.setObjectName("documentationPanel")
        self.current_article_id: str = "overview"
        self._styled_for = None
        self._init_ui()
        self._populate_categories()
        self._populate_articles()
        self._apply_theme()
        self._fit_contents_width()
        self.select_article("overview")

    def _fit_contents_width(self, cap: int = 380):
        """Open the table of contents wide enough for the longest title (up to `cap` px)."""
        metrics = self.article_list.fontMetrics()
        longest = max((metrics.horizontalAdvance(art.title) for art in ARTICLES), default=200)
        frame = 2 * self.article_list.frameWidth() + 8 + 20 + 16  # list padding, item padding, scrollbar
        self.splitter.setSizes([min(cap, longest + frame), 900])

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        # ── Top Toolbar ─────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search documentation, topics, or keywords…")
        self.search_action = self.search_input.addAction(QIcon(), QLineEdit.ActionPosition.LeadingPosition)
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(260)
        self.search_input.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self.search_input, 1)

        # Category Filter
        self.cat_combo = QComboBox()
        self.cat_combo.setMinimumWidth(160)
        self.cat_combo.currentIndexChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.cat_combo)

        # Quick IDE Navigation Actions
        self.btn_hotkeys = QPushButton("Hotkeys")
        self.btn_hotkeys.setToolTip("Open Settings › Hotkeys")
        self.btn_hotkeys.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hotkeys.clicked.connect(self._open_hotkeys)
        toolbar.addWidget(self.btn_hotkeys)

        self.btn_hub = QPushButton("Community Hub")
        self.btn_hub.setToolTip("Explore Community Hub components")
        self.btn_hub.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hub.clicked.connect(self._open_hub)
        toolbar.addWidget(self.btn_hub)

        root.addLayout(toolbar)

        # ── Main Splitter ────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left Column: Topic List
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_header = label("TABLE OF CONTENTS", "eyebrow")
        left_layout.addWidget(left_header)

        self.article_list = QListWidget()
        self.article_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        # No horizontal scrollbar: the list opens wide enough for its titles (see _fit_contents_width),
        # and if it's dragged narrower they elide, with the full title in the tooltip
        self.article_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.article_list.itemClicked.connect(self._on_article_clicked)
        left_layout.addWidget(self.article_list, 1)

        splitter.addWidget(left_container)

        # Right Column: Content Viewer
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Header Info Card
        self.header_card = QFrame()
        self.header_card.setObjectName("card")
        hc_layout = QVBoxLayout(self.header_card)
        hc_layout.setContentsMargins(14, 12, 14, 12)
        hc_layout.setSpacing(4)

        meta_row = QHBoxLayout()
        self.badge_cat = label("GETTING STARTED", "badge")
        meta_row.addWidget(self.badge_cat)
        meta_row.addStretch()
        hc_layout.addLayout(meta_row)

        self.title_lbl = label("Welcome & Architecture Overview", "heading")
        self.title_lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
        hc_layout.addWidget(self.title_lbl)

        self.summary_lbl = label("High-level architecture overview.", "muted")
        self.summary_lbl.setWordWrap(True)
        hc_layout.addWidget(self.summary_lbl)

        right_layout.addWidget(self.header_card)

        # Rich Text Browser
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_anchor_clicked)
        right_layout.addWidget(self.browser, 1)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.splitter = splitter
        root.addWidget(splitter, 1)

    # ── Theme ───────────────────────────────────────────────────────────────────

    def changeEvent(self, event):
        # A theme switch sets the application palette, which reaches every widget as a PaletteChange
        super().changeEvent(event)
        if (event.type() == QEvent.Type.PaletteChange and self._styled_for is not None
                and self._styled_for != _theme_key()):
            self._apply_theme()
            scroll = self.browser.verticalScrollBar().value()
            self.select_article(self.current_article_id)  # the article's HTML carries theme colors
            self.browser.verticalScrollBar().setValue(scroll)

    def _apply_theme(self):
        """Colors for the panel's own widgets, all from the current theme."""
        self._styled_for = _theme_key()
        border, raised = theme_color("border"), theme_color("raised")
        self.article_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {border}; border-radius: 6px; padding: 4px; outline: 0; }} "
            "QListWidget::item { padding: 8px 10px; border: 0; border-radius: 4px; margin-bottom: 2px; } "
            f"QListWidget::item:hover {{ background: {theme_color('surface')}; }} "
            f"QListWidget::item:selected {{ background: {raised}; color: {theme_color('accent')}; "
            "font-weight: 600; }"
        )
        self.header_card.setStyleSheet(
            f"QFrame#card {{ background: {theme_color('surface')}; border: 1px solid {border}; border-radius: 6px; }}"
        )
        self.browser.setStyleSheet(
            f"QTextBrowser {{ border: 1px solid {border}; border-radius: 6px; padding: 16px; font-size: 13px; }}"
        )
        self.btn_hotkeys.setIcon(svg_icon("hotkeys", _BUTTON_ICON_COLORS, size=16))
        self.btn_hub.setIcon(svg_icon("hub", _BUTTON_ICON_COLORS, size=16))
        self.search_action.setIcon(svg_icon("search", _SEARCH_ICON_COLORS, size=16))

    def _article_css(self) -> str:
        """Styles for an article. Qt rich text supports a CSS subset (no border-radius, no rgba,
        no :hover or :nth-child), so this keeps to what it renders, in theme colors."""
        text_col, accent_col, border = theme_color("text"), theme_color("accent"), theme_color("border")
        return f"""
            body {{ color: {text_col}; font-size: 13px; }}
            h2 {{ color: {accent_col}; margin-top: 4px; margin-bottom: 12px; font-size: 18px; }}
            h3 {{ color: {accent_col}; margin-top: 18px; margin-bottom: 8px; font-size: 15px; }}
            p, li {{ color: {text_col}; }}
            code {{ color: {accent_col}; background-color: {theme_color('surface')};
                    font-family: ui-monospace, Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 12px; }}
            pre {{ margin: 0; color: {text_col};
                   font-family: ui-monospace, Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 12px; }}
            pre code {{ background-color: transparent; color: {text_col}; }}
            a {{ color: {accent_col}; text-decoration: none; font-weight: 500; }}
            table {{ margin-top: 10px; margin-bottom: 10px; border-collapse: collapse; }}
            th, td {{ padding: 8px 12px; border: 1px solid {border}; text-align: left; }}
            th {{ background-color: {theme_color('surface')}; }}
        """

    # ── Articles ────────────────────────────────────────────────────────────────

    def _populate_categories(self):
        self.cat_combo.clear()
        self.cat_combo.addItem("All Categories", "all")
        categories = sorted({art.category for art in ARTICLES})
        for cat in categories:
            self.cat_combo.addItem(cat, cat)

    def _populate_articles(self):
        self.article_list.clear()
        query = self.search_input.text().strip()
        cat = self.cat_combo.currentData()

        matched = search_articles(query)
        if cat and cat != "all":
            matched = [art for art in matched if art.category == cat]

        for art in matched:
            item = QListWidgetItem(f"{art.title}")
            item.setData(Qt.ItemDataRole.UserRole, art.id)
            item.setToolTip(f"{art.title}\n[{art.category}] {art.summary}")
            self.article_list.addItem(item)

        if self.article_list.count() == 0:
            empty_item = QListWidgetItem("No matching topics found")
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.article_list.addItem(empty_item)

    def _on_search_changed(self, text: str):
        self._populate_articles()
        if self.article_list.count() > 0:
            first_item = self.article_list.item(0)
            art_id = first_item.data(Qt.ItemDataRole.UserRole)
            if art_id:
                self.select_article(art_id)

    def _on_filter_changed(self, index: int):
        self._populate_articles()
        if self.article_list.count() > 0:
            first_item = self.article_list.item(0)
            art_id = first_item.data(Qt.ItemDataRole.UserRole)
            if art_id:
                self.select_article(art_id)

    def _on_article_clicked(self, item: QListWidgetItem):
        art_id = item.data(Qt.ItemDataRole.UserRole)
        if art_id:
            self.select_article(art_id)

    def select_article(self, article_id: str):
        self.current_article_id = article_id

        # Update List Selection
        for i in range(self.article_list.count()):
            item = self.article_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == article_id:
                self.article_list.setCurrentItem(item)
                break

        # Find Article
        article: DocArticle | None = None
        for art in ARTICLES:
            if art.id == article_id:
                article = art
                break

        if not article:
            return

        # Update Header Card
        self.badge_cat.setText(article.category.upper())
        self.title_lbl.setText(article.title)
        self.summary_lbl.setText(article.summary)

        self.browser.document().setDefaultStyleSheet(self._article_css())
        self.browser.setHtml(f"<html><body>{_boxed_code_blocks(article.html_content)}</body></html>")

    def _on_anchor_clicked(self, url: QUrl):
        url_str = url.toString()
        if url_str.startswith("ide://"):
            pane_id = url_str.replace("ide://", "")
            self._navigate_ide(pane_id)
        elif url_str.startswith("#"):
            # Internal anchor in document
            self.browser.scrollToAnchor(url_str.lstrip("#"))
        else:
            # Check if it's an article ID link
            for art in ARTICLES:
                if art.id == url_str:
                    self.select_article(art.id)
                    return

    # ── IDE navigation ──────────────────────────────────────────────────────────

    def _open_hotkeys(self):
        if self.context and hasattr(self.context, "open_settings"):
            self.context.open_settings("hotkeys")

    def _open_hub(self):
        if self.context and hasattr(self.context, "open_component_hub"):
            self.context.open_component_hub()

    def _navigate_ide(self, pane_id: str):
        if pane_id == "settings" and self.context and hasattr(self.context, "open_settings"):
            self.context.open_settings()  # Settings isn't a sidebar tab
        elif self.context and hasattr(self.context, "select_sidebar_tab"):
            self.context.select_sidebar_tab(pane_id)

    def cleanup(self):
        """Clean up resources on deactivation."""
        self.context = None
        self.browser.clear()
