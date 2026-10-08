"""Interactive Documentation panel widget for Theta-IDE."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from frontend.theme import theme_color
from frontend.widgets import label

from .content import ARTICLES, DocArticle, get_all_articles, search_articles

if TYPE_CHECKING:
    from frontend.plugins.context import PluginContext


class DocumentationPanel(QWidget):
    """Main documentation browser panel in Theta-IDE."""

    def __init__(self, context: PluginContext | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.context = context
        self.setObjectName("documentationPanel")
        self.current_article_id: str = "overview"
        self._init_ui()
        self._populate_categories()
        self._populate_articles()
        self.select_article("overview")

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        # ── Top Toolbar ─────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍  Search documentation, topics, or keywords…")
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
        btn_hotkeys = QPushButton("⚡ Hotkeys")
        btn_hotkeys.setToolTip("Switch to Hotkeys settings")
        btn_hotkeys.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_hotkeys.clicked.connect(lambda: self._navigate_ide("settings"))
        toolbar.addWidget(btn_hotkeys)

        btn_hub = QPushButton("🌐 Community Hub")
        btn_hub.setToolTip("Explore Community Hub extensions")
        btn_hub.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_hub.clicked.connect(lambda: self._navigate_ide("components"))
        toolbar.addWidget(btn_hub)

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
        self.article_list.setStyleSheet(
            "QListWidget { border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 4px; } "
            "QListWidget::item { padding: 8px 10px; border-radius: 4px; margin-bottom: 2px; } "
            "QListWidget::item:hover { background: rgba(255, 255, 255, 0.06); } "
            "QListWidget::item:selected { background: rgba(254, 128, 25, 0.22); color: #fe8019; font-weight: 600; }"
        )
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
        self.header_card.setStyleSheet(
            "QFrame#card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 12px; }"
        )
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
        self.browser.setStyleSheet(
            "QTextBrowser { border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 16px; font-size: 13px; line-height: 1.6; } "
        )
        right_layout.addWidget(self.browser, 1)

        splitter.addWidget(right_container)
        splitter.setSizes([260, 800])
        root.addWidget(splitter, 1)

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
            item.setToolTip(f"[{art.category}] {art.summary}")
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

        # Style HTML with Gruvbox / Theme Palette
        bg = theme_color("bg")
        text_col = theme_color("text")
        muted_col = theme_color("muted")
        accent_col = theme_color("accent")

        styled_html = f"""
        <html>
        <head>
          <style>
            body {{
              color: {text_col};
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
              font-size: 13px;
              line-height: 1.6;
            }}
            h2 {{
              color: {accent_col};
              margin-top: 4px;
              margin-bottom: 12px;
              font-size: 18px;
              border-bottom: 1px solid rgba(255, 255, 255, 0.1);
              padding-bottom: 6px;
            }}
            h3 {{
              color: {accent_col};
              margin-top: 18px;
              margin-bottom: 8px;
              font-size: 15px;
            }}
            p, li {{
              color: {text_col};
            }}
            code {{
              background-color: rgba(255, 255, 255, 0.08);
              color: #fabd2f;
              padding: 2px 5px;
              border-radius: 4px;
              font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
              font-size: 12px;
            }}
            pre {{
              background-color: rgba(0, 0, 0, 0.35);
              border: 1px solid rgba(255, 255, 255, 0.12);
              border-radius: 6px;
              padding: 12px 14px;
              margin: 10px 0;
            }}
            pre code {{
              background: transparent;
              padding: 0;
              color: #ebdbb2;
            }}
            a {{
              color: {accent_col};
              text-decoration: none;
              font-weight: 500;
            }}
            a:hover {{
              text-decoration: underline;
            }}
            table {{
              margin: 12px 0;
              border-collapse: collapse;
              width: 100%;
            }}
            th, td {{
              padding: 8px 12px;
              border: 1px solid rgba(255, 255, 255, 0.12);
              text-align: left;
            }}
            tr:nth-child(even) {{
              background-color: rgba(255, 255, 255, 0.02);
            }}
          </style>
        </head>
        <body>
          {article.html_content}
        </body>
        </html>
        """
        self.browser.setHtml(styled_html)

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

    def _navigate_ide(self, pane_id: str):
        if self.context and hasattr(self.context, "select_sidebar_tab"):
            self.context.select_sidebar_tab(pane_id)

    def cleanup(self):
        """Clean up resources on deactivation."""
        self.context = None
        self.browser.clear()
