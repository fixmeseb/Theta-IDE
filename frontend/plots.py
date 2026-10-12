"""Interactive local plot viewer, using Qt without additional dependencies."""
import csv
import io
import math
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QFileDialog,
    QMessageBox, QStackedWidget, QGraphicsView, QGraphicsScene,
)
from .widgets import label
from .theme import theme_color


def read_metrics_csv(path):
    """Read finite numeric columns of a CSV file; retain gaps and ignore text metadata columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return read_metrics_csv_stream(stream)


def read_metrics_csv_text(text):
    """read_metrics_csv() for CSV text already in memory (e.g. a file served by the backend)."""
    return read_metrics_csv_stream(io.StringIO(text.lstrip("\ufeff"), newline=""))


def read_metrics_csv_stream(stream):
    """Numeric columns of a CSV stream, by header; text columns are skipped."""
    reader = csv.DictReader(stream)
    headers = reader.fieldnames
    if not headers or len(set(headers)) != len(headers) or any(not h.strip() for h in headers):
        raise ValueError("CSV needs unique, nonempty column headers.")
    rows = []
    for row in reader:
        if len(rows) >= 50000:
            raise ValueError("This preview supports CSV files with up to 50,000 rows.")
        if None in row or None in row.values():
            raise ValueError("A CSV row has a different number of fields than its header.")
        rows.append(row)
    columns = {}
    for header in headers:
        values = []
        numeric = True
        for row in rows:
            text = row[header].strip()
            try:
                value = float(text) if text else None
            except ValueError:
                numeric = False
                break  # Text metadata is not an axis candidate.
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Column '{header}' contains NaN or infinity.")
            values.append(value)
        if numeric and any(v is not None for v in values):
            columns[header] = values
    if len(columns) < 2:
        raise ValueError("Choose a CSV with at least two numeric columns and one data row.")
    return columns


class PlotCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.series = []
        self.xlabel, self.ylabel = "step", "reward"
        self.bounds = (0., 1., 0., 1.)
        self.auto_fit = True
        self.drag = None
        self.hover = None

    def area(self):
        return QRectF(76, 38, max(1, self.width() - 124), max(1, self.height() - 140))

    def set_data(self, series, xlabel, ylabel, reset=True):
        self.series, self.xlabel, self.ylabel = series, xlabel, ylabel
        self.hover = None
        if reset or self.auto_fit:
            self.reset_view()
        self.update()

    def reset_view(self):
        points = [p for _, data, _ in self.series for p in data]
        if points:
            xs, ys = zip(*points)
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            dx = (xmax - xmin) * .05 or max(abs(xmin) * .05, .5)
            dy = (ymax - ymin) * .08 or max(abs(ymin) * .05, .5)
            self.bounds = xmin - dx, xmax + dx, ymin - dy, ymax + dy
        else:
            self.bounds = 0., 1., 0., 1.
        self.auto_fit = True
        self.update()

    def project(self, x, y):
        a = self.area()
        x0, x1, y0, y1 = self.bounds
        return QPointF(a.left() + (x - x0) / (x1 - x0) * a.width(),
                       a.bottom() - (y - y0) / (y1 - y0) * a.height())

    def zoom(self, factor, position=None):
        a = self.area()
        position = position or a.center()
        fx = (position.x() - a.left()) / a.width()
        fy = (a.bottom() - position.y()) / a.height()
        x0, x1, y0, y1 = self.bounds
        x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        bounds = (x + (x0 - x) * factor, x + (x1 - x) * factor,
                  y + (y0 - y) * factor, y + (y1 - y) * factor)
        if all(math.isfinite(v) for v in bounds) and bounds[1] > bounds[0] and bounds[3] > bounds[2]:
            self.bounds = bounds
            self.auto_fit = False
            self.hover = None
            self.update()

    def wheelEvent(self, event):
        if self.area().contains(event.position()):
            self.zoom(.8 if event.angleDelta().y() > 0 else 1.25, event.position())
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.area().contains(event.position()):
            self.drag = (event.position(), self.bounds)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag:
            origin, (x0, x1, y0, y1) = self.drag
            dx = (event.position().x() - origin.x()) / self.area().width() * (x1 - x0)
            dy = (event.position().y() - origin.y()) / self.area().height() * (y1 - y0)
            self.bounds = x0 - dx, x1 - dx, y0 + dy, y1 + dy
            self.auto_fit = False
            self.hover = None
        else:
            self.hover = event.position() if self.area().contains(event.position()) else None
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def leaveEvent(self, event):
        self.hover = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme_color("base")))
        p.setFont(QFont("Segoe UI", 9))
        a = self.area()
        x0, x1, y0, y1 = self.bounds
        p.setPen(QColor(theme_color("text")))
        p.drawText(18, 23, f"{self.ylabel} / {self.xlabel}")
        for i in range(6):
            f = i / 5
            x, y = a.left() + a.width() * f, a.bottom() - a.height() * f
            p.setPen(QColor(theme_color("raised")))
            p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            p.setPen(QColor(theme_color("muted")))
            p.drawText(QRectF(x - 35, a.bottom() + 7, 70, 20), Qt.AlignmentFlag.AlignCenter, f"{x0 + (x1 - x0) * f:.4g}")
            p.drawText(QRectF(2, y - 10, 66, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{y0 + (y1 - y0) * f:.4g}")
        p.drawText(QRectF(a.left(), a.bottom() + 30, a.width(), 20), Qt.AlignmentFlag.AlignCenter, self.xlabel)
        p.save()
        p.setClipRect(a)
        nearest = None
        for name, points, color in self.series:
            path = QPainterPath()
            for i, (x, y) in enumerate(points):
                point = self.project(x, y)
                if i == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
                if self.hover and a.contains(point):
                    distance = (point.x() - self.hover.x()) ** 2 + (point.y() - self.hover.y()) ** 2
                    if nearest is None or distance < nearest[0]:
                        nearest = distance, point, x, y, color
            p.setPen(QPen(QColor(theme_color(color)), 2))
            p.drawPath(path)
            if len(points) == 1:
                p.drawEllipse(self.project(*points[0]), 3, 3)
        if nearest and nearest[0] < 1600:
            _, point, x, y, color = nearest
            p.setPen(QPen(QColor(theme_color(color)), 2))
            p.drawEllipse(point, 5, 5)
            self.setToolTip(f"{self.xlabel}: {x:.6g}\n{self.ylabel}: {y:.6g}")
        else:
            self.setToolTip("")
        p.restore()
        if not any(data for _, data, _ in self.series):
            p.setPen(QColor(theme_color("muted")))
            p.drawText(a, Qt.AlignmentFlag.AlignCenter, "No plottable points yet.\nStart a demo run or open a metrics CSV.")
        for i, (name, _, color) in enumerate(self.series):
            p.setPen(QColor(theme_color(color)))
            p.drawText(QRectF(a.left(), self.height() - 43 + i * 19, a.width(), 19),
                       Qt.AlignmentFlag.AlignLeft, f"●  {name}")


class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(theme_color("base")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def load(self, pixmap):
        self.scene().clear()
        self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.reset_view()

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor(theme_color("base")))

    def reset_view(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        scale = self.transform().m11() / factor
        if .01 <= scale <= 100:
            self.scale(1 / factor, 1 / factor)

    def wheelEvent(self, event):
        self.zoom(.8 if event.angleDelta().y() > 0 else 1.25)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()


class PlotViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.runs = []
        self.imports = {}
        self.signature = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(label("Plot viewer", "heading"))
        tools = QHBoxLayout()
        for text, callback in (("Open CSV / image…", self.open_file), ("+", lambda: self.current_canvas().zoom(.8)),
                               ("−", lambda: self.current_canvas().zoom(1.25)),
                               ("Fit", lambda: self.current_canvas().reset_view()), ("Export PNG…", self.export_png)):
            button = QPushButton(text)
            if text in ("+", "−"):
                button.setToolTip("Zoom in" if text == "+" else "Zoom out")
                button.setAccessibleName(button.toolTip())
            button.clicked.connect(callback)
            tools.addWidget(button)
        tools.addStretch()
        layout.addLayout(tools)
        sources = QHBoxLayout()
        self.source, self.compare = QComboBox(), QComboBox()
        for combo in (self.source, self.compare):
            combo.setMinimumWidth(100)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        sources.addWidget(label("Source", "muted"))
        sources.addWidget(self.source, 1)
        sources.addWidget(label("Overlay", "muted"))
        sources.addWidget(self.compare, 1)
        sources.addStretch()
        layout.addLayout(sources)
        axes = QHBoxLayout()
        self.xaxis, self.yaxis = QComboBox(), QComboBox()
        for combo in (self.source, self.compare):
            combo.setMaximumWidth(420)
        for combo in (self.xaxis, self.yaxis):
            combo.setMaximumWidth(240)
        axes.addWidget(label("X", "muted"))
        axes.addWidget(self.xaxis, 1)
        axes.addWidget(label("Y", "muted"))
        axes.addWidget(self.yaxis, 1)
        axes.addStretch()
        layout.addLayout(axes)
        self.stack = QStackedWidget()
        self.canvas, self.image = PlotCanvas(), ImageCanvas()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.image)
        layout.addWidget(self.stack, 1)
        self.caption = label("", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        layout.addWidget(label("Scroll to zoom · Drag to pan · Double-click to fit · Hover a curve to inspect", "muted"))
        self.source.currentIndexChanged.connect(self.source_changed)
        self.compare.currentIndexChanged.connect(self.redraw)
        self.xaxis.currentTextChanged.connect(self.redraw)
        self.yaxis.currentTextChanged.connect(self.redraw)

    def current_canvas(self):
        return self.stack.currentWidget()

    def update_runs(self, runs):
        self.runs = runs
        ids = tuple(run["id"] for run in runs)
        if ids != self.signature:
            self.signature = ids
            self.populate_sources()
        else:
            self.redraw(reset=False)

    def populate_sources(self):
        old = self.source.currentData()
        overlay = self.compare.currentData()
        self.source.blockSignals(True)
        self.compare.blockSignals(True)
        self.source.clear()
        self.compare.clear()
        self.compare.addItem("None", None)
        for run in self.runs:
            title = f"{run['config']['name']} · {run['id']}"
            self.source.addItem(title, run["id"])
            self.compare.addItem(title, run["id"])
        for key, (title, _) in self.imports.items():
            self.source.addItem(title, key)
        index = self.source.findData(old)
        self.source.setCurrentIndex(index if index >= 0 else 0)
        self.compare.setCurrentIndex(max(0, self.compare.findData(overlay)))
        self.source.blockSignals(False)
        self.compare.blockSignals(False)
        self.source_changed()

    def data_for(self, key):
        if key in self.imports:
            return self.imports[key]
        for run in self.runs:
            if run["id"] == key:
                source = "simulated" if run.get("simulated", True) else "trained"
                return f"{run['config']['name']} [{run['id']}] · {source}", {
                    field: [m.get(field) for m in run["metrics"]]
                    for field in ["step"] + sorted({k for m in run["metrics"] for k in m} - {"step"})}
        return "No source", {}

    def source_changed(self, *_):
        _, data = self.data_for(self.source.currentData())
        is_image = isinstance(data, QPixmap)
        self.stack.setCurrentIndex(1 if is_image else 0)
        for widget in (self.xaxis, self.yaxis, self.compare):
            widget.setEnabled(not is_image)
        if is_image:
            self.image.load(data)
            self.caption.setText(f"Local image · {data.width()} × {data.height()} pixels")
            return
        for combo, preferred in ((self.xaxis, "step"), (self.yaxis, "reward")):
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(data))
            choice = previous if previous in data else preferred
            index = combo.findText(choice)
            combo.setCurrentIndex(index if index >= 0 else min(1 if combo is self.yaxis else 0, len(data) - 1))
            combo.blockSignals(False)
        self.redraw()

    def redraw(self, *_, reset=True):
        name, data = self.data_for(self.source.currentData())
        if isinstance(data, QPixmap):
            return
        x, y = self.xaxis.currentText(), self.yaxis.currentText()
        series = []
        warnings = []
        sources = [(name, data, "#b8bb26")]
        if self.compare.currentData() and self.compare.currentData() != self.source.currentData():
            other_name, other_data = self.data_for(self.compare.currentData())
            sources.append((other_name, other_data, "#83a598"))
        for title, columns, color in sources:
            if x not in columns or y not in columns:
                warnings.append(f"{title}: chosen axes unavailable")
                continue
            pairs = sorted((a, b) for a, b in zip(columns[x], columns[y]) if a is not None and b is not None)
            series.append((title, pairs, color))
        self.canvas.set_data(series, x, y, reset=reset)
        count = sum(len(points) for _, points, _ in series)
        imported = self.source.currentData() in self.imports
        provenance = "Imported CSV · session only" if imported else "Run metrics"
        self.caption.setText(f"{provenance} · {count:,} points" + (" · " + "; ".join(warnings) if warnings else ""))

    def show_run(self, run_id):
        index = self.source.findData(run_id)
        if index >= 0:
            self.source.setCurrentIndex(index)

    def open_path(self, path):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            data = read_metrics_csv(path)
        else:
            data = QPixmap(str(path))
            if data.isNull():
                raise ValueError("Could not read that image. Choose a PNG, JPEG, or BMP file.")
        key = "file:" + str(path.resolve())
        self.imports[key] = path.name, data
        self.populate_sources()
        self.source.setCurrentIndex(self.source.findData(key))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open plot or metrics", "",
                                             "Plots and metrics (*.csv *.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                self.open_path(path)
            except (OSError, ValueError, csv.Error) as exc:
                QMessageBox.warning(self, "Could not open plot", str(exc))

    def save_png(self, path):
        canvas = self.current_canvas()
        target = canvas.viewport() if isinstance(canvas, ImageCanvas) else canvas
        if not target.grab().save(str(path), "PNG"):
            raise OSError("The image could not be saved. Check the destination folder.")

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export current plot view", "thetaide-plot.png", "PNG (*.png)")
        if path:
            try:
                self.save_png(path)
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))
