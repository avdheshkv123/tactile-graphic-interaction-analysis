import json

from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QFrame, QListWidget, QListWidgetItem, QSplitter, QSizePolicy,
)


# ─── Canvas Widget ────────────────────────────────────────────────────────────
class ImageLabel(QLabel):
    """Clickable image canvas that draws rubber-band rects and overlays saved regions."""

    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab  = parent_tab
        self.start_point = None
        self.end_point   = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("canvas_area")

    # ── mouse events (preserved from original) ────────────────────────────────
    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        self.start_point = event.pos()
        self.end_point   = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())
            self.parent_tab.add_region(x1, y1, x2, y2)
            self.start_point = None
            self.end_point   = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.pixmap() is None:
            # Placeholder when no image is loaded
            painter.setPen(QColor("#303050"))
            font = QFont("Segoe UI", 13)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "No Graphic Loaded\n\n"
                "Click  'Load Warped Image'  in the sidebar\n"
                "to upload a reference template and begin marking regions."
            )
            return

        # The pixmap is centered in the label with KeepAspectRatio.
        # Compute the pixel offset of the image's top-left corner inside
        # the label so that regions are drawn on top of the actual image
        # content rather than relative to the full label area.
        pm   = self.pixmap()
        pm_w = pm.width()
        pm_h = pm.height()
        ox   = (self.width()  - pm_w) / 2
        oy   = (self.height() - pm_h) / 2

        # Draw all saved regions as semi-transparent overlays
        for name, coords in self.parent_tab.regions.items():
            x1_n, y1_n, x2_n, y2_n = coords
            rx = int(x1_n * pm_w + ox)
            ry = int(y1_n * pm_h + oy)
            rw = int((x2_n - x1_n) * pm_w)
            rh = int((y2_n - y1_n) * pm_h)

            painter.setPen(QPen(QColor("#0ea5e9"), 2))
            painter.setBrush(QBrush(QColor(14, 165, 233, 40)))
            painter.drawRect(QRect(QPoint(rx, ry), QPoint(rx + rw, ry + rh)))

            # Region name label
            painter.setPen(QColor("#0ea5e9"))
            lbl_font = QFont("Segoe UI", 10, QFont.Bold)
            painter.setFont(lbl_font)
            painter.drawText(rx + 4, ry + 16, name)

        # Draw active rubber-band rectangle
        if self.start_point and self.end_point:
            painter.setPen(QPen(QColor("#ef4444"), 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(239, 68, 68, 25)))
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect.normalized())


# ─── Layout Helpers ───────────────────────────────────────────────────────────
def _card(title=None):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(8)
    if title:
        lbl = QLabel(title.upper())
        lbl.setObjectName("lbl_subheader")
        layout.addWidget(lbl)
        sep = QFrame()
        sep.setObjectName("divider")
        layout.addWidget(sep)
        layout.addSpacing(4)
    return frame, layout


def _step_label(number, text, state="pending"):
    """state: 'pending' | 'active' | 'done'"""
    obj_map = {"pending": "lbl_step_pending", "active": "lbl_step_active", "done": "lbl_step_done"}
    prefix  = {"pending": "○", "active": "►", "done": "✓"}
    lbl = QLabel(f"  {prefix[state]}  Step {number}:  {text}")
    lbl.setObjectName(obj_map[state])
    return lbl


# ─── Tab ─────────────────────────────────────────────────────────────────────
class RegionTab(QWidget):
    def __init__(self):
        super().__init__()

        # ── state (preserved from original) ──────────────────────────────────
        self.image_path     = None
        self.regions        = {}
        self.region_counter = 0

        # ── widgets ──────────────────────────────────────────────────────────
        self.image_label = ImageLabel(self)

        self.load_button = QPushButton("Load Warped Image")
        self.save_button = QPushButton("Save Regions JSON")
        self.save_button.setObjectName("btn_success")

        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.setObjectName("btn_danger")

        self.clear_button = QPushButton("Clear All Regions")
        self.clear_button.setObjectName("btn_secondary")

        self.region_list = QListWidget()
        self.region_list.setMinimumHeight(120)

        # Step indicator labels (updated when user progresses)
        self.step1_label = _step_label(1, "Load Reference Image", "active")
        self.step2_label = _step_label(2, "Draw Regions of Interest", "pending")
        self.step3_label = _step_label(3, "Save Region Map  (JSON)", "pending")

        # ── layout ───────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # ── Left sidebar ──────────────────────────────────────────────────────
        sidebar, sidebar_layout = _card()

        steps_header = QLabel("WORKFLOW")
        steps_header.setObjectName("lbl_subheader")
        sidebar_layout.addWidget(steps_header)
        sep = QFrame()
        sep.setObjectName("divider")
        sidebar_layout.addWidget(sep)
        sidebar_layout.addSpacing(6)

        sidebar_layout.addWidget(self.step1_label)
        sidebar_layout.addWidget(self.step2_label)
        sidebar_layout.addWidget(self.step3_label)

        sidebar_layout.addSpacing(14)
        sidebar_layout.addWidget(self.load_button)
        sidebar_layout.addWidget(self.save_button)

        sidebar_layout.addSpacing(14)

        regions_header = QLabel("MAPPED REGIONS")
        regions_header.setObjectName("lbl_subheader")
        sidebar_layout.addWidget(regions_header)
        sep2 = QFrame()
        sep2.setObjectName("divider")
        sidebar_layout.addWidget(sep2)

        sidebar_layout.addWidget(self.region_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self.delete_button)
        btn_row.addWidget(self.clear_button)
        sidebar_layout.addLayout(btn_row)

        sidebar.setFixedWidth(248)
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # ── Right: canvas area ────────────────────────────────────────────────
        canvas_card, canvas_layout = _card()
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        canvas_layout.addWidget(self.image_label, 1)

        splitter.addWidget(sidebar)
        splitter.addWidget(canvas_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(0)
        root.addWidget(splitter, 1)

        # ── connections ───────────────────────────────────────────────────────
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_regions)
        self.delete_button.clicked.connect(self._delete_selected)
        self.clear_button.clicked.connect(self._clear_all)

    # ── public API (preserved from original) ──────────────────────────────────
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if path:
            self.image_path     = path
            self.regions        = {}
            self.region_counter = 0
            self.region_list.clear()

            pixmap = QPixmap(path)
            self.image_label.setPixmap(
                pixmap.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self._update_steps(image_loaded=True)

    def add_region(self, x1, y1, x2, y2):
        name = chr(ord("A") + self.region_counter)
        self.region_counter += 1

        pm = self.image_label.pixmap()
        if pm is None:
            return
        pm_w = pm.width()
        pm_h = pm.height()
        # Pixel offset of the image's top-left corner inside the centred label
        ox = (self.image_label.width()  - pm_w) / 2
        oy = (self.image_label.height() - pm_h) / 2

        # Normalise relative to the actual image content, not the full label.
        # This ensures saved coordinates match the [0,1]² space used by the
        # video analysis pipeline and spatial transition graph.
        region = [
            round(max(0.0, min(1.0, (x1 - ox) / pm_w)), 4),
            round(max(0.0, min(1.0, (y1 - oy) / pm_h)), 4),
            round(max(0.0, min(1.0, (x2 - ox) / pm_w)), 4),
            round(max(0.0, min(1.0, (y2 - oy) / pm_h)), 4),
        ]
        self.regions[name] = region

        item = QListWidgetItem(
            f"  {name}   [{region[0]:.3f}, {region[1]:.3f}, {region[2]:.3f}, {region[3]:.3f}]"
        )
        item.setData(Qt.UserRole, name)
        self.region_list.addItem(item)
        self.image_label.update()
        self._update_steps(image_loaded=True, has_regions=True)

    def save_regions(self):
        if not self.regions:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Regions JSON", "regions.json", "JSON Files (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.regions, f, indent=4)
            self._update_steps(image_loaded=True, has_regions=True, saved=True)

    # ── private helpers ───────────────────────────────────────────────────────
    def _delete_selected(self):
        item = self.region_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        self.regions.pop(name, None)
        self.region_list.takeItem(self.region_list.row(item))
        self.image_label.update()

    def _clear_all(self):
        self.regions        = {}
        self.region_counter = 0
        self.region_list.clear()
        self.image_label.update()
        self._update_steps(image_loaded=self.image_path is not None)

    def _update_steps(self, image_loaded=False, has_regions=False, saved=False):
        def _refresh(lbl, state):
            lbl.setObjectName(
                "lbl_step_done"    if state == "done"    else
                "lbl_step_active"  if state == "active"  else
                "lbl_step_pending"
            )
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

        _refresh(self.step1_label, "done"    if image_loaded else "active")
        _refresh(self.step2_label, "done"    if has_regions  else
                                   "active"  if image_loaded else "pending")
        _refresh(self.step3_label, "done"    if saved        else
                                   "active"  if has_regions  else "pending")
