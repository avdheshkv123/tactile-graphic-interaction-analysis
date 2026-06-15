import sys
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QFrame, QProgressBar, QFileDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor

from core.video_analysis import run_batch_video_analysis


# ─── Worker Thread ────────────────────────────────────────────────────────────
class AnalysisWorker(QThread):
    log_line      = Signal(str)
    progress      = Signal(int, str)          # percent, human-readable status
    analysis_done = Signal(list, str)         # video_names, run_info
    analysis_err  = Signal(str)

    _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, video_folder, model_path, region_json_path,
                 tactile_image_path=None):
        super().__init__()
        self.video_folder       = video_folder
        self.model_path         = model_path
        self.region_json_path   = region_json_path
        self.tactile_image_path = tactile_image_path
        self._orig_stdout       = None

    def write(self, text):
        if text and text.strip():
            self.log_line.emit(text.rstrip("\n"))

    def flush(self):
        pass

    def run(self):
        self._orig_stdout = sys.stdout
        sys.stdout = self
        try:
            videos = sorted(
                f.name for f in Path(self.video_folder).iterdir()
                if f.suffix.lower() in self._VIDEO_EXTS
            )
            self.progress.emit(5, "Step 1/3: Initializing detection models…")
            self.log_line.emit(f"Found {len(videos)} video(s) to process.")

            self.progress.emit(15, "Step 2/3: Running video analysis pipeline…")
            run_batch_video_analysis(
                self.video_folder,
                self.model_path,
                self.region_json_path,
                tactile_image_path=self.tactile_image_path,
            )

            self.progress.emit(100, "Step 3/3: Analysis complete.")
            self.analysis_done.emit(videos, self.video_folder)

        except Exception as exc:
            self.analysis_err.emit(str(exc))
        finally:
            sys.stdout = self._orig_stdout


# ─── Clickable Drop-Zone Frame ────────────────────────────────────────────────
class DropZoneFrame(QFrame):
    """A styled dashed-border frame that emits clicked() on mouse press."""
    clicked = Signal()

    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName("drop_zone")
        self.setMinimumHeight(88)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignCenter)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #8080c0; background: transparent;"
            " letter-spacing: 0.5px;"
        )

        self.subtitle_lbl = QLabel(subtitle)
        self.subtitle_lbl.setAlignment(Qt.AlignCenter)
        self.subtitle_lbl.setWordWrap(True)
        self.subtitle_lbl.setStyleSheet(
            "font-size: 11px; color: #404060; font-style: italic; background: transparent;"
        )

        layout.addWidget(title_lbl)
        layout.addWidget(self.subtitle_lbl)

    def set_selected(self, filename):
        self.subtitle_lbl.setText(filename)
        self.subtitle_lbl.setStyleSheet(
            "font-size: 11px; color: #22c55e; font-weight: 600; background: transparent;"
        )
        self.setStyleSheet(
            "QFrame#drop_zone { background-color: #0d1f18; border: 2px solid #22c55e; border-radius: 8px; }"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─── Layout Helpers ───────────────────────────────────────────────────────────
def _card(title=None):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 18)
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


# ─── Tab ─────────────────────────────────────────────────────────────────────
class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()

        # ── state (preserved from original) ──────────────────────────────────
        self.video_folder       = None
        self.model_path         = None
        self.region_json_path   = None
        self.tactile_image_path = None
        self._worker            = None

        # ── input drop zones ─────────────────────────────────────────────────
        self.video_zone   = DropZoneFrame("Video Folder",  "Click to select folder containing videos")
        self.model_zone   = DropZoneFrame("YOLO Model",    "Click to select .pt model file")
        self.region_zone  = DropZoneFrame("Region Map",    "Click to select regions.json file")
        self.tactile_zone = DropZoneFrame("Tactile Graphic",
                                          "Click to select graphic photo (optional — used as spatial graph background)")

        # ── execution block ───────────────────────────────────────────────────
        self.process_button = QPushButton("Run Full Analysis Pipeline")
        self.process_button.setObjectName("btn_success")
        self.process_button.setMinimumHeight(46)
        self.process_button.setEnabled(False)

        # ── status + progress ─────────────────────────────────────────────────
        self.status_label = QLabel("Waiting for inputs…")
        self.status_label.setObjectName("lbl_field")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")

        # ── console log ───────────────────────────────────────────────────────
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText(
            "Analysis log output will appear here once the pipeline begins…"
        )

        self.clear_button = QPushButton("Clear Log")
        self.clear_button.setObjectName("btn_secondary")

        # ── layout ───────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Row 1: four drop zones side by side
        inputs_card, inputs_layout = _card("Input Configuration")
        zones_row = QHBoxLayout()
        zones_row.setSpacing(10)
        zones_row.addWidget(self.video_zone,   1)
        zones_row.addWidget(self.model_zone,   1)
        zones_row.addWidget(self.region_zone,  1)
        zones_row.addWidget(self.tactile_zone, 1)
        inputs_layout.addLayout(zones_row)
        root.addWidget(inputs_card)

        # Row 2: execution button
        root.addWidget(self.process_button)

        # Row 3: status card (progress + status text)
        status_card, status_layout = _card("Pipeline Status")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        root.addWidget(status_card)

        # Row 4: console log (expands to fill remaining space)
        log_card, log_layout = _card("Analysis Log")
        log_layout.addWidget(self.output_box, 1)
        log_footer = QHBoxLayout()
        log_footer.addStretch()
        log_footer.addWidget(self.clear_button)
        log_layout.addLayout(log_footer)
        root.addWidget(log_card, 1)

        # ── connections ───────────────────────────────────────────────────────
        self.video_zone.clicked.connect(self.select_video_folder)
        self.model_zone.clicked.connect(self.select_model)
        self.region_zone.clicked.connect(self.select_region_json)
        self.tactile_zone.clicked.connect(self.select_tactile_image)
        self.process_button.clicked.connect(self.run_analysis)
        self.clear_button.clicked.connect(self.output_box.clear)

    # ── public API (preserved from original) ──────────────────────────────────
    def update_process_button(self):
        ready = (
            self.video_folder     is not None and
            self.model_path       is not None and
            self.region_json_path is not None
        )
        self.process_button.setEnabled(ready)
        if ready:
            self.status_label.setText("All inputs configured — ready to run.")

    def select_video_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Videos"
        )
        if folder:
            self.video_folder = folder
            self.video_zone.set_selected(os.path.basename(folder))
            self.update_process_button()

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch Models (*.pt)"
        )
        if path:
            self.model_path = path
            self.model_zone.set_selected(os.path.basename(path))
            self.update_process_button()

    def select_region_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Region JSON", "", "JSON Files (*.json)"
        )
        if path:
            self.region_json_path = path
            self.region_zone.set_selected(os.path.basename(path))
            self.update_process_button()

    def select_tactile_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Tactile Graphic Photo", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if path:
            self.tactile_image_path = path
            self.tactile_zone.set_selected(os.path.basename(path))

    def run_analysis(self):
        self.output_box.clear()
        self._log("=" * 58)
        self._log("  ANALYSIS PIPELINE STARTED")
        self._log("=" * 58)
        self._log(f"  Video Folder  :  {self.video_folder}")
        self._log(f"  Model         :  {self.model_path}")
        self._log(f"  Region Map    :  {self.region_json_path}")
        self._log(f"  Tactile Photo :  {self.tactile_image_path or '(none — spatial graph background disabled)'}")
        self._log("")
        self._set_running(True)

        self._worker = AnalysisWorker(
            video_folder=self.video_folder,
            model_path=self.model_path,
            region_json_path=self.region_json_path,
            tactile_image_path=self.tactile_image_path,
        )
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.analysis_done.connect(self._on_done)
        self._worker.analysis_err.connect(self._on_error)
        self._worker.start()

    # ── private helpers ───────────────────────────────────────────────────────
    def _log(self, text):
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.insertPlainText(text + "\n")
        self.output_box.moveCursor(QTextCursor.End)

    def _on_progress(self, pct, message):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def _set_running(self, running):
        self.process_button.setEnabled(not running)
        self.video_zone.setEnabled(not running)
        self.model_zone.setEnabled(not running)
        self.region_zone.setEnabled(not running)
        self.tactile_zone.setEnabled(not running)
        if running:
            self.process_button.setText("Analysis Running…")
            self.progress_bar.setValue(0)
        else:
            self.process_button.setText("Run Full Analysis Pipeline")

    def _on_done(self, videos, run_folder):
        self._log("")
        self._log("=" * 58)
        self._log("  ANALYSIS COMPLETE")
        self._log(f"  Videos Processed  :  {len(videos)}")
        self._log(f"  Output Directory  :  {run_folder}")
        self._log("")
        for name in videos:
            self._log(f"    ✓  {name}")
        self._log("=" * 58)
        self._set_running(False)
        self.status_label.setText(f"Complete — {len(videos)} video(s) processed successfully.")

    def _on_error(self, msg):
        self._log("")
        self._log(f"  ERROR: {msg}")
        self._set_running(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Pipeline failed — see log for details.")
