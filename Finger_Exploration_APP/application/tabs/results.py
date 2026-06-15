import csv
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QMovie, QColor
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QListWidget, QComboBox, QSplitter, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QDialog,
    QGraphicsView, QGraphicsScene, QScrollArea,
)


# ─── Clickable image label (click → MP4 prompt; double-click → zoom dialog) ──
class ClickableImageLabel(QLabel):
    clicked        = Signal()
    double_clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


# ─── Zoomable graphics view (scroll-wheel zoom + drag-to-pan) ────────────────
class ZoomableGraphicsView(QGraphicsView):
    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)


# ─── Zoom + Annotate dialog ───────────────────────────────────────────────────
class PlotDetailDialog(QDialog):
    """Opens a plot in a zoomable/pannable view with a notes editor."""

    def __init__(self, image_path, notes_path, session_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plot Detail  —  {session_name}")
        self.setMinimumSize(840, 640)
        self.notes_path  = notes_path
        self._image_path = image_path

        # ── Graphics view ─────────────────────────────────────────────────────
        self.scene = QGraphicsScene()
        self.view  = ZoomableGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(QColor("#0a0a12"))
        self.view.setStyleSheet("border: 1px solid #252538; border-radius: 6px;")

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            item = self.scene.addPixmap(pixmap)
            self.view.fitInView(item, Qt.KeepAspectRatio)
        else:
            self.scene.addText("Could not load image", self.font())

        # ── Notes editor ─────────────────────────────────────────────────────
        notes_header = QLabel("OBSERVATIONS & FINDINGS")
        notes_header.setObjectName("lbl_subheader")

        self.notes_edit = QTextEdit()
        self.notes_edit.setObjectName("json_view")
        self.notes_edit.setMinimumHeight(90)
        self.notes_edit.setMaximumHeight(140)
        self.notes_edit.setPlaceholderText(
            "Record interesting findings, anomalies, or observations here…"
        )

        # Load any existing saved notes
        if os.path.exists(notes_path):
            try:
                with open(notes_path, "r", encoding="utf-8") as f:
                    self.notes_edit.setText(json.load(f).get("notes", ""))
            except Exception:
                pass

        # ── Buttons ───────────────────────────────────────────────────────────
        hint_lbl = QLabel("Scroll to zoom  ·  Drag to pan  ·  Double-click to reset zoom")
        hint_lbl.setObjectName("lbl_muted")

        save_btn  = QPushButton("Save Notes")
        reset_btn = QPushButton("Reset Zoom")
        reset_btn.setObjectName("btn_secondary")
        close_btn = QPushButton("Close")
        close_btn.setObjectName("btn_secondary")

        save_btn.clicked.connect(self._save_notes)
        reset_btn.clicked.connect(self._reset_zoom)
        close_btn.clicked.connect(self.close)

        btn_row = QHBoxLayout()
        btn_row.addWidget(hint_lbl)
        btn_row.addStretch()
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.view, 1)
        layout.addWidget(notes_header)
        layout.addWidget(self.notes_edit)
        layout.addLayout(btn_row)

    def mouseDoubleClickEvent(self, event):
        self._reset_zoom()
        super().mouseDoubleClickEvent(event)

    def _reset_zoom(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _save_notes(self):
        data = {
            "notes":    self.notes_edit.toPlainText(),
            "plot":     os.path.basename(self._image_path),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(self.notes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(self, "Saved", f"Notes saved:\n{self.notes_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))


# ─── Layout helpers ───────────────────────────────────────────────────────────
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


def _field_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("lbl_field")
    return lbl


class MetricCard(QFrame):
    def __init__(self, title, value="—", accent="#0ea5e9"):
        super().__init__()
        self.setObjectName("metric_card")
        self.setStyleSheet(f"QFrame#metric_card {{ border-left: 3px solid {accent}; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)

        self.value_lbl = QLabel(value)
        self.value_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {accent}; background: transparent;"
        )
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet("font-size: 11px; color: #606080; background: transparent;")

        layout.addWidget(self.value_lbl)
        layout.addWidget(self.title_lbl)

    def set_value(self, val):
        self.value_lbl.setText(str(val))


# ─── Main Results Tab ─────────────────────────────────────────────────────────
class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.run_folder      = None
        self._session_data   = None
        self._current_movie  = None   # keep QMovie alive while GIF is playing
        self._current_plot_path = None

        # ── Toolbar ───────────────────────────────────────────────────────────
        self.load_run_button = QPushButton("Load Output Folder")
        self.export_pdf_btn  = QPushButton("Export PDF Report")
        self.export_csv_btn  = QPushButton("Download CSV Data")
        self.export_pdf_btn.setObjectName("btn_secondary")
        self.export_csv_btn.setObjectName("btn_secondary")
        self.export_pdf_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)

        # ── Session list + visualization controls (left panel) ────────────────
        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(140)

        self.plot_dropdown = QComboBox()
        self.plot_dropdown.addItems([
            "cumulative_trajectory.png",
            "cumulative_heatmap.png",
            "cumulative_direction.png",
            "cumulative_transition_graph.png",
            "cumulative_spatial_transition_graph.png",
        ])

        self.load_json_button = QPushButton("Load Session Summary")
        self.load_json_button.setObjectName("btn_secondary")

        # ── Plot viewer ───────────────────────────────────────────────────────
        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(280)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setObjectName("lbl_placeholder")
        self.image_label.setText(
            "No plot loaded\n\n"
            "Select a session — the active plot type loads automatically.\n"
            "Double-click the plot to open the Zoom & Annotate view."
        )

        self.annotate_btn = QPushButton("Zoom & Annotate")
        self.annotate_btn.setObjectName("btn_secondary")
        self.annotate_btn.setEnabled(False)

        self.open_player_btn = QPushButton("Open in System Player")
        self.open_player_btn.setObjectName("btn_secondary")
        self.open_player_btn.setEnabled(False)
        self.open_player_btn.setVisible(False)

        # ── KPI metric cards ──────────────────────────────────────────────────
        self.kpi_steps  = MetricCard("Videos Processed",          accent="#0ea5e9")
        self.kpi_unique = MetricCard("Total Exploration (s)",      accent="#22c55e")
        self.kpi_most   = MetricCard("Avg Per Video (s)",          accent="#f59e0b")
        self.kpi_trans  = MetricCard("Total Region Transitions",   accent="#a855f7")

        # ── Region metrics table ──────────────────────────────────────────────
        self.region_metrics_table = QTableWidget(0, 3)
        self.region_metrics_table.setHorizontalHeaderLabels(
            ["Region", "Avg First Visit (s)", "Total Dwell Frames"]
        )
        self.region_metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.region_metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.region_metrics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.region_metrics_table.setAlternatingRowColors(True)
        self.region_metrics_table.verticalHeader().setVisible(False)
        self.region_metrics_table.setMinimumHeight(80)
        self.region_metrics_table.setObjectName("region_table")

        # ── JSON summary viewer ───────────────────────────────────────────────
        self.json_output = QTextEdit()
        self.json_output.setReadOnly(True)
        self.json_output.setObjectName("json_view")
        self.json_output.setPlaceholderText("Session summary will appear here after loading…")
        self.json_output.setMinimumHeight(80)

        # ── Build layout ──────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Toolbar row
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(self.load_run_button)
        toolbar.addStretch()
        toolbar.addWidget(self.export_pdf_btn)
        toolbar.addWidget(self.export_csv_btn)
        root.addLayout(toolbar)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(4)

        # ── Left panel (fixed 240 px) ─────────────────────────────────────────
        left_card, left_layout = _card()

        sess_hdr = QLabel("SESSION")
        sess_hdr.setObjectName("lbl_subheader")
        left_layout.addWidget(sess_hdr)
        sep1 = QFrame(); sep1.setObjectName("divider")
        left_layout.addWidget(sep1)
        left_layout.addWidget(self.video_list, 1)

        left_layout.addSpacing(8)
        vis_hdr = QLabel("VISUALIZATION")
        vis_hdr.setObjectName("lbl_subheader")
        left_layout.addWidget(vis_hdr)
        sep2 = QFrame(); sep2.setObjectName("divider")
        left_layout.addWidget(sep2)
        left_layout.addWidget(_field_label("Plot Type"))
        left_layout.addWidget(self.plot_dropdown)
        left_layout.addSpacing(6)
        left_layout.addWidget(self.load_json_button)

        left_card.setFixedWidth(240)
        left_card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # ── Right vertical splitter ───────────────────────────────────────────
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(4)

        # Plot viewer card (top half)
        viewer_card, viewer_layout = _card("Plot Viewer")
        viewer_layout.addWidget(self.image_label, 1)
        annotate_row = QHBoxLayout()
        annotate_row.addStretch()
        annotate_row.addWidget(self.open_player_btn)
        annotate_row.addWidget(self.annotate_btn)
        viewer_layout.addLayout(annotate_row)
        right_splitter.addWidget(viewer_card)

        # Session analytics card (bottom half) — wrapped in QScrollArea
        analytics_scroll = QScrollArea()
        analytics_scroll.setWidgetResizable(True)
        analytics_scroll.setFrameShape(QFrame.NoFrame)
        analytics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        analytics_inner = QWidget()
        analytics_layout = QVBoxLayout(analytics_inner)
        analytics_layout.setContentsMargins(16, 16, 16, 16)
        analytics_layout.setSpacing(8)

        analytics_hdr = QLabel("SESSION ANALYTICS")
        analytics_hdr.setObjectName("lbl_subheader")
        analytics_layout.addWidget(analytics_hdr)
        sep3 = QFrame(); sep3.setObjectName("divider")
        analytics_layout.addWidget(sep3)

        kpi_row = QGridLayout()
        kpi_row.setSpacing(8)
        kpi_row.addWidget(self.kpi_steps,  0, 0)
        kpi_row.addWidget(self.kpi_unique, 0, 1)
        kpi_row.addWidget(self.kpi_most,   0, 2)
        kpi_row.addWidget(self.kpi_trans,  0, 3)
        analytics_layout.addLayout(kpi_row)

        region_hdr = QLabel("REGION METRICS")
        region_hdr.setObjectName("lbl_subheader")
        analytics_layout.addWidget(region_hdr)
        analytics_layout.addWidget(self.region_metrics_table)

        json_hdr = QLabel("RAW SESSION SUMMARY")
        json_hdr.setObjectName("lbl_subheader")
        analytics_layout.addWidget(json_hdr)
        analytics_layout.addWidget(self.json_output)
        analytics_layout.addStretch()

        analytics_scroll.setWidget(analytics_inner)

        # Wrap in a card-styled outer frame
        analytics_card = QFrame()
        analytics_card.setObjectName("card")
        ac_layout = QVBoxLayout(analytics_card)
        ac_layout.setContentsMargins(0, 0, 0, 0)
        ac_layout.addWidget(analytics_scroll)

        right_splitter.addWidget(analytics_card)
        right_splitter.setStretchFactor(0, 3)   # plot viewer gets more space
        right_splitter.setStretchFactor(1, 2)   # analytics gets decent share

        main_splitter.addWidget(left_card)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        root.addWidget(main_splitter, 1)

        # ── Connections ───────────────────────────────────────────────────────
        self.load_run_button.clicked.connect(self.load_run_folder)
        self.load_json_button.clicked.connect(self.load_summary_json)
        self.export_pdf_btn.clicked.connect(self._export_pdf)
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.annotate_btn.clicked.connect(self._open_detail_dialog)
        self.open_player_btn.clicked.connect(self._open_in_system_player)
        self.image_label.double_clicked.connect(self._open_detail_dialog)
        self.video_list.currentItemChanged.connect(self._on_session_changed)
        self.plot_dropdown.currentIndexChanged.connect(self._on_plot_type_changed)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_run_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return

        self._session_data  = None
        self._current_movie = None
        self.video_list.clear()
        self._reset_analytics()

        MARKER = "cohort_summary.json"

        # Case A — selected folder is itself a cohort session
        if os.path.exists(os.path.join(folder, MARKER)):
            self.run_folder = os.path.dirname(folder)
            self.video_list.addItem(os.path.basename(folder))
            self.export_pdf_btn.setEnabled(True)
            self.export_csv_btn.setEnabled(True)
            return

        # Case B — selected folder is a parent containing cohort sub-folders
        self.run_folder = folder
        sessions = [
            item for item in sorted(os.listdir(folder))
            if os.path.isdir(os.path.join(folder, item))
            and os.path.exists(os.path.join(folder, item, MARKER))
        ]

        if not sessions:
            QMessageBox.information(
                self, "No Cohort Sessions Found",
                "No valid cohort analysis sessions were found.\n\n"
                "You can select:\n"
                "  • A cohort folder directly (containing cohort_summary.json), or\n"
                "  • A parent folder holding multiple cohort sub-folders.\n\n"
                "Run  'Analyze Video Folder'  pipeline first."
            )
            return

        for s in sessions:
            self.video_list.addItem(s)

        self.export_pdf_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)

    def get_selected_video_folder(self):
        item = self.video_list.currentItem()
        if item is None or self.run_folder is None:
            return None
        return os.path.join(self.run_folder, item.text())

    def load_plot(self):
        folder = self.get_selected_video_folder()
        if folder is None:
            return

        # Stop any running movie first
        if self._current_movie is not None:
            self._current_movie.stop()
            self._current_movie = None

        plot_name = self.plot_dropdown.currentText()
        plot_path = os.path.join(folder, plot_name)
        self._current_plot_path = plot_path

        if not os.path.exists(plot_path):
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(
                f"Plot not available:  {plot_name}\n\n"
                "Run the analysis pipeline to generate this plot,\n"
                "or select a different plot type."
            )
            self.annotate_btn.setEnabled(False)
            self.open_player_btn.setEnabled(False)
            self.open_player_btn.setVisible(False)
            return

        ext = plot_path.lower().rsplit(".", 1)[-1]

        if ext == "mp4":
            # ── MP4 — show video-icon placeholder; click → open prompt ────────
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(
                "▶️\n\n"
                f"{plot_name}\n\n"
                "Tap here or click the button below\n"
                "to open in your system video player."
            )
            self.image_label.setStyleSheet(
                "QLabel { color: #0ea5e9; font-size: 36px; font-weight: 700; }"
            )
            # Single-click on label → same confirm-and-open dialog
            try:
                self.image_label.clicked.disconnect()
            except RuntimeError:
                pass
            self.image_label.clicked.connect(self._open_in_system_player)
            self.annotate_btn.setEnabled(False)
            self.open_player_btn.setEnabled(True)
            self.open_player_btn.setVisible(True)
        elif ext == "gif":
            # ── Animated GIF via QMovie ───────────────────────────────────────
            self.image_label.setStyleSheet("")
            try:
                self.image_label.clicked.disconnect()
            except RuntimeError:
                pass
            movie = QMovie(plot_path)
            self._current_movie = movie

            def _update_frame(_frame_num):
                frame_pix = movie.currentPixmap()
                if not frame_pix.isNull():
                    scaled = frame_pix.scaled(
                        max(self.image_label.width() - 20, 100),
                        max(self.image_label.height() - 20, 100),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self.image_label.setPixmap(scaled)

            movie.frameChanged.connect(_update_frame)
            self.image_label.setText("")
            movie.start()
            self.annotate_btn.setEnabled(False)
            self.open_player_btn.setEnabled(False)
            self.open_player_btn.setVisible(False)
        else:
            # ── Static PNG ───────────────────────────────────────────────────
            self.image_label.setStyleSheet("")
            try:
                self.image_label.clicked.disconnect()
            except RuntimeError:
                pass
            pixmap = QPixmap(plot_path)
            if not pixmap.isNull():
                self.image_label.setPixmap(
                    pixmap.scaled(
                        max(self.image_label.width() - 20, 100),
                        max(self.image_label.height() - 20, 100),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                self.image_label.setText("")
                self.annotate_btn.setEnabled(True)
            self.open_player_btn.setEnabled(False)
            self.open_player_btn.setVisible(False)

    def load_summary_json(self):
        folder = self.get_selected_video_folder()
        if folder is None:
            QMessageBox.warning(self, "No Session Selected",
                                "Please select a session from the list first.")
            return

        summary_path = os.path.join(folder, "cohort_summary.json")
        if not os.path.exists(summary_path):
            QMessageBox.warning(self, "File Not Found",
                                f"Cannot find cohort_summary.json in:\n{folder}")
            return

        with open(summary_path, "r") as f:
            data = json.load(f)

        self._session_data = data
        self.json_output.setText(json.dumps(data, indent=4))
        self._populate_kpi(data)
        self._populate_region_table(data)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _on_session_changed(self, current, _previous):
        if current is not None:
            self._session_data  = None
            self._current_movie = None
            self._reset_analytics()
            self.load_plot()

    def _on_plot_type_changed(self, _index):
        if self.get_selected_video_folder() is not None:
            self.load_plot()

    def _reset_analytics(self):
        self.kpi_steps.set_value("—")
        self.kpi_unique.set_value("—")
        self.kpi_most.set_value("—")
        self.kpi_trans.set_value("—")
        self.region_metrics_table.setRowCount(0)
        self.json_output.clear()
        self.annotate_btn.setEnabled(False)

    def _open_in_system_player(self):
        if not self._current_plot_path or not os.path.exists(self._current_plot_path):
            return
        fname = os.path.basename(self._current_plot_path)
        reply = QMessageBox.question(
            self,
            "Open Video",
            f"Open  {fname}  in your system video player?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            try:
                os.startfile(self._current_plot_path)
            except Exception as exc:
                QMessageBox.critical(self, "Cannot Open", str(exc))

    def _open_detail_dialog(self):
        folder = self.get_selected_video_folder()
        if folder is None:
            return

        plot_name = self.plot_dropdown.currentText()
        ext = plot_name.lower().rsplit(".", 1)[-1]
        if ext in ("gif", "mp4"):
            QMessageBox.information(
                self, "Not Available",
                "Zoom & Annotate works only with static plot images (PNG).\n"
                "Please select trajectory.png, heatmap.png, etc."
            )
            return

        plot_path = os.path.join(folder, plot_name)
        if not os.path.exists(plot_path):
            QMessageBox.warning(self, "Plot Not Found", f"Cannot find:\n{plot_path}")
            return

        notes_path   = os.path.join(folder, plot_name.replace(".png", "_notes.json"))
        session_name = os.path.basename(folder)
        dialog = PlotDetailDialog(plot_path, notes_path, session_name, self)
        dialog.exec()

    def _populate_kpi(self, data):
        self.kpi_steps.set_value(data.get("total_videos_processed", "—"))
        self.kpi_unique.set_value(data.get("total_exploration_time_s", "—"))
        self.kpi_most.set_value(data.get("avg_exploration_time_per_video_s", "—"))
        self.kpi_trans.set_value(data.get("total_transitions", "—"))

    def _populate_region_table(self, data):
        self.region_metrics_table.setRowCount(0)
        avg_fv    = data.get("region_avg_first_visit_s", {})
        dwell     = data.get("region_dwell_frames", {})
        all_rgns  = sorted(set(avg_fv) | set(dwell))
        for region in all_rgns:
            row = self.region_metrics_table.rowCount()
            self.region_metrics_table.insertRow(row)
            values = [
                region,
                str(avg_fv.get(region, "—")),
                str(dwell.get(region, "—")),
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.region_metrics_table.setItem(row, col, item)

    # ── CSV Export ────────────────────────────────────────────────────────────

    def _export_csv(self):
        folder = self.get_selected_video_folder()
        if folder is None:
            QMessageBox.warning(self, "No Session Selected",
                                "Please select a session from the list first.")
            return

        summary_path = os.path.join(folder, "cohort_summary.json")
        if not os.path.exists(summary_path):
            QMessageBox.warning(self, "File Not Found",
                                "Load a cohort session summary before exporting CSV.")
            return

        if self._session_data is None:
            with open(summary_path, "r") as f:
                self._session_data = json.load(f)

        data = self._session_data
        default_name = os.path.basename(folder) + "_cohort_data.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", default_name, "CSV Files (*.csv)"
        )
        if not save_path:
            return

        try:
            avg_fv = data.get("region_avg_first_visit_s", {})
            dwell  = data.get("region_dwell_frames", {})
            rows   = [
                {
                    "Region":             r,
                    "Avg_First_Visit_s":  avg_fv.get(r, ""),
                    "Total_Dwell_Frames": dwell.get(r, ""),
                }
                for r in sorted(set(avg_fv) | set(dwell))
            ]

            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["Region", "Avg_First_Visit_s", "Total_Dwell_Frames"])
                writer.writeheader()
                writer.writerows(rows)

            QMessageBox.information(self, "Export Complete",
                f"CSV saved:\n{save_path}\n\n{len(rows)} region rows written.")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    # ── PDF Export ────────────────────────────────────────────────────────────

    def _export_pdf(self):
        folder = self.get_selected_video_folder()
        if folder is None:
            QMessageBox.warning(self, "No Session Selected",
                                "Please select a session from the list first.")
            return

        session_name = os.path.basename(folder)
        default_name = session_name + "_report.pdf"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", default_name, "PDF Files (*.pdf)"
        )
        if not save_path:
            return

        try:
            kpi_data = [
                ("Videos Processed",       self.kpi_steps.value_lbl.text()),
                ("Total Exploration (s)",  self.kpi_unique.value_lbl.text()),
                ("Avg Per Video (s)",      self.kpi_most.value_lbl.text()),
                ("Total Transitions",      self.kpi_trans.value_lbl.text()),
            ]
            kpi_colors = ["#0ea5e9", "#22c55e", "#f59e0b", "#a855f7"]
            traj_path  = os.path.join(folder, "cumulative_trajectory.png")

            with PdfPages(save_path) as pdf:
                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")

                fig.text(0.5, 0.966, "Tactile Finger Exploration — Session Report",
                         ha="center", va="top", fontsize=16,
                         fontweight="bold", color="#1a1a2e")
                fig.text(0.5, 0.935, f"Session:  {session_name}",
                         ha="center", va="top", fontsize=10, color="#404060")
                fig.text(0.5, 0.910,
                         "Cohort Analysis  |  TGFPAS Research Edition",
                         ha="center", va="top", fontsize=8,
                         color="#9090a8", style="italic")

                rule = Line2D([0.05, 0.95], [0.900, 0.900],
                              transform=fig.transFigure,
                              color="#0ea5e9", linewidth=1.5)
                fig.add_artist(rule)

                gs = gridspec.GridSpec(2, 2, left=0.08, right=0.92,
                                       top=0.890, bottom=0.715,
                                       hspace=0.45, wspace=0.35)
                for idx, ((label, value), color) in enumerate(zip(kpi_data, kpi_colors)):
                    ax = fig.add_subplot(gs[idx // 2, idx % 2])
                    ax.set_facecolor("#f4f6ff")
                    ax.text(0.5, 0.62, value, ha="center", va="center",
                            fontsize=22, fontweight="bold", color=color,
                            transform=ax.transAxes)
                    ax.text(0.5, 0.24, label, ha="center", va="center",
                            fontsize=9, color="#606080", transform=ax.transAxes)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(2.0)
                    ax.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)

                if os.path.exists(traj_path):
                    ax_img = fig.add_axes([0.06, 0.04, 0.88, 0.62])
                    ax_img.imshow(plt.imread(traj_path))
                    ax_img.axis("off")
                    ax_img.set_title("Cumulative Scan-Path Trajectory", fontsize=12,
                                     fontweight="bold", color="#1a1a2e", pad=8)
                else:
                    fig.text(0.5, 0.38,
                             "(cumulative_trajectory.png not found — run pipeline first)",
                             ha="center", va="center", fontsize=10, color="#aaaacc")

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            QMessageBox.information(self, "Export Complete",
                f"PDF report saved:\n{save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
