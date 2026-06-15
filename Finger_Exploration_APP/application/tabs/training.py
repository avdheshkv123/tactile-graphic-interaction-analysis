import sys
import os

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSpinBox, QComboBox, QFrame, QSplitter, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor

from core.training import train_model


# ─── Worker Thread ────────────────────────────────────────────────────────────
class TrainingWorker(QThread):
    log_line      = Signal(str)
    training_done = Signal(str, str, str, int)   # out_folder, model_folder, samples_folder, num_samples
    training_err  = Signal(str)

    def __init__(self, dataset_yaml, base_model, epochs, batch_size):
        super().__init__()
        self.dataset_yaml = dataset_yaml
        self.base_model   = base_model
        self.epochs       = epochs
        self.batch_size   = batch_size
        self._orig_stdout = None

    # redirect stdout → emit log signals (thread-safe via Qt's queued connection)
    def write(self, text):
        if text and text.strip():
            self.log_line.emit(text.rstrip("\n"))

    def flush(self):
        pass

    def run(self):
        self._orig_stdout = sys.stdout
        sys.stdout = self
        try:
            result = train_model(
                dataset_yaml_path=self.dataset_yaml,
                base_model=self.base_model,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
            self.training_done.emit(
                result.get("output_folder", ""),
                result.get("model_folder", ""),
                result.get("samples_folder", ""),
                int(result.get("num_samples_generated", 0)),
            )
        except Exception as exc:
            self.training_err.emit(str(exc))
        finally:
            sys.stdout = self._orig_stdout


# ─── Layout Helpers ───────────────────────────────────────────────────────────
def _card(title=None):
    """Return (QFrame, inner QVBoxLayout) styled as a dark card panel."""
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


def _field_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("lbl_field")
    return lbl


# ─── Tab ─────────────────────────────────────────────────────────────────────
class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()

        # ── state (preserved from original) ──────────────────────────────────
        self.dataset_yaml_path = None
        self._worker           = None

        # ── widgets ──────────────────────────────────────────────────────────
        self.dataset_label = QLabel("No data.yaml selected")
        self.dataset_label.setObjectName("lbl_path")
        self.dataset_label.setWordWrap(True)

        self.dataset_button = QPushButton("Browse  data.yaml")
        self.dataset_button.setObjectName("btn_secondary")

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            "yolo11n-obb.pt",
            "yolo11s-obb.pt",
            "yolo11m-obb.pt",
            "yolo11l-obb.pt",
        ])

        self.epochs_box = QSpinBox()
        self.epochs_box.setRange(1, 1000)
        self.epochs_box.setValue(100)
        self.epochs_box.setSuffix("  epochs")

        self.batch_box = QSpinBox()
        self.batch_box.setRange(1, 128)
        self.batch_box.setValue(8)
        self.batch_box.setSuffix("  images / batch")

        self.train_button = QPushButton("Start Training Process")
        self.train_button.setObjectName("btn_success")
        self.train_button.setMinimumHeight(44)
        self.train_button.setEnabled(False)

        self.status_label = QLabel("Status: Idle — select a dataset to begin")
        self.status_label.setObjectName("lbl_muted")

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText(
            "Training output will stream here once training begins..."
        )

        self.clear_button = QPushButton("Clear Console")
        self.clear_button.setObjectName("btn_secondary")

        # ── layout ───────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # ── Left: parameter card ──────────────────────────────────────────────
        left_card, left_layout = _card("Training Parameters")

        left_layout.addWidget(_field_label("Dataset Configuration File  (data.yaml)"))
        left_layout.addWidget(self.dataset_button)
        left_layout.addWidget(self.dataset_label)

        left_layout.addSpacing(10)
        left_layout.addWidget(_field_label("Base Model Variant"))
        left_layout.addWidget(self.model_dropdown)

        left_layout.addSpacing(10)
        left_layout.addWidget(_field_label("Training Epochs"))
        left_layout.addWidget(self.epochs_box)

        left_layout.addSpacing(10)
        left_layout.addWidget(_field_label("Batch Size"))
        left_layout.addWidget(self.batch_box)

        left_layout.addStretch()

        sep = QFrame()
        sep.setObjectName("divider")
        left_layout.addWidget(sep)
        left_layout.addSpacing(6)

        left_layout.addWidget(self.train_button)
        left_layout.addWidget(self.status_label)

        left_card.setMinimumWidth(270)
        left_card.setMaximumWidth(360)
        left_card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # ── Right: console card ───────────────────────────────────────────────
        right_card, right_layout = _card("Training Console")

        right_layout.addWidget(self.output_box, 1)

        right_footer = QHBoxLayout()
        right_footer.addStretch()
        right_footer.addWidget(self.clear_button)
        right_layout.addLayout(right_footer)

        splitter.addWidget(left_card)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(0)
        root.addWidget(splitter, 1)

        # ── connections ───────────────────────────────────────────────────────
        self.dataset_button.clicked.connect(self.select_dataset_yaml)
        self.train_button.clicked.connect(self.start_training)
        self.clear_button.clicked.connect(self.output_box.clear)

    # ── public API (preserved) ────────────────────────────────────────────────
    def select_dataset_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.dataset_yaml_path = path
            self.dataset_label.setText(os.path.basename(path))
            self.train_button.setEnabled(True)
            self.status_label.setText("Status: Dataset loaded — ready to train")

    def start_training(self):
        self.output_box.clear()
        self._log("=" * 58)
        self._log("  TRAINING PROCESS STARTED")
        self._log("=" * 58)
        self._log(f"  Dataset  :  {self.dataset_yaml_path}")
        self._log(f"  Model    :  {self.model_dropdown.currentText()}")
        self._log(f"  Epochs   :  {self.epochs_box.value()}")
        self._log(f"  Batch    :  {self.batch_box.value()}")
        self._log("")

        self._set_running(True)

        self._worker = TrainingWorker(
            dataset_yaml=self.dataset_yaml_path,
            base_model=self.model_dropdown.currentText(),
            epochs=self.epochs_box.value(),
            batch_size=self.batch_box.value(),
        )
        self._worker.log_line.connect(self._log)
        self._worker.training_done.connect(self._on_done)
        self._worker.training_err.connect(self._on_error)
        self._worker.start()

    # ── private helpers ───────────────────────────────────────────────────────
    def _log(self, text):
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.insertPlainText(text + "\n")
        self.output_box.moveCursor(QTextCursor.End)

    def _set_running(self, running):
        self.train_button.setEnabled(not running)
        self.dataset_button.setEnabled(not running)
        self.epochs_box.setEnabled(not running)
        self.batch_box.setEnabled(not running)
        self.model_dropdown.setEnabled(not running)
        if running:
            self.status_label.setText("Status: Training in progress…")
            self.train_button.setText("Training…")
        else:
            self.train_button.setText("Start Training Process")

    def _on_done(self, output_folder, model_folder, samples_folder, num_samples):
        self._log("")
        self._log("=" * 58)
        self._log("  TRAINING COMPLETE")
        self._log(f"  Output Folder  :  {output_folder}")
        self._log(f"  Model Folder   :  {model_folder}")
        self._log(f"  Samples Folder :  {samples_folder}")
        self._log(f"  Crop Samples   :  {num_samples} images saved")
        self._log("=" * 58)
        self._set_running(False)
        self.status_label.setText("Status: Completed successfully")

    def _on_error(self, msg):
        self._log("")
        self._log(f"  ERROR: {msg}")
        self._set_running(False)
        self.status_label.setText("Status: Failed — see console for details")
