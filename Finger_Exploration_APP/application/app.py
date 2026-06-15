import sys

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QFrame,
)
from tabs.analysis import AnalysisTab
from tabs.region import RegionTab
from tabs.results import ResultsTab
from tabs.training import TrainingTab

# ─── Global Stylesheet ────────────────────────────────────────────────────────
GLOBAL_STYLESHEET = """
/* ── Base ── */
QWidget {
    background-color: #0f0f17;
    color: #e8e8f0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
    border: none;
    outline: none;
}

/* ── Scrollbars ── */
QScrollBar:vertical {
    background: #1a1a28;
    width: 8px;
    border-radius: 4px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3a58;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #5555a0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #1a1a28;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #3a3a58;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Tab Widget ── */
QTabWidget::pane {
    border: 1px solid #252538;
    border-top: none;
    background: #131320;
    border-radius: 0 0 8px 8px;
}
QTabBar { background: transparent; }
QTabBar::tab {
    background: #1a1a28;
    color: #7070a0;
    padding: 11px 24px;
    border: 1px solid #252538;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 3px;
    font-weight: 500;
    font-size: 13px;
    min-width: 160px;
}
QTabBar::tab:selected {
    background: #131320;
    color: #0ea5e9;
    border-bottom: 2px solid #0ea5e9;
    font-weight: 700;
}
QTabBar::tab:hover:!selected {
    background: #22223a;
    color: #b0b0cc;
}

/* ── Buttons ── */
QPushButton {
    background-color: #0ea5e9;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
    min-height: 34px;
}
QPushButton:hover { background-color: #38bdf8; }
QPushButton:pressed { background-color: #0284c7; }
QPushButton:disabled { background-color: #252538; color: #4a4a68; }

QPushButton#btn_secondary {
    background-color: #252538;
    color: #c0c0e0;
    border: 1px solid #353555;
}
QPushButton#btn_secondary:hover {
    background-color: #303050;
    color: #e0e0f8;
    border-color: #5555aa;
}
QPushButton#btn_secondary:pressed { background-color: #1e1e35; }
QPushButton#btn_secondary:disabled {
    background-color: #1a1a28;
    color: #3a3a58;
    border-color: #252538;
}

QPushButton#btn_danger { background-color: #ef4444; color: #fff; }
QPushButton#btn_danger:hover { background-color: #f87171; }
QPushButton#btn_danger:pressed { background-color: #dc2626; }

QPushButton#btn_success { background-color: #22c55e; color: #fff; }
QPushButton#btn_success:hover { background-color: #4ade80; }
QPushButton#btn_success:pressed { background-color: #16a34a; }
QPushButton#btn_success:disabled { background-color: #1a3028; color: #2a5040; }

/* ── Inputs ── */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #1e1e30;
    color: #e8e8f0;
    border: 1px solid #30305a;
    border-radius: 5px;
    padding: 6px 10px;
    min-height: 32px;
    selection-background-color: #0ea5e9;
    selection-color: #fff;
}
QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: #5555aa; }
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #0ea5e9;
    background-color: #1a1a2a;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #252540;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #353558;
}

/* ── ComboBox ── */
QComboBox {
    background-color: #1e1e30;
    color: #e8e8f0;
    border: 1px solid #30305a;
    border-radius: 5px;
    padding: 6px 10px;
    min-height: 32px;
}
QComboBox:hover { border-color: #5555aa; }
QComboBox:focus { border-color: #0ea5e9; background-color: #1a1a2a; }
QComboBox::drop-down {
    background-color: #252540;
    border: none;
    border-radius: 0 5px 5px 0;
    width: 26px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e30;
    color: #e8e8f0;
    border: 1px solid #30305a;
    border-radius: 5px;
    selection-background-color: #0ea5e9;
    selection-color: #fff;
    outline: none;
    padding: 2px;
}
QComboBox QAbstractItemView::item { padding: 6px 10px; border-radius: 3px; }

/* ── TextEdit (console) ── */
QTextEdit {
    background-color: #080810;
    color: #e8c46a;
    border: 1px solid #1e1e35;
    border-radius: 6px;
    font-family: "Consolas", "Fira Code", "Courier New", monospace;
    font-size: 12px;
    padding: 10px;
    selection-background-color: #2a3a55;
    selection-color: #e8e8f0;
}
QTextEdit#json_view {
    color: #88d498;
    background-color: #0a0f10;
}

/* ── Labels ── */
QLabel { background: transparent; }
QLabel#lbl_header {
    font-size: 15px;
    font-weight: 700;
    color: #f0f0ff;
    letter-spacing: 0.5px;
}
QLabel#lbl_subheader {
    font-size: 11px;
    font-weight: 700;
    color: #7070a0;
    letter-spacing: 1.5px;
}
QLabel#lbl_field {
    font-size: 12px;
    font-weight: 500;
    color: #9090b8;
}
QLabel#lbl_muted {
    font-size: 11px;
    color: #50506a;
}
QLabel#lbl_value {
    font-size: 26px;
    font-weight: 700;
    color: #0ea5e9;
}
QLabel#lbl_path {
    font-size: 11px;
    color: #6060a0;
    font-style: italic;
    background-color: #1a1a28;
    border: 1px solid #252538;
    border-radius: 4px;
    padding: 5px 8px;
}
QLabel#lbl_placeholder {
    color: #303050;
    font-size: 14px;
}
QLabel#lbl_step_pending { color: #303050; font-size: 13px; }
QLabel#lbl_step_active  { color: #0ea5e9; font-weight: 600; font-size: 13px; }
QLabel#lbl_step_done    { color: #22c55e; font-weight: 600; font-size: 13px; }

/* ── Frames / Cards ── */
QFrame#card {
    background-color: #1a1a28;
    border: 1px solid #252538;
    border-radius: 8px;
}
QFrame#card_elevated {
    background-color: #1e1e30;
    border: 1px solid #2e2e50;
    border-radius: 8px;
}
QFrame#drop_zone {
    background-color: #131320;
    border: 2px dashed #2e2e50;
    border-radius: 8px;
}
QFrame#drop_zone:hover {
    border-color: #0ea5e9;
    background-color: #111120;
}
QFrame#divider { background-color: #252538; max-height: 1px; min-height: 1px; }
QFrame#metric_card {
    background-color: #1a1a28;
    border: 1px solid #252538;
    border-left: 3px solid #0ea5e9;
    border-radius: 6px;
}

/* ── List Widget ── */
QListWidget {
    background-color: #1a1a28;
    color: #d8d8f0;
    border: 1px solid #252538;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    margin: 1px 2px;
}
QListWidget::item:selected { background-color: #0ea5e9; color: #fff; }
QListWidget::item:hover:!selected { background-color: #252538; }

/* ── Progress Bar ── */
QProgressBar {
    background-color: #1a1a28;
    border: 1px solid #252538;
    border-radius: 6px;
    color: #e8e8f0;
    font-size: 11px;
    font-weight: 600;
    text-align: center;
    max-height: 22px;
    min-height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #0284c7, stop: 1 #0ea5e9
    );
    border-radius: 5px;
}

/* ── Splitter ── */
QSplitter::handle { background: #252538; }
QSplitter::handle:hover { background: #353558; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }

/* ── Table Widget ── */
QTableWidget {
    background-color: #1a1a28;
    color: #d8d8f0;
    border: 1px solid #252538;
    border-radius: 6px;
    gridline-color: #252538;
    outline: none;
}
QTableWidget::item {
    padding: 5px 10px;
}
QTableWidget::item:selected {
    background-color: #0ea5e9;
    color: #fff;
}
QTableWidget::item:alternate {
    background-color: #1e1e30;
}
QHeaderView::section {
    background-color: #252540;
    color: #9090b8;
    padding: 6px 10px;
    border: none;
    border-right: 1px solid #303055;
    border-bottom: 1px solid #303055;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
QHeaderView::section:last {
    border-right: none;
}
QTableCornerButton::section {
    background-color: #252540;
    border: none;
}
"""


# ─── Main Window ─────────────────────────────────────────────────────────────
class FingerPatternApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tactile Graphics Finger Pattern Analysis System")
        self.resize(1440, 900) # Initial size, can be resized by user
        self.setMinimumSize(1100, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 12, 16, 16)
        body_layout.setSpacing(0)

        self.tabs = QTabWidget()

        self.training_tab = TrainingTab()
        self.region_tab   = RegionTab()
        self.analysis_tab = AnalysisTab()
        self.results_tab  = ResultsTab()

        self.tabs.addTab(self.training_tab, "  Train the YOLO Model  ")
        self.tabs.addTab(self.region_tab,   "  Mark Salient Regions  ")
        self.tabs.addTab(self.analysis_tab, "  Analyze Video Folder  ")
        self.tabs.addTab(self.results_tab,  "  Results Dashboard  ")

        body_layout.addWidget(self.tabs)
        root.addWidget(body, 1)

    def _build_header(self):
        hdr = QFrame() 
        hdr.setFixedHeight(62)
        hdr.setStyleSheet(
            "QFrame { background-color: #0a0a12;"
            " border-bottom: 2px solid #0ea5e9; }"
        )
        layout = QHBoxLayout(hdr)
        layout.setContentsMargins(22, 0, 22, 0)
        layout.setSpacing(12)

        badge = QLabel("TGFPAS")  # Tactile Graphics Finger Pattern Analysis System
        badge.setStyleSheet(
            "background-color: #0ea5e9; color: #fff;"
            " font-size: 11px; font-weight: 900;"
            " padding: 4px 10px; border-radius: 4px; letter-spacing: 1px;"
        )

        title = QLabel("Tactile Graphics Finger Pattern Analysis System")
        title.setStyleSheet(
            "color: #e8e8f8; font-size: 15px;"
            " font-weight: 700; letter-spacing: 0.3px;"
        )

        layout.addWidget(badge)
        layout.addWidget(title)
        layout.addStretch()

        version = QLabel("v1.0.0  —  Research Edition")
        version.setStyleSheet("color: #35355a; font-size: 11px;")
        layout.addWidget(version)

        return hdr


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET) # Apply the global stylesheet to the entire application

    window = FingerPatternApp() 
    window.show()

    sys.exit(app.exec())
