from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from config import AppConfig, load_config, save_config
from exporters import (
    export_json_collective,
    export_json_individual,
    export_pascal_voc,
    export_yolo,
    save_patches,
)


APP_ROOT = Path(__file__).resolve().parent
RESOURCES = APP_ROOT / "resources"


@dataclass
class ImageState:
    original: Optional[np.ndarray] = None
    annotated: Optional[np.ndarray] = None
    image_path: Optional[Path] = None

    @property
    def width(self) -> int:
        return 0 if self.original is None else int(self.original.shape[1])

    @property
    def height(self) -> int:
        return 0 if self.original is None else int(self.original.shape[0])


class ImageCanvas(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
        self._state: Optional[ImageState] = None
        self._brush_color = (255, 182, 56)
        self._brush_size = 20
        self._draw_mode = True
        self._last_point: Optional[QtCore.QPoint] = None

    def set_state(self, state: ImageState) -> None:
        self._state = state
        self._last_point = None
        self._refresh()

    def set_brush(self, color: tuple[int, int, int], size: int, draw_mode: bool) -> None:
        self._brush_color = color
        self._brush_size = size
        self._draw_mode = draw_mode

    def _refresh(self) -> None:
        if not self._state or self._state.annotated is None:
            self.clear()
            return
        pixmap = self._to_pixmap(self._state.annotated)
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def show_temp_image(self, image: np.ndarray) -> None:
        pixmap = self._to_pixmap(image)
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def _to_pixmap(self, image: np.ndarray) -> QtGui.QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(qimage)

    def _image_pos(self, point: QtCore.QPoint) -> Optional[tuple[int, int]]:
        if not self._state or self._state.annotated is None:
            return None
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        label_size = self.size()
        pixmap_size = pixmap.size()
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            return None

        scale = min(
            label_size.width() / pixmap_size.width(),
            label_size.height() / pixmap_size.height(),
        )
        displayed_w = pixmap_size.width() * scale
        displayed_h = pixmap_size.height() * scale
        offset_x = (label_size.width() - displayed_w) / 2
        offset_y = (label_size.height() - displayed_h) / 2

        x = int((point.x() - offset_x) / scale)
        y = int((point.y() - offset_y) / scale)

        if x < 0 or y < 0:
            return None
        if x >= pixmap_size.width() or y >= pixmap_size.height():
            return None
        return x, y

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.modifiers() & QtCore.Qt.AltModifier:
            self._last_point = event.position().toPoint()
        else:
            self._last_point = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._state or self._state.annotated is None:
            return
        if not (event.modifiers() & QtCore.Qt.AltModifier):
            self._last_point = None
            return
        current_point = event.position().toPoint()
        if self._last_point is None:
            self._last_point = current_point
            return

        p1 = self._image_pos(self._last_point)
        p2 = self._image_pos(current_point)
        if p1 is None or p2 is None:
            self._last_point = current_point
            return

        if self._draw_mode:
            cv2.line(
                self._state.annotated,
                p1,
                p2,
                self._brush_color,
                int(self._brush_size),
            )
        else:
            self._erase_at(p2)

        self._last_point = current_point
        self._refresh()

    def _erase_at(self, point: tuple[int, int]) -> None:
        if not self._state or self._state.annotated is None or self._state.original is None:
            return
        x, y = point
        size = int(self._brush_size)
        x1 = max(0, x - size // 2)
        y1 = max(0, y - size // 2)
        x2 = min(self._state.annotated.shape[1], x + size // 2)
        y2 = min(self._state.annotated.shape[0], y + size // 2)
        self._state.annotated[y1:y2, x1:x2] = self._state.original[y1:y2, x1:x2]


class ImageWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Main View")
        self.canvas = ImageCanvas()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

    def set_state(self, state: ImageState) -> None:
        self.canvas.set_state(state)


class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, title: str, image: np.ndarray, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = self._to_pixmap(image)
        label.setPixmap(pixmap)
        layout.addWidget(label)
        self.resize(800, 600)

    def _to_pixmap(self, image: np.ndarray) -> QtGui.QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(qimage)


class DefectDetectApp(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.config_path = APP_ROOT / "config.txt"
        self.cfg = load_config(self.config_path)
        self.default_cfg = AppConfig()
        self.state = ImageState()
        self.masks: List[np.ndarray] = []
        self.patch_coords: List[List[int]] = []
        self.patches: List[List[np.ndarray]] = []
        self.ratings: List[List[int]] = []
        self.file_list: List[Path] = []
        self.current_index = 0
        self._preview_windows: List[QtWidgets.QDialog] = []

        self.main_window = QtWidgets.QWidget()
        self.main_window.setWindowTitle("DefectDetect")
        self.main_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        self.main_layout = QtWidgets.QVBoxLayout(self.main_window)

        logo = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap(str(RESOURCES / "logo.png"))
        logo.setPixmap(logo_pixmap)
        logo.setScaledContents(True)
        self.main_layout.addWidget(logo)

        self.image_window = ImageWindow()
        self.image_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))

        self._build_settings_window()
        self._build_tools_window()
        self._build_main_buttons()
        self._build_end_annotation_window()

    def _build_main_buttons(self) -> None:
        load_image_btn = QtWidgets.QPushButton("Load Image")
        load_multi_btn = QtWidgets.QPushButton("Load Multiple Images")
        load_session_btn = QtWidgets.QPushButton("Load Session")
        settings_btn = QtWidgets.QPushButton("Settings")

        self.main_layout.addWidget(load_image_btn)
        self.main_layout.addWidget(load_multi_btn)
        self.main_layout.addWidget(load_session_btn)
        self.main_layout.addWidget(settings_btn)

        load_image_btn.clicked.connect(lambda: self._on_load_action(1))
        load_multi_btn.clicked.connect(lambda: self._on_load_action(3))
        load_session_btn.clicked.connect(lambda: self._on_load_action(2))
        settings_btn.clicked.connect(self.settings_window.show)

    def _build_tools_window(self) -> None:
        self.tools_window = QtWidgets.QWidget()
        self.tools_window.setWindowTitle("DefectDetect")
        self.tools_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        self.tools_layout = QtWidgets.QVBoxLayout(self.tools_window)

        marker_layout = QtWidgets.QHBoxLayout()
        self.tools_layout.addLayout(marker_layout)

        self.marker_buttons = []
        self._add_marker_button(marker_layout, "1.png", self.cfg.names[0], 1)
        self._add_marker_button(marker_layout, "2.png", self.cfg.names[1], 2)
        self._add_marker_button(marker_layout, "3.png", self.cfg.names[2], 3)
        self._add_marker_button(marker_layout, "4.png", self.cfg.names[3], 4)
        self._add_marker_button(marker_layout, "8.png", self.cfg.names[4], 8)
        self._add_marker_button(marker_layout, "5.png", self.cfg.names[6], 5, text=True)
        self._add_marker_button(marker_layout, "6.png", self.cfg.names[7], 6, text=True)
        self._add_marker_button(marker_layout, "7.png", self.cfg.names[5], 7, text=True)

        thickness_label = QtWidgets.QLabel("Thickness:")
        marker_layout.addWidget(thickness_label)

        for icon, size in [("najtanji.png", 10), ("srednji.png", 20), ("najdeblji.png", 30)]:
            btn = QtWidgets.QToolButton()
            btn.setIcon(QtGui.QIcon(str(RESOURCES / icon)))
            btn.setToolTip(str(size))
            btn.clicked.connect(lambda _=False, s=size: self._set_marker_thickness(s))
            marker_layout.addWidget(btn)

        action_layout = QtWidgets.QHBoxLayout()
        self.tools_layout.addLayout(action_layout)

        remove_btn = QtWidgets.QPushButton("Remove Annotations")
        remove_btn.pressed.connect(self._show_original)
        remove_btn.released.connect(self._show_annotated)
        action_layout.addWidget(remove_btn)

        delete_btn = QtWidgets.QPushButton("Delete Annotations...")
        delete_btn.clicked.connect(self._confirm_delete_annotations)
        action_layout.addWidget(delete_btn)

        end_btn = QtWidgets.QPushButton("End Annotation...")
        end_btn.clicked.connect(self._on_end_annotation)
        action_layout.addWidget(end_btn)

        end_session_btn = QtWidgets.QPushButton("End Session")
        end_session_btn.clicked.connect(self._end_session)
        action_layout.addWidget(end_session_btn)

        navigation_layout = QtWidgets.QHBoxLayout()
        self.tools_layout.addLayout(navigation_layout)

        prev_btn = QtWidgets.QToolButton()
        prev_btn.setText("Previous")
        prev_btn.clicked.connect(self._prev_image)
        navigation_layout.addWidget(prev_btn)

        next_btn = QtWidgets.QToolButton()
        next_btn.setText("Next")
        next_btn.clicked.connect(self._next_image)
        navigation_layout.addWidget(next_btn)

    def _build_settings_window(self) -> None:
        self.settings_window = QtWidgets.QWidget()
        self.settings_window.setWindowTitle("Settings")
        self.settings_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        layout = QtWidgets.QVBoxLayout(self.settings_window)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        tab_names = QtWidgets.QWidget()
        names_layout = QtWidgets.QFormLayout(tab_names)
        self.name_edits = []
        for label in [
            "Marker 1 name:",
            "Marker 2 name:",
            "Marker 3 name:",
            "Marker 4 name:",
            "Marker 5 name:",
            "Eraser name:",
            "Edge name:",
            "Surface name:",
            "Class Correct name:",
            "Class Leather name:",
        ]:
            edit = QtWidgets.QLineEdit()
            self.name_edits.append(edit)
            names_layout.addRow(label, edit)
        tabs.addTab(tab_names, "Names")

        tab_patches = QtWidgets.QWidget()
        patches_layout = QtWidgets.QFormLayout(tab_patches)
        self.patch_width_edit = QtWidgets.QLineEdit()
        self.patch_height_edit = QtWidgets.QLineEdit()
        self.patch_stride_x_edit = QtWidgets.QLineEdit()
        self.patch_stride_y_edit = QtWidgets.QLineEdit()
        validator = QtGui.QDoubleValidator(0, 9999, 2)
        for edit in [
            self.patch_width_edit,
            self.patch_height_edit,
            self.patch_stride_x_edit,
            self.patch_stride_y_edit,
        ]:
            edit.setValidator(validator)
        patches_layout.addRow("Width:", self.patch_width_edit)
        patches_layout.addRow("Height:", self.patch_height_edit)
        patches_layout.addRow("Horizontal stride:", self.patch_stride_x_edit)
        patches_layout.addRow("Vertical stride:", self.patch_stride_y_edit)
        tabs.addTab(tab_patches, "Patches")

        tab_other = QtWidgets.QWidget()
        other_layout = QtWidgets.QFormLayout(tab_other)
        self.thickness_combo = QtWidgets.QComboBox()
        for i in range(10, 75, 5):
            self.thickness_combo.addItem(str(i))
        other_layout.addRow("Marker Thickness:", self.thickness_combo)

        tolerance_label = QtWidgets.QLabel(
            "Setting tolerance to assign rating 1 to patch.\n"
            "This tolerance represents the percentage of the patch\n"
            "covered by the annotation in relation to its surface.\n"
            "It also limits the minimum presence of the annotation\n"
            "for assigning a rating of 1."
        )
        tolerance_label.setWordWrap(True)
        self.tolerance_edit = QtWidgets.QLineEdit()
        tol_validator = QtGui.QIntValidator(1, 100)
        self.tolerance_edit.setValidator(tol_validator)
        other_layout.addRow(tolerance_label)
        other_layout.addRow("Choose ratio (1-100):", self.tolerance_edit)
        tabs.addTab(tab_other, "Other")

        buttons_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons_layout)
        default_btn = QtWidgets.QPushButton("Default")
        save_btn = QtWidgets.QPushButton("Save")
        buttons_layout.addWidget(default_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(save_btn)

        default_btn.clicked.connect(self._reset_defaults)
        save_btn.clicked.connect(self._save_settings)

        self._sync_settings_ui()

    def _build_end_annotation_window(self) -> None:
        self.end_annotation_window = QtWidgets.QWidget()
        self.end_annotation_window.setWindowTitle("End of annotating")
        self.end_annotation_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        layout = QtWidgets.QVBoxLayout(self.end_annotation_window)

        show_masks_btn = QtWidgets.QPushButton("Show Masks")
        show_masks_btn.clicked.connect(self._show_masks)
        layout.addWidget(show_masks_btn)

        create_patches_btn = QtWidgets.QPushButton("Create Patches...")
        create_patches_btn.clicked.connect(self._open_patch_dim_window)
        layout.addWidget(create_patches_btn)

        export_patches_btn = QtWidgets.QPushButton("Export patches...")
        export_patches_btn.clicked.connect(self._open_export_window)
        layout.addWidget(export_patches_btn)

        save_image_btn = QtWidgets.QPushButton("Save image")
        save_image_btn.clicked.connect(self._save_session)
        layout.addWidget(save_image_btn)

        self._build_patch_dim_window()
        self._build_export_window()

    def _build_patch_dim_window(self) -> None:
        self.patch_dim_window = QtWidgets.QDialog(self.end_annotation_window)
        self.patch_dim_window.setWindowTitle("Creating patches")
        self.patch_dim_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        layout = QtWidgets.QVBoxLayout(self.patch_dim_window)
        self.patch_dim_edits: List[QtWidgets.QLineEdit] = []
        labels = ["Width", "Height", "Horizontal stride", "Vertical stride"]
        for idx, label in enumerate(labels):
            layout.addWidget(QtWidgets.QLabel(f"{label}:"))
            edit = QtWidgets.QLineEdit()
            edit.setValidator(QtGui.QIntValidator())
            layout.addWidget(edit)
            self.patch_dim_edits.append(edit)
        self._sync_patch_dim_edits()
        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)
        create_btn = QtWidgets.QPushButton("Create")
        create_btn.clicked.connect(self._create_patches)
        btn_layout.addWidget(create_btn)

    def _build_export_window(self) -> None:
        self.choose_export_window = QtWidgets.QDialog(self.end_annotation_window)
        self.choose_export_window.setWindowTitle("Export type selection")
        self.choose_export_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        layout = QtWidgets.QVBoxLayout(self.choose_export_window)

        export_all_btn = QtWidgets.QPushButton("Export all patches")
        export_selected_btn = QtWidgets.QPushButton("Export selected patches...")
        layout.addWidget(export_all_btn)
        layout.addWidget(export_selected_btn)

        self._export_mask_yes = QtWidgets.QCheckBox("Yes")
        self._export_mask_no = QtWidgets.QCheckBox("No")
        self._export_mask_no.setChecked(True)
        self._pair_checkbox(self._export_mask_yes, self._export_mask_no)
        layout.addWidget(self._wrap_checkbox("Mask export:", self._export_mask_yes, self._export_mask_no))

        self._json_collective = QtWidgets.QCheckBox("Collective")
        self._json_individual = QtWidgets.QCheckBox("Individual")
        self._json_collective.setChecked(True)
        self._pair_checkbox(self._json_collective, self._json_individual)
        layout.addWidget(self._wrap_checkbox("JSON file export:", self._json_collective, self._json_individual))

        self._yolo_yes = QtWidgets.QCheckBox("Yes")
        self._yolo_no = QtWidgets.QCheckBox("No")
        self._yolo_no.setChecked(True)
        self._pair_checkbox(self._yolo_yes, self._yolo_no)
        layout.addWidget(self._wrap_checkbox("YOLO export:", self._yolo_yes, self._yolo_no))

        self._pascal_yes = QtWidgets.QCheckBox("Yes")
        self._pascal_no = QtWidgets.QCheckBox("No")
        self._pascal_no.setChecked(True)
        self._pair_checkbox(self._pascal_yes, self._pascal_no)
        layout.addWidget(self._wrap_checkbox("Pascal VOC export:", self._pascal_yes, self._pascal_no))

        export_all_btn.clicked.connect(self._export_all)
        export_selected_btn.clicked.connect(self._open_grades_window)

        self._build_grades_window()

    def _build_grades_window(self) -> None:
        self.grades_window = QtWidgets.QDialog(self.choose_export_window)
        self.grades_window.setWindowTitle("Exporting patches")
        self.grades_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        layout = QtWidgets.QVBoxLayout(self.grades_window)

        self.rating_checkboxes: List[List[QtWidgets.QCheckBox]] = []
        self.grade_labels: List[QtWidgets.QLabel] = []
        class_labels = [
            self.cfg.names[0],
            self.cfg.names[1],
            self.cfg.names[2],
            self.cfg.names[3],
            self.cfg.names[4],
            self.cfg.names[6],
            self.cfg.names[7],
            self.cfg.names[8],
            self.cfg.names[9],
        ]
        for label in class_labels:
            row = QtWidgets.QHBoxLayout()
            widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QVBoxLayout(widget)
            label_widget = QtWidgets.QLabel(f"Choose ratings for {label}:")
            self.grade_labels.append(label_widget)
            row_layout.addWidget(label_widget)
            checks = []
            check_row = QtWidgets.QHBoxLayout()
            for rating in ["0", "1", "2"]:
                cb = QtWidgets.QCheckBox(rating)
                checks.append(cb)
                check_row.addWidget(cb)
            row_layout.addLayout(check_row)
            layout.addWidget(widget)
            self.rating_checkboxes.append(checks)

        export_btn = QtWidgets.QPushButton("Export")
        export_btn.clicked.connect(self._export_selected)
        layout.addWidget(export_btn)

    def _pair_checkbox(self, yes: QtWidgets.QCheckBox, no: QtWidgets.QCheckBox) -> None:
        yes.toggled.connect(lambda checked: no.setChecked(False) if checked else None)
        no.toggled.connect(lambda checked: yes.setChecked(False) if checked else None)

    def _wrap_checkbox(self, label: str, yes: QtWidgets.QCheckBox, no: QtWidgets.QCheckBox) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel(label))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(yes)
        row.addWidget(no)
        layout.addLayout(row)
        return widget

    def _sync_settings_ui(self) -> None:
        for edit, name in zip(self.name_edits, self.cfg.names):
            edit.setText(name)
        self.patch_width_edit.setText(str(self.cfg.patch_width))
        self.patch_height_edit.setText(str(self.cfg.patch_height))
        self.patch_stride_x_edit.setText(str(self.cfg.patch_stride_x))
        self.patch_stride_y_edit.setText(str(self.cfg.patch_stride_y))
        self.thickness_combo.setCurrentText(str(self.cfg.marker_thickness))
        self.tolerance_edit.setText(str(self.cfg.rating1_tolerance))

    def _sync_patch_dim_edits(self) -> None:
        values = [
            self.cfg.patch_width,
            self.cfg.patch_height,
            self.cfg.patch_stride_x,
            self.cfg.patch_stride_y,
        ]
        for edit, value in zip(self.patch_dim_edits, values):
            edit.setText(str(value))

    def _set_marker_thickness(self, size: int) -> None:
        self.cfg.marker_thickness = size
        self.thickness_combo.setCurrentText(str(size))
        save_config(self.config_path, self.cfg)
        self._update_canvas_brush()

    def _update_canvas_brush(self, marker_id: Optional[int] = None) -> None:
        marker = marker_id if marker_id is not None else getattr(self, "current_marker", 1)
        self.current_marker = marker
        draw_mode = marker != 7
        color = self._marker_color(marker)
        self.image_window.canvas.set_brush(color, self.cfg.marker_thickness, draw_mode)

    def _marker_color(self, marker_id: int) -> tuple[int, int, int]:
        mapping = {
            1: (255, 182, 56),
            2: (69, 200, 255),
            3: (49, 49, 255),
            4: (110, 193, 0),
            5: (255, 82, 140),
            6: (196, 102, 255),
            8: (36, 137, 244),
        }
        return mapping.get(marker_id, (255, 182, 56))

    def _add_marker_button(self, layout: QtWidgets.QHBoxLayout, icon: str, tooltip: str, marker_id: int, text: bool = False) -> None:
        btn = QtWidgets.QToolButton()
        btn.setIcon(QtGui.QIcon(str(RESOURCES / icon)))
        btn.setToolTip(tooltip)
        if text:
            btn.setText(tooltip)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.clicked.connect(lambda _=False, mid=marker_id: self._update_canvas_brush(mid))
        layout.addWidget(btn)
        self.marker_buttons.append(btn)

    def _on_load_action(self, mode: int) -> None:
        if mode == 1:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.main_window,
                "Choose image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
            if not file_path:
                return
            self._load_image(Path(file_path))
            self.file_list = []
        elif mode == 2:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self.main_window, "Load session", ""
            )
            if not folder:
                return
            self._load_session(Path(folder))
            self.file_list = []
        elif mode == 3:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self.main_window, "Choose folder", ""
            )
            if not folder:
                return
            self.file_list = self._collect_images(Path(folder))
            self.current_index = 0
            if self.file_list:
                self._load_image(self.file_list[self.current_index])

        if self.state.annotated is not None:
            self.image_window.set_state(self.state)
            self.image_window.show()
            self.tools_window.show()

    def _collect_images(self, folder: Path) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

    def _load_image(self, path: Path) -> None:
        image = cv2.imread(str(path))
        if image is None:
            return
        self.state = ImageState(original=image, annotated=image.copy(), image_path=path)
        self._update_canvas_brush()
        self._clear_exports()

    def _load_session(self, folder: Path) -> None:
        annotation_folder = None
        for name in ["Annotation", "Annoatation"]:
            candidate = folder / name
            if candidate.exists():
                annotation_folder = candidate
                break
        original_folder = folder / "Original"

        annotated_image = None
        if annotation_folder and annotation_folder.exists():
            annotated_image = self._load_first_image(annotation_folder)
        original_image = None
        if original_folder.exists():
            original_image = self._load_first_image(original_folder)

        if original_image is None:
            return

        if annotated_image is None:
            annotated_image = original_image.copy()

        self.state = ImageState(original=original_image, annotated=annotated_image, image_path=None)
        self._update_canvas_brush()
        self._clear_exports()

    def _load_first_image(self, folder: Path) -> Optional[np.ndarray]:
        for path in self._collect_images(folder):
            image = cv2.imread(str(path))
            if image is not None:
                return image
        return None

    def _prev_image(self) -> None:
        if not self.file_list or self.current_index <= 0:
            return
        self.current_index -= 1
        self._load_image(self.file_list[self.current_index])
        self.image_window.set_state(self.state)

    def _next_image(self) -> None:
        if not self.file_list or self.current_index >= len(self.file_list) - 1:
            return
        self.current_index += 1
        self._load_image(self.file_list[self.current_index])
        self.image_window.set_state(self.state)

    def _show_original(self) -> None:
        if self.state.original is None:
            return
        self.image_window.canvas.show_temp_image(self.state.original)

    def _show_annotated(self) -> None:
        if self.state.original is None:
            return
        if self.state.annotated is None:
            self.state.annotated = self.state.original.copy()
        self.image_window.canvas.set_state(self.state)

    def _confirm_delete_annotations(self) -> None:
        if self.state.original is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self.tools_window,
            "Deleting Annotations",
            "Are you sure you want to delete all annotations?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.state.annotated = self.state.original.copy()
            self.image_window.canvas.set_state(self.state)

    def _end_session(self) -> None:
        for widget in QtWidgets.QApplication.topLevelWidgets():
            widget.close()

    def _reset_defaults(self) -> None:
        self.cfg = AppConfig()
        self._sync_settings_ui()
        self._sync_patch_dim_edits()
        save_config(self.config_path, self.cfg)
        self._refresh_marker_labels()
        self._update_canvas_brush()

    def _save_settings(self) -> None:
        self.cfg.marker_thickness = int(self.thickness_combo.currentText())
        self.cfg.rating1_tolerance = int(self.tolerance_edit.text() or self.cfg.rating1_tolerance)
        for idx, edit in enumerate(self.name_edits):
            self.cfg.names[idx] = edit.text()
        self.cfg.patch_width = int(float(self.patch_width_edit.text() or self.cfg.patch_width))
        self.cfg.patch_height = int(float(self.patch_height_edit.text() or self.cfg.patch_height))
        self.cfg.patch_stride_x = int(float(self.patch_stride_x_edit.text() or self.cfg.patch_stride_x))
        self.cfg.patch_stride_y = int(float(self.patch_stride_y_edit.text() or self.cfg.patch_stride_y))
        save_config(self.config_path, self.cfg)
        self._refresh_marker_labels()
        self._sync_patch_dim_edits()
        self._update_canvas_brush()

    def _refresh_marker_labels(self) -> None:
        labels = [self.cfg.names[0], self.cfg.names[1], self.cfg.names[2], self.cfg.names[3], self.cfg.names[4], self.cfg.names[6], self.cfg.names[7], self.cfg.names[5]]
        for btn, label in zip(self.marker_buttons, labels):
            btn.setToolTip(label)
            if btn.text():
                btn.setText(label)
        if hasattr(self, "grade_labels"):
            grade_names = [
                self.cfg.names[0],
                self.cfg.names[1],
                self.cfg.names[2],
                self.cfg.names[3],
                self.cfg.names[4],
                self.cfg.names[6],
                self.cfg.names[7],
                self.cfg.names[8],
                self.cfg.names[9],
            ]
            for label_widget, name in zip(self.grade_labels, grade_names):
                label_widget.setText(f"Choose ratings for {name}:")

    def _open_patch_dim_window(self) -> None:
        self._sync_patch_dim_edits()
        self.patch_dim_window.show()

    def _create_patches(self) -> None:
        if self.state.annotated is None or self.state.original is None:
            return
        self.patch_dim_window.close()
        values = [int(edit.text() or 0) for edit in self.patch_dim_edits]
        if any(v <= 0 for v in values):
            return
        self.cfg.patch_width, self.cfg.patch_height, self.cfg.patch_stride_x, self.cfg.patch_stride_y = values
        save_config(self.config_path, self.cfg)

        self._prepare_masks()
        self._generate_patches()
        if self.state.annotated is not None:
            preview = self.state.annotated.copy()
            for coord in self.patch_coords:
                x, y = coord
                cv2.rectangle(
                    preview,
                    (x, y),
                    (x + self.cfg.patch_width, y + self.cfg.patch_height),
                    (0, 255, 0),
                    2,
                )
            self._show_preview("Patches view", preview)

    def _open_export_window(self) -> None:
        self.choose_export_window.show()

    def _open_grades_window(self) -> None:
        self.grades_window.show()

    def _on_end_annotation(self) -> None:
        if self.state.annotated is None:
            return
        self._clear_exports()
        self._prepare_masks()
        self.end_annotation_window.show()

    def _show_masks(self) -> None:
        if self.state.annotated is None:
            return
        self._prepare_masks()
        titles = [
            self.cfg.names[0],
            self.cfg.names[1],
            self.cfg.names[2],
            self.cfg.names[3],
            self.cfg.names[4],
            self.cfg.names[6],
            self.cfg.names[7],
            self.cfg.names[8],
            self.cfg.names[9],
        ]
        for title, mask in zip(titles, self.masks):
            self._show_preview(title, mask)

    def _prepare_masks(self) -> None:
        if self.state.annotated is None:
            return
        hsv = cv2.cvtColor(self.state.annotated, cv2.COLOR_BGR2HSV)
        masks = []
        hsv_values = [
            (101, 199, 255),
            (21, 186, 255),
            (0, 206, 255),
            (77, 255, 193),
            (15, 217, 244),
            (130, 173, 255),
            (162, 153, 255),
        ]
        for hsv_value in hsv_values:
            lower = np.array(hsv_value, dtype=np.uint8)
            upper = np.array(hsv_value, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        combined = masks[0] | masks[1] | masks[2] | masks[3] | masks[4] | masks[5] | masks[6]
        correct = cv2.bitwise_not(combined)
        masks.append(correct)
        leather = cv2.bitwise_not(masks[5] | masks[6])
        masks.append(leather)
        self.masks = masks

    def _generate_patches(self) -> None:
        self.patch_coords = []
        self.patches = []
        self.ratings = []
        if self.state.original is None:
            return
        if not self.masks:
            self._prepare_masks()
        masks = self.masks

        patch_groups: List[List[np.ndarray]] = [[] for _ in range(10)]
        ratings = [[] for _ in range(9)]

        for x in range(0, self.state.width - self.cfg.patch_width + 1, self.cfg.patch_stride_x):
            for y in range(0, self.state.height - self.cfg.patch_height + 1, self.cfg.patch_stride_y):
                rect = (x, y, self.cfg.patch_width, self.cfg.patch_height)
                self.patch_coords.append([x, y])

                patch_groups[0].append(self.state.original[y : y + rect[3], x : x + rect[2]].copy())

                class_patches = []
                for idx in range(9):
                    class_patch = masks[idx][y : y + rect[3], x : x + rect[2]].copy()
                    class_patches.append(class_patch)
                for idx, patch in enumerate(class_patches):
                    patch_groups[idx + 1].append(patch)
                    ratings[idx].append(self._rate_patch(patch))

        self.patches = patch_groups
        self.ratings = ratings

    def _rate_patch(self, patch: np.ndarray) -> int:
        white = int(cv2.countNonZero(patch))
        total = int(patch.size)
        threshold = self.cfg.rating1_tolerance * 0.01 * total
        if white <= total - threshold:
            return 0
        if white > threshold:
            return 2
        return 1

    def _export_all(self) -> None:
        if not self.patches or not self.patch_coords:
            return
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.choose_export_window, "Choose directory", str(Path.cwd())
        )
        if not target_dir:
            return
        root = Path(target_dir)
        image_name = self._image_name()
        patches_root = root / f"Patches_{image_name}"
        names = self._patch_names()

        patch_sets = [self.patches[0]] if self._export_mask_no.isChecked() else self.patches
        names_used = ["Original"] if self._export_mask_no.isChecked() else names

        save_patches(patches_root, image_name, patch_sets, names_used)

        class_names = self._class_names()
        if self._json_collective.isChecked():
            export_json_collective(
                patches_root,
                image_name,
                self.cfg.patch_width,
                self.cfg.patch_height,
                self.patch_coords,
                class_names,
                names_used,
                self.ratings,
                self._export_mask_yes.isChecked(),
            )
        else:
            export_json_individual(
                patches_root,
                image_name,
                self.cfg.patch_width,
                self.cfg.patch_height,
                self.patch_coords,
                class_names,
                names_used,
                self.ratings,
                self._export_mask_yes.isChecked(),
            )

        if self._yolo_yes.isChecked():
            export_yolo(
                patches_root,
                self.state.width,
                self.state.height,
                self.cfg.patch_width,
                self.cfg.patch_height,
                self.patch_coords,
                self.ratings,
            )
        if self._pascal_yes.isChecked():
            export_pascal_voc(
                patches_root,
                image_name,
                str(self.state.image_path) if self.state.image_path else "",
                self.state.width,
                self.state.height,
                self.cfg.patch_width,
                self.cfg.patch_height,
                self.patch_coords,
                class_names,
                self.ratings,
            )

        self.choose_export_window.close()

    def _export_selected(self) -> None:
        if not self.patches or not self.patch_coords:
            return
        selected = self._selected_indices()
        if not selected:
            self.grades_window.close()
            return
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.choose_export_window, "Choose directory", str(Path.cwd())
        )
        if not target_dir:
            return
        root = Path(target_dir)
        image_name = self._image_name()
        patches_root = root / f"Patches_{image_name}"
        names = self._patch_names()
        names_used = ["Original"] if self._export_mask_no.isChecked() else names

        selected_patches = []
        for group in self.patches:
            selected_patches.append([group[i] for i in selected])

        selected_ratings = []
        for rating_list in self.ratings:
            selected_ratings.append([rating_list[i] for i in selected])

        save_patches(patches_root, image_name, [selected_patches[0]] if self._export_mask_no.isChecked() else selected_patches, names_used)

        class_names = self._class_names()
        coords = [self.patch_coords[i] for i in selected]
        if self._json_collective.isChecked():
            export_json_collective(
                patches_root,
                image_name,
                self.cfg.patch_width,
                self.cfg.patch_height,
                coords,
                class_names,
                names_used,
                selected_ratings,
                self._export_mask_yes.isChecked(),
            )
        else:
            export_json_individual(
                patches_root,
                image_name,
                self.cfg.patch_width,
                self.cfg.patch_height,
                coords,
                class_names,
                names_used,
                selected_ratings,
                self._export_mask_yes.isChecked(),
            )

        if self._yolo_yes.isChecked():
            export_yolo(
                patches_root,
                self.state.width,
                self.state.height,
                self.cfg.patch_width,
                self.cfg.patch_height,
                coords,
                selected_ratings,
            )
        if self._pascal_yes.isChecked():
            export_pascal_voc(
                patches_root,
                image_name,
                str(self.state.image_path) if self.state.image_path else "",
                self.state.width,
                self.state.height,
                self.cfg.patch_width,
                self.cfg.patch_height,
                coords,
                class_names,
                selected_ratings,
            )

        self.grades_window.close()
        self.choose_export_window.close()

    def _selected_indices(self) -> List[int]:
        if not self.ratings:
            return []
        wanted = []
        for row in self.rating_checkboxes:
            row_vals = []
            for idx, cb in enumerate(row):
                if cb.isChecked():
                    row_vals.append(idx)
            wanted.append(row_vals)

        selected = []
        for i in range(len(self.patch_coords)):
            match = True
            for class_idx, allowed in enumerate(wanted):
                if not allowed:
                    match = False
                    break
                if self.ratings[class_idx][i] not in allowed:
                    match = False
                    break
            if match:
                selected.append(i)
        return selected

    def _save_session(self) -> None:
        if self.state.original is None:
            return
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.end_annotation_window, "Choose main directory", str(Path.cwd())
        )
        if not target_dir:
            return
        root = Path(target_dir)
        image_name = self._image_name()
        result_dir = root / f"Results_{image_name}"
        (result_dir / "Original").mkdir(parents=True, exist_ok=True)
        (result_dir / "Masks").mkdir(parents=True, exist_ok=True)
        (result_dir / "Annotation").mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(result_dir / "Original" / f"{image_name}.bmp"), self.state.original)

        self._prepare_masks()
        mask_names = [
            self.cfg.names[0],
            self.cfg.names[1],
            self.cfg.names[2],
            self.cfg.names[3],
            self.cfg.names[4],
            self.cfg.names[6],
            self.cfg.names[7],
            self.cfg.names[8],
            self.cfg.names[9],
        ]
        for name, mask in zip(mask_names, self.masks):
            filename = f"{name}.bmp"
            cv2.imwrite(str(result_dir / "Masks" / filename), mask)

        annotated = self.state.annotated if self.state.annotated is not None else self.state.original
        cv2.imwrite(str(result_dir / "Annotation" / f"Anotacija_{image_name}.bmp"), annotated)

        QtWidgets.QMessageBox.information(self.tools_window, "Saving", "Image is saved.")

    def _image_name(self) -> str:
        if self.state.image_path:
            return self.state.image_path.stem
        return "image"

    def _patch_names(self) -> List[str]:
        return [
            "Original",
            f"Mask {self.cfg.names[0]}",
            f"Mask {self.cfg.names[1]}",
            f"Mask {self.cfg.names[2]}",
            f"Mask {self.cfg.names[3]}",
            f"Mask {self.cfg.names[4]}",
            f"Mask {self.cfg.names[6]}",
            f"Mask {self.cfg.names[7]}",
            f"Mask {self.cfg.names[8]}",
            f"Mask {self.cfg.names[9]}",
        ]

    def _class_names(self) -> List[str]:
        names = self.cfg.names.copy()
        del names[5]
        return names

    def _show_preview(self, title: str, image: np.ndarray) -> None:
        dialog = PreviewDialog(title, image, self.main_window)
        dialog.show()
        self._preview_windows.append(dialog)

    def _clear_exports(self) -> None:
        self.masks = []
        self.patch_coords = []
        self.patches = []
        self.ratings = []


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    defect_app = DefectDetectApp()
    defect_app.main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
