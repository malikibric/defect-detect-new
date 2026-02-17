from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class DrawMode(Enum):
    """Modes for drawing annotations"""
    FREE_HAND = "free_hand"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


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


class ZoomableImageCanvas(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
        self._state: Optional[ImageState] = None
        self._brush_color = (255, 182, 56)
        self._brush_size = 20
        self._draw_mode = True
        self._last_point: Optional[QtCore.QPoint] = None
        self._zoom_level = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 5.0
        self._drawing_mode = DrawMode.FREE_HAND
        self._shape_start_point: Optional[tuple[int, int]] = None
        self._is_drawing_shape = False
        self._temp_image: Optional[np.ndarray] = None
        self._undo_stack: List[np.ndarray] = []
        self._max_undo = 25

    def set_state(self, state: ImageState) -> None:
        self._state = state
        self._last_point = None
        self._zoom_level = 1.0
        self._shape_start_point = None
        self._is_drawing_shape = False
        self._temp_image = None
        self._undo_stack = []
        self._refresh()

    def _push_undo_state(self) -> None:
        if not self._state or self._state.annotated is None:
            return
        self._undo_stack.append(self._state.annotated.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def undo_last_action(self) -> None:
        if not self._state or self._state.annotated is None or not self._undo_stack:
            return
        self._state.annotated = self._undo_stack.pop()
        self._refresh()

    def set_brush(self, color: tuple[int, int, int], size: int, draw_mode: bool) -> None:
        self._brush_color = color
        self._brush_size = size
        self._draw_mode = draw_mode

    def set_drawing_mode(self, mode: DrawMode) -> None:
        """Set the drawing mode (free-hand, rectangle, ellipse)"""
        self._drawing_mode = mode
        self._shape_start_point = None
        self._is_drawing_shape = False
        self._temp_image = None

    def zoom_in(self) -> None:
        self._zoom_level = min(self._zoom_level * 1.2, self._max_zoom)
        self._refresh()

    def zoom_out(self) -> None:
        self._zoom_level = max(self._zoom_level / 1.2, self._min_zoom)
        self._refresh()

    def reset_zoom(self) -> None:
        self._zoom_level = 1.0
        self._refresh()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def _refresh(self) -> None:
        if not self._state or self._state.annotated is None:
            self.clear()
            return
        pixmap = self._to_pixmap(self._state.annotated)
        if self._zoom_level != 1.0:
            scaled_size = pixmap.size() * self._zoom_level
            pixmap = pixmap.scaled(scaled_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def show_temp_image(self, image: np.ndarray) -> None:
        pixmap = self._to_pixmap(image)
        if self._zoom_level != 1.0:
            scaled_size = pixmap.size() * self._zoom_level
            pixmap = pixmap.scaled(scaled_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
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

        x = int((point.x() - offset_x) / (scale * self._zoom_level))
        y = int((point.y() - offset_y) / (scale * self._zoom_level))

        if x < 0 or y < 0:
            return None
        original_width = int(pixmap_size.width() / self._zoom_level)
        original_height = int(pixmap_size.height() / self._zoom_level)
        if x >= original_width or y >= original_height:
            return None
        return x, y

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            self._last_point = None
            self._shape_start_point = None
            self._is_drawing_shape = False
            super().mousePressEvent(event)
            return

        if self._drawing_mode == DrawMode.FREE_HAND:
            # Free-hand drawing
            self._push_undo_state()
            self._last_point = event.position().toPoint()
        else:
            # Shape drawing (rectangle or ellipse)
            pos = self._image_pos(event.position().toPoint())
            if pos:
                self._push_undo_state()
                self._shape_start_point = pos
                self._is_drawing_shape = True
                # Save current state for preview
                if self._state and self._state.annotated is not None:
                    self._temp_image = self._state.annotated.copy()
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._state or self._state.annotated is None:
            return
        if not (event.buttons() & QtCore.Qt.LeftButton):
            self._last_point = None
            return

        if self._drawing_mode == DrawMode.FREE_HAND:
            # Free-hand drawing
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
        
        elif self._is_drawing_shape and self._shape_start_point:
            # Shape drawing preview
            current_pos = self._image_pos(event.position().toPoint())
            if current_pos and self._temp_image is not None:
                # Restore temp image for preview
                self._state.annotated = self._temp_image.copy()
                
                # Draw preview shape
                if self._draw_mode:
                    self._draw_shape_preview(self._shape_start_point, current_pos)
                else:
                    # For eraser, show empty preview
                    pass
                
                self._refresh()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._is_drawing_shape and self._shape_start_point:
            # Finalize shape drawing
            current_pos = self._image_pos(event.position().toPoint())
            if current_pos and self._state and self._state.annotated is not None:
                if self._temp_image is not None:
                    self._state.annotated = self._temp_image.copy()
                
                if self._draw_mode:
                    self._draw_shape_final(self._shape_start_point, current_pos)
                else:
                    self._erase_shape(self._shape_start_point, current_pos)
                
                self._refresh()
            
            # Reset shape drawing state
            self._shape_start_point = None
            self._is_drawing_shape = False
            self._temp_image = None
        
        super().mouseReleaseEvent(event)

    def _draw_shape_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Draw a preview of the shape being drawn - FILLED"""
        if not self._state or self._state.annotated is None:
            return
        
        if self._drawing_mode == DrawMode.RECTANGLE:
            cv2.rectangle(
                self._state.annotated,
                start,
                end,
                self._brush_color,
                -1,  # Filled instead of outline
            )
        elif self._drawing_mode == DrawMode.ELLIPSE:
            center_x = (start[0] + end[0]) // 2
            center_y = (start[1] + end[1]) // 2
            radius_x = abs(end[0] - start[0]) // 2
            radius_y = abs(end[1] - start[1]) // 2
            if radius_x > 0 and radius_y > 0:
                cv2.ellipse(
                    self._state.annotated,
                    (center_x, center_y),
                    (radius_x, radius_y),
                    0,
                    0,
                    360,
                    self._brush_color,
                    -1,  # Filled instead of outline
                )

    def _draw_shape_final(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Draw the final shape"""
        self._draw_shape_preview(start, end)

    def _erase_shape(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Erase a rectangular or elliptical area"""
        if not self._state or self._state.annotated is None or self._state.original is None:
            return
        
        x1, y1 = start
        x2, y2 = end
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Bounds checking
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self._state.annotated.shape[1], x2)
        y2 = min(self._state.annotated.shape[0], y2)
        
        if self._drawing_mode == DrawMode.RECTANGLE:
            # Erase rectangle area
            self._state.annotated[y1:y2, x1:x2] = self._state.original[y1:y2, x1:x2]
        elif self._drawing_mode == DrawMode.ELLIPSE:
            # Create ellipse mask
            mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            center_x = (x2 - x1) // 2
            center_y = (y2 - y1) // 2
            radius_x = center_x
            radius_y = center_y
            if radius_x > 0 and radius_y > 0:
                cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
                # Apply mask to erase
                region = self._state.annotated[y1:y2, x1:x2]
                original_region = self._state.original[y1:y2, x1:x2]
                for i in range(3):  # For each color channel
                    region[:, :, i] = np.where(mask > 0, original_region[:, :, i], region[:, :, i])

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


# Keep old ImageCanvas for compatibility
ImageCanvas = ZoomableImageCanvas


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
        self.image_states: Dict[Path, ImageState] = {}
        self.export_mask_items: List[Tuple[str, np.ndarray]] = []
        self.export_mask_index = 0
        self.export_patch_image: Optional[np.ndarray] = None
        self.export_patch_title = "Patch view"

        # Main window with tabs
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("DefectDetect")
        self.main_window.setWindowIcon(QtGui.QIcon(str(RESOURCES / "logo-mini.png")))
        self.main_window.resize(1200, 800)

        # Central widget with tab widget
        central_widget = QtWidgets.QWidget()
        self.main_window.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Logo at top
        logo = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap(str(RESOURCES / "logo.png"))
        logo.setPixmap(logo_pixmap.scaled(400, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        logo.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(logo)

        # Tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Build all tabs
        self._build_home_tab()
        self._build_image_tab()
        self._build_settings_tab()
        self._build_export_tab()

        # Initially disable tabs until image is loaded
        self._set_tabs_enabled(False)

    def _set_tabs_enabled(self, enabled: bool) -> None:
        """Enable/disable tabs that require an image to be loaded"""
        for i in range(1, self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, enabled)

    def _build_home_tab(self) -> None:
        """Home tab with main actions"""
        home_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(home_widget)
        layout.setSpacing(10)

        # Add some spacing at top
        layout.addStretch()

        load_image_btn = QtWidgets.QPushButton("Load Image")
        load_image_btn.setMinimumHeight(50)
        load_multi_btn = QtWidgets.QPushButton("Load Multiple Images")
        load_multi_btn.setMinimumHeight(50)
        load_session_btn = QtWidgets.QPushButton("Load Session")
        load_session_btn.setMinimumHeight(50)

        layout.addWidget(load_image_btn)
        layout.addWidget(load_multi_btn)
        layout.addWidget(load_session_btn)
        layout.addStretch()

        load_image_btn.clicked.connect(lambda: self._on_load_action(1))
        load_multi_btn.clicked.connect(lambda: self._on_load_action(3))
        load_session_btn.clicked.connect(lambda: self._on_load_action(2))

        self.tab_widget.addTab(home_widget, "Home")

    def _build_image_tab(self) -> None:
        """Image view tab with canvas, actions, tools and navigation"""
        image_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(image_widget)

        # Top actions row
        actions_toolbar = QtWidgets.QHBoxLayout()
        undo_btn = QtWidgets.QPushButton("Undo")
        undo_btn.clicked.connect(self._undo_annotation)
        remove_btn = QtWidgets.QPushButton("Remove Ann")
        remove_btn.pressed.connect(self._show_original)
        remove_btn.released.connect(self._show_annotated)
        delete_btn = QtWidgets.QPushButton("Delete Ann")
        delete_btn.clicked.connect(self._confirm_delete_annotations)
        save_image_btn = QtWidgets.QPushButton("Save")
        save_image_btn.clicked.connect(self._save_session)
        end_session_btn = QtWidgets.QPushButton("End")
        end_session_btn.clicked.connect(self._end_session)
        for btn in [undo_btn, remove_btn, delete_btn, save_image_btn, end_session_btn]:
            actions_toolbar.addWidget(btn)
        actions_toolbar.addStretch()
        layout.addLayout(actions_toolbar)

        # Zoom controls
        zoom_toolbar = QtWidgets.QHBoxLayout()
        zoom_in_btn = QtWidgets.QPushButton("Zoom In (+)")
        zoom_out_btn = QtWidgets.QPushButton("Zoom Out (-)")
        zoom_reset_btn = QtWidgets.QPushButton("Reset Zoom (1:1)")
        zoom_label = QtWidgets.QLabel("Ctrl + Mouse Wheel to zoom")
        
        zoom_toolbar.addWidget(zoom_in_btn)
        zoom_toolbar.addWidget(zoom_out_btn)
        zoom_toolbar.addWidget(zoom_reset_btn)
        zoom_toolbar.addStretch()
        zoom_toolbar.addWidget(zoom_label)
        layout.addLayout(zoom_toolbar)

        # Main content: canvas on left, tools on right
        content_layout = QtWidgets.QHBoxLayout()

        self.canvas = ZoomableImageCanvas()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        content_layout.addWidget(scroll, 1)

        right_tools_tabs = QtWidgets.QTabWidget()
        right_tools_tabs.setTabPosition(QtWidgets.QTabWidget.East)
        right_tools_tabs.setMaximumWidth(360)

        tools_widget = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout(tools_widget)

        # Drawing mode section
        draw_mode_group = QtWidgets.QGroupBox("Shapes")
        draw_mode_layout = QtWidgets.QHBoxLayout(draw_mode_group)
        self.draw_mode_buttons = []

        freehand_btn = QtWidgets.QPushButton("✏️ Free-hand")
        freehand_btn.setCheckable(True)
        freehand_btn.setChecked(True)
        freehand_btn.clicked.connect(lambda: self._set_drawing_mode(DrawMode.FREE_HAND))
        draw_mode_layout.addWidget(freehand_btn)
        self.draw_mode_buttons.append(freehand_btn)

        rect_btn = QtWidgets.QPushButton("▭ Rectangle")
        rect_btn.setCheckable(True)
        rect_btn.clicked.connect(lambda: self._set_drawing_mode(DrawMode.RECTANGLE))
        draw_mode_layout.addWidget(rect_btn)
        self.draw_mode_buttons.append(rect_btn)

        ellipse_btn = QtWidgets.QPushButton("⬭ Ellipse")
        ellipse_btn.setCheckable(True)
        ellipse_btn.clicked.connect(lambda: self._set_drawing_mode(DrawMode.ELLIPSE))
        draw_mode_layout.addWidget(ellipse_btn)
        self.draw_mode_buttons.append(ellipse_btn)

        tools_layout.addWidget(draw_mode_group)

        # Markers and eraser
        marker_group = QtWidgets.QGroupBox("Markers & Eraser")
        marker_layout = QtWidgets.QVBoxLayout(marker_group)

        marker_buttons_layout = QtWidgets.QHBoxLayout()
        self.marker_buttons = []
        self._add_marker_button(marker_buttons_layout, "1.png", self.cfg.names[0], 1)
        self._add_marker_button(marker_buttons_layout, "2.png", self.cfg.names[1], 2)
        self._add_marker_button(marker_buttons_layout, "3.png", self.cfg.names[2], 3)
        self._add_marker_button(marker_buttons_layout, "4.png", self.cfg.names[3], 4)
        self._add_marker_button(marker_buttons_layout, "8.png", self.cfg.names[4], 8)
        marker_layout.addLayout(marker_buttons_layout)

        marker_buttons_layout2 = QtWidgets.QHBoxLayout()
        self._add_marker_button(marker_buttons_layout2, "5.png", self.cfg.names[6], 5, text=True)
        self._add_marker_button(marker_buttons_layout2, "6.png", self.cfg.names[7], 6, text=True)
        self._add_marker_button(marker_buttons_layout2, "7.png", self.cfg.names[5], 7, text=True)
        marker_layout.addLayout(marker_buttons_layout2)

        thickness_layout = QtWidgets.QHBoxLayout()
        thickness_layout.addWidget(QtWidgets.QLabel("Thickness:"))
        for icon, size in [("najtanji.png", 10), ("srednji.png", 20), ("najdeblji.png", 30)]:
            btn = QtWidgets.QToolButton()
            btn.setIcon(QtGui.QIcon(str(RESOURCES / icon)))
            btn.setToolTip(str(size))
            btn.clicked.connect(lambda _=False, s=size: self._set_marker_thickness(s))
            thickness_layout.addWidget(btn)
        thickness_layout.addStretch()
        marker_layout.addLayout(thickness_layout)

        tools_layout.addWidget(marker_group)
        tools_layout.addStretch()

        right_tools_tabs.addTab(tools_widget, "Tools")
        content_layout.addWidget(right_tools_tabs)

        layout.addLayout(content_layout, 1)

        # Bottom image navigation
        self.image_nav_widget = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(self.image_nav_widget)
        self.prev_btn = QtWidgets.QPushButton("← Previous Image")
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn = QtWidgets.QPushButton("Next Image →")
        self.next_btn.clicked.connect(self._next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        self.image_nav_widget.setVisible(False)
        layout.addWidget(self.image_nav_widget)

        zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        zoom_reset_btn.clicked.connect(self.canvas.reset_zoom)

        self.tab_widget.addTab(image_widget, "Image View")

    def _build_settings_tab(self) -> None:
        """Settings tab"""
        settings_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(settings_widget)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # Names sub-tab
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

        # Patches sub-tab
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

        # Other sub-tab
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

        # Buttons
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
        self.tab_widget.addTab(settings_widget, "Settings")

    def _build_export_tab(self) -> None:
        """Export tab with all export options"""
        export_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(export_widget)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(420)

        # Main actions
        main_group = QtWidgets.QGroupBox("Generate & Preview")
        main_layout = QtWidgets.QVBoxLayout(main_group)

        show_masks_btn = QtWidgets.QPushButton("Show Masks")
        show_masks_btn.clicked.connect(self._show_masks)
        main_layout.addWidget(show_masks_btn)

        create_patches_btn = QtWidgets.QPushButton("Create Patches...")
        create_patches_btn.clicked.connect(self._open_patch_dim_window)
        main_layout.addWidget(create_patches_btn)
        left_layout.addWidget(main_group)

        # Export options
        export_group = QtWidgets.QGroupBox("Export Options")
        export_layout = QtWidgets.QVBoxLayout(export_group)

        self._export_mask_yes = QtWidgets.QCheckBox("Yes")
        self._export_mask_no = QtWidgets.QCheckBox("No")
        self._export_mask_no.setChecked(True)
        self._pair_checkbox(self._export_mask_yes, self._export_mask_no)
        export_layout.addWidget(self._wrap_checkbox("Mask export:", self._export_mask_yes, self._export_mask_no))

        self._json_collective = QtWidgets.QCheckBox("Collective")
        self._json_individual = QtWidgets.QCheckBox("Individual")
        self._json_collective.setChecked(True)
        self._pair_checkbox(self._json_collective, self._json_individual)
        export_layout.addWidget(self._wrap_checkbox("JSON file export:", self._json_collective, self._json_individual))

        self._yolo_yes = QtWidgets.QCheckBox("Yes")
        self._yolo_no = QtWidgets.QCheckBox("No")
        self._yolo_no.setChecked(True)
        self._pair_checkbox(self._yolo_yes, self._yolo_no)
        export_layout.addWidget(self._wrap_checkbox("YOLO export:", self._yolo_yes, self._yolo_no))

        self._pascal_yes = QtWidgets.QCheckBox("Yes")
        self._pascal_no = QtWidgets.QCheckBox("No")
        self._pascal_no.setChecked(True)
        self._pair_checkbox(self._pascal_yes, self._pascal_no)
        export_layout.addWidget(self._wrap_checkbox("Pascal VOC export:", self._pascal_yes, self._pascal_no))
        left_layout.addWidget(export_group)

        # Export buttons
        export_buttons_group = QtWidgets.QGroupBox("Export Patches")
        export_buttons_layout = QtWidgets.QVBoxLayout(export_buttons_group)

        export_all_btn = QtWidgets.QPushButton("Export All Patches")
        export_all_btn.clicked.connect(self._export_all)
        export_buttons_layout.addWidget(export_all_btn)

        export_selected_btn = QtWidgets.QPushButton("Export Selected Patches...")
        export_selected_btn.clicked.connect(self._open_grades_window)
        export_buttons_layout.addWidget(export_selected_btn)
        left_layout.addWidget(export_buttons_group)

        left_layout.addStretch()
        layout.addWidget(left_panel)

        # Right preview area (embedded instead of popup windows)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        masks_group = QtWidgets.QGroupBox("Masks")
        masks_layout = QtWidgets.QVBoxLayout(masks_group)
        nav_layout = QtWidgets.QHBoxLayout()
        self.export_prev_mask_btn = QtWidgets.QPushButton("← Prev")
        self.export_prev_mask_btn.clicked.connect(self._export_prev_mask)
        self.export_next_mask_btn = QtWidgets.QPushButton("Next →")
        self.export_next_mask_btn.clicked.connect(self._export_next_mask)
        nav_layout.addWidget(self.export_prev_mask_btn)
        nav_layout.addWidget(self.export_next_mask_btn)
        nav_layout.addStretch()
        masks_layout.addLayout(nav_layout)

        self.export_mask_name_label = QtWidgets.QLabel("No masks shown")
        self.export_mask_name_label.setAlignment(QtCore.Qt.AlignCenter)
        masks_layout.addWidget(self.export_mask_name_label)

        self.export_mask_image_label = QtWidgets.QLabel()
        self.export_mask_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.export_mask_image_label.setMinimumSize(420, 240)
        self.export_mask_image_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        masks_layout.addWidget(self.export_mask_image_label)

        patch_group = QtWidgets.QGroupBox("Patch view")
        patch_layout = QtWidgets.QVBoxLayout(patch_group)
        self.export_patch_name_label = QtWidgets.QLabel("No patch view generated")
        self.export_patch_name_label.setAlignment(QtCore.Qt.AlignCenter)
        patch_layout.addWidget(self.export_patch_name_label)

        self.export_patch_image_label = QtWidgets.QLabel()
        self.export_patch_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.export_patch_image_label.setMinimumSize(420, 240)
        self.export_patch_image_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        patch_layout.addWidget(self.export_patch_image_label)

        right_layout.addWidget(masks_group)
        right_layout.addWidget(patch_group)
        layout.addWidget(right_panel, 1)

        self._update_export_mask_preview()
        self._update_export_patch_preview()

        self.tab_widget.addTab(export_widget, "Export")

        # Build dialogs
        self._build_patch_dim_window()
        self._build_grades_window()

    def _build_patch_dim_window(self) -> None:
        self.patch_dim_window = QtWidgets.QDialog(self.main_window)
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

    def _build_grades_window(self) -> None:
        self.grades_window = QtWidgets.QDialog(self.main_window)
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

    def _set_drawing_mode(self, mode: DrawMode) -> None:
        """Set the drawing mode and update button states"""
        if hasattr(self, 'canvas'):
            self.canvas.set_drawing_mode(mode)
        
        # Update button states
        for i, btn in enumerate(self.draw_mode_buttons):
            if i == 0 and mode == DrawMode.FREE_HAND:
                btn.setChecked(True)
            elif i == 1 and mode == DrawMode.RECTANGLE:
                btn.setChecked(True)
            elif i == 2 and mode == DrawMode.ELLIPSE:
                btn.setChecked(True)
            else:
                btn.setChecked(False)

    def _update_canvas_brush(self, marker_id: Optional[int] = None) -> None:
        marker = marker_id if marker_id is not None else getattr(self, "current_marker", 1)
        self.current_marker = marker
        draw_mode = marker != 7
        color = self._marker_color(marker)
        if hasattr(self, 'canvas'):
            self.canvas.set_brush(color, self.cfg.marker_thickness, draw_mode)

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
            self.image_states = {}
            self._load_image(Path(file_path))
            self.file_list = []
        elif mode == 2:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self.main_window, "Load session", ""
            )
            if not folder:
                return
            self.image_states = {}
            self._load_session(Path(folder))
            self.file_list = []
        elif mode == 3:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self.main_window, "Choose folder", ""
            )
            if not folder:
                return
            self.image_states = {}
            self.file_list = self._collect_images(Path(folder))
            self.current_index = 0
            if self.file_list:
                self._load_image(self.file_list[self.current_index])

        if self.state.annotated is not None:
            self.canvas.set_state(self.state)
            self._set_tabs_enabled(True)
            self.tab_widget.setCurrentIndex(1)  # Switch to Image View tab
            self._update_navigation_visibility()

    def _collect_images(self, folder: Path) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

    def _load_image(self, path: Path) -> None:
        image_path = path.resolve()
        existing_state = self.image_states.get(image_path)
        if existing_state is not None:
            self.state = existing_state
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                return
            state = ImageState(original=image, annotated=image.copy(), image_path=image_path)
            self.image_states[image_path] = state
            self.state = state
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
        self.canvas.set_state(self.state)
        self._update_navigation_visibility()

    def _next_image(self) -> None:
        if not self.file_list or self.current_index >= len(self.file_list) - 1:
            return
        self.current_index += 1
        self._load_image(self.file_list[self.current_index])
        self.canvas.set_state(self.state)
        self._update_navigation_visibility()

    def _update_navigation_visibility(self) -> None:
        if not hasattr(self, "image_nav_widget"):
            return
        has_multiple = len(self.file_list) > 1
        self.image_nav_widget.setVisible(has_multiple)
        if hasattr(self, "prev_btn"):
            self.prev_btn.setEnabled(has_multiple and self.current_index > 0)
        if hasattr(self, "next_btn"):
            self.next_btn.setEnabled(has_multiple and self.current_index < len(self.file_list) - 1)

    def _undo_annotation(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.undo_last_action()

    def _show_original(self) -> None:
        if self.state.original is None:
            return
        self.canvas.show_temp_image(self.state.original)

    def _show_annotated(self) -> None:
        if self.state.original is None:
            return
        if self.state.annotated is None:
            self.state.annotated = self.state.original.copy()
        self.canvas.set_state(self.state)

    def _confirm_delete_annotations(self) -> None:
        if self.state.original is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self.main_window,
            "Deleting Annotations",
            "Are you sure you want to delete all annotations?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.state.annotated = self.state.original.copy()
            self.canvas.set_state(self.state)

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
            self.export_patch_image = preview
            self.export_patch_title = "Patches view"
            self._update_export_patch_preview()

    def _open_grades_window(self) -> None:
        self.grades_window.show()

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
        self.export_mask_items = [(title, mask) for title, mask in zip(titles, self.masks)]
        self.export_mask_index = 0
        self._update_export_mask_preview()

    def _array_to_qpixmap(self, image: np.ndarray) -> QtGui.QPixmap:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(qimage)

    def _set_export_preview_label(self, label: QtWidgets.QLabel, image: np.ndarray) -> None:
        pixmap = self._array_to_qpixmap(image)
        target_w = max(label.width(), 420)
        target_h = max(label.height(), 240)
        scaled = pixmap.scaled(target_w, target_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _update_export_mask_preview(self) -> None:
        if not hasattr(self, "export_mask_name_label"):
            return
        if not self.export_mask_items:
            self.export_mask_name_label.setText("No masks shown")
            self.export_mask_image_label.clear()
            self.export_prev_mask_btn.setEnabled(False)
            self.export_next_mask_btn.setEnabled(False)
            return

        title, image = self.export_mask_items[self.export_mask_index]
        self.export_mask_name_label.setText(title)
        self._set_export_preview_label(self.export_mask_image_label, image)
        self.export_prev_mask_btn.setEnabled(self.export_mask_index > 0)
        self.export_next_mask_btn.setEnabled(self.export_mask_index < len(self.export_mask_items) - 1)

    def _update_export_patch_preview(self) -> None:
        if not hasattr(self, "export_patch_name_label"):
            return
        if self.export_patch_image is None:
            self.export_patch_name_label.setText("No patch view generated")
            self.export_patch_image_label.clear()
            return
        self.export_patch_name_label.setText(self.export_patch_title)
        self._set_export_preview_label(self.export_patch_image_label, self.export_patch_image)

    def _export_prev_mask(self) -> None:
        if self.export_mask_index <= 0:
            return
        self.export_mask_index -= 1
        self._update_export_mask_preview()

    def _export_next_mask(self) -> None:
        if self.export_mask_index >= len(self.export_mask_items) - 1:
            return
        self.export_mask_index += 1
        self._update_export_mask_preview()

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
            # Fill enclosed regions so outlined rectangle/ellipse annotations
            # are exported as complete selected areas.
            mask = self._fill_enclosed_regions(mask)
            masks.append(mask)
        combined = masks[0] | masks[1] | masks[2] | masks[3] | masks[4] | masks[5] | masks[6]
        correct = cv2.bitwise_not(combined)
        masks.append(correct)
        leather = cv2.bitwise_not(masks[5] | masks[6])
        masks.append(leather)
        self.masks = masks

    def _fill_enclosed_regions(self, mask: np.ndarray) -> np.ndarray:
        """Fill closed contours in a binary mask.

        This ensures a drawn outline (rectangle/ellipse) behaves like a full
        selected region for patch extraction and export.
        """
        if mask.size == 0:
            return mask

        filled = mask.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
        return filled

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
            self.main_window, "Choose directory", str(Path.cwd())
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

        # Export completed

    def _export_selected(self) -> None:
        if not self.patches or not self.patch_coords:
            return
        selected = self._selected_indices()
        if not selected:
            self.grades_window.close()
            return
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.main_window, "Choose directory", str(Path.cwd())
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
            self.main_window, "Choose main directory", str(Path.cwd())
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

        QtWidgets.QMessageBox.information(self.main_window, "Saving", "Image is saved.")

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

    def _clear_exports(self) -> None:
        self.masks = []
        self.patch_coords = []
        self.patches = []
        self.ratings = []
        self.export_mask_items = []
        self.export_mask_index = 0
        self.export_patch_image = None
        self.export_patch_title = "Patch view"
        self._update_export_mask_preview()
        self._update_export_patch_preview()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    defect_app = DefectDetectApp()
    defect_app.main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
