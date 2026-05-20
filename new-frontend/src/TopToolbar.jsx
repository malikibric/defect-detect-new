import React, { useState, useEffect } from "react";
import {
  Save,
  Settings,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Trash2,
  Sun,
  Moon,
} from "lucide-react";
import logoColor from "./assets/logo color.png";

import nameColor from "./assets/name color.png";
export default function TopToolbar({
  zoom,
  onZoomChange,
  onOpenSettings,
  onLogoClick,
  darkMode,
  onToggleTheme,
  annotationsVisible,
  onToggleAnnotationsVisibility,
  selectedAnnotation,
  onDeleteSelected,
}) {
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setShowDeleteModal(false);
      }
    };

    if (showDeleteModal) {
      window.addEventListener("keydown", onKeyDown);
    }

    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showDeleteModal]);

  const openDeleteModal = () => setShowDeleteModal(true);
  const closeDeleteModal = () => setShowDeleteModal(false);
  const confirmDelete = () => {
    if (onDeleteSelected) onDeleteSelected();
    setShowDeleteModal(false);
  };

  return (
    <>
      <header className="top-toolbar">
        <div className="top-toolbar-left">
          <button
            className="top-toolbar-logo"
            type="button"
            onClick={onLogoClick}
          >
            <span className="logo-icon" aria-hidden>
              <img src={logoColor} alt="" />
            </span>
            <img
              src={nameColor}
              alt="Annotic"
              style={{
                height: "27.5px",
                width: "auto",
                display: "block",
                marginLeft: "-20px",
              }}
            />
          </button>

          <div
            className="top-toolbar-file-actions"
            aria-label="File actions"
          ></div>

          <div className="top-toolbar-actions">
            <button type="button" className="top-toolbar-icon" title="Undo">
              <Undo2 />
            </button>
            <button type="button" className="top-toolbar-icon" title="Redo">
              <Redo2 />
            </button>
            <div className="top-toolbar-zoom">
              <button
                type="button"
                className="top-toolbar-icon"
                onClick={() => onZoomChange(Math.max(25, zoom - 10))}
                title="Zoom out"
              >
                <ZoomOut />
              </button>
              <span
                className="top-toolbar-zoom-label"
                style={{ fontWeight: 400, fontSize: "0.875rem" }}
              >
                {zoom}%
              </span>
              <button
                type="button"
                className="top-toolbar-icon"
                onClick={() => onZoomChange(Math.min(300, zoom + 10))}
                title="Zoom in"
              >
                <ZoomIn />
              </button>
              <button
                type="button"
                className="top-toolbar-icon"
                onClick={() => onZoomChange(100)}
                title="Reset zoom"
              >
                <RotateCcw />
              </button>
            </div>

            <div className="top-toolbar-extra-actions">
              <button
                type="button"
                className={`top-toolbar-icon ${!annotationsVisible ? "active" : ""}`}
                onClick={onToggleAnnotationsVisibility}
                title={
                  annotationsVisible ? "Hide annotations" : "Show annotations"
                }
              >
                {annotationsVisible ? <Eye /> : <EyeOff />}
              </button>
              <button type="button" className="top-toolbar-icon" title="Save">
                <Save />
              </button>
              <button
                type="button"
                className="top-toolbar-icon"
                onClick={openDeleteModal}
                title="Delete selected"
              >
                <span className="top-toolbar-icon-inner">
                  <Trash2 />
                </span>
              </button>
            </div>
          </div>
        </div>

        <div className="top-toolbar-right">
          <button
            type="button"
            className="top-toolbar-icon"
            onClick={onToggleTheme}
            title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {darkMode ? <Sun /> : <Moon />}
          </button>

          <button
            type="button"
            className="top-toolbar-icon"
            onClick={onOpenSettings}
            title="Settings"
          >
            <Settings />
          </button>
        </div>
      </header>
      {showDeleteModal ? (
        <div className="confirm-modal-overlay" onClick={closeDeleteModal}>
          <div
            className="confirm-modal-card"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>Are you sure you want to delete all annotations?</h2>
            <div className="confirm-modal-actions">
              <button
                type="button"
                className="delete-confirm-button delete-confirm-button--secondary"
                style={{
                  backgroundColor: "#1F2937",
                  color: "#E5E7EB",
                  border: "1px solid #1F2937",
                  borderRadius: "8px",
                  padding: "8px 14px",
                  minWidth: "72px",
                  cursor: "pointer",
                  fontSize: "0.9rem",
                  fontWeight: 700,
                }}
                onClick={closeDeleteModal}
              >
                No
              </button>
              <button
                type="button"
                className="delete-confirm-button delete-confirm-button--primary"
                style={{
                  backgroundColor: "#5B21B6",
                  color: "#ffffff",
                  border: "1px solid #5B21B6",
                  borderRadius: "8px",
                  padding: "8px 14px",
                  minWidth: "72px",
                  cursor: "pointer",
                  fontSize: "0.9rem",
                  fontWeight: 700,
                }}
                onClick={confirmDelete}
              >
                Yes
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
