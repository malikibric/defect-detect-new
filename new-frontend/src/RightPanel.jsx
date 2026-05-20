import { useEffect, useRef, useState } from "react";
import {
  X,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  Eye,
  LayoutGrid,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Plus,
  Download,
  Layers,
} from "lucide-react";

export default function RightPanel({ detections = [], file = null, annotations = [], onAccept, onReject, onDelete }) {
  const [activeTab, setActiveTab] = useState("annotations");
  const [patchSettings, setPatchSettings] = useState({
    width: 200,
    height: 200,
    horizontalStride: 50,
    verticalStride: 50,
  });
  const [showPatches, setShowPatches] = useState(true);
  const [patchesCreated, setPatchesCreated] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [expandPreview, setExpandPreview] = useState(true);
  const [expandPatchView, setExpandPatchView] = useState(true);
  const [expandExportOptions, setExpandExportOptions] = useState(true);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const [maskIndex, setMaskIndex] = useState(0);
  const [patchIndex, setPatchIndex] = useState(0);
  const [maskZoom, setMaskZoom] = useState(1);
  const [patchZoom, setPatchZoom] = useState(1);
  const [fullscreenMode, setFullscreenMode] = useState(null);
  const [fullscreenIndex, setFullscreenIndex] = useState(0);
  const [fullscreenZoom, setFullscreenZoom] = useState(1);
  const [isPanningFullscreen, setIsPanningFullscreen] = useState(false);
  const fullscreenViewportRef = useRef(null);
  const panStartRef = useRef({ x: 0, y: 0, scrollLeft: 0, scrollTop: 0 });
  const patchViewRef = useRef(null);

  const [maskExport, setMaskExport] = useState(false);
  const [jsonCollective, setJsonCollective] = useState(true);
  const [jsonIndividual, setJsonIndividual] = useState(false);
  const [yoloExport, setYoloExport] = useState(false);
  const [pascalExport, setPascalExport] = useState(false);
  const [ratingById, setRatingById] = useState({});
  const [patches, setPatches] = useState([]);
  const [patchError, setPatchError] = useState(null);
  const [patchLoading, setPatchLoading] = useState(false);

  const maskPreviewItems = detections.map((d, i) => ({ id: d.id ?? `mask-${i}`, label: d.class_name ?? `Defect ${i + 1}` }));

  const exportModalDefects = detections.map((d) => ({ id: d.id, name: d.class_name ?? "Defect" }));

  const handleRatingChange = (id, value) => {
    setRatingById((prev) => ({
      ...prev,
      [id]: prev[id] === value ? null : value,
    }));
  };

  useEffect(() => {
    if (!showExportModal) return;
    const initialRatings = detections.reduce((acc, det) => {
      acc[det.id] = acc[det.id] ?? null;
      return acc;
    }, {});
    setRatingById((prev) => ({ ...initialRatings, ...prev }));
  }, [showExportModal, detections]);

  const clampZoom = (value) => Math.min(3, Math.max(0.5, value));

  const clampIndex = (current, delta, length) => {
    if (!length) return 0;
    const next = current + delta;
    if (next < 0) return 0;
    if (next >= length) return length - 1;
    return next;
  };

  const maskCount = maskPreviewItems.length;
  const patchCount = patches.length;
  const patchItem = patches[patchIndex];

  const fullscreenCount = fullscreenMode === "mask" ? maskCount : patchCount;
  const fullscreenItem =
    fullscreenMode === "mask"
      ? maskPreviewItems[fullscreenIndex]
      : patches[fullscreenIndex];

  // Keep indexes within bounds when detections change
  useEffect(() => {
    if (maskIndex >= maskCount) setMaskIndex(0);
    if (patchIndex >= patchCount) setPatchIndex(0);
    if (fullscreenIndex >= fullscreenCount) setFullscreenIndex(0);
  }, [
    maskCount,
    patchCount,
    fullscreenCount,
    maskIndex,
    patchIndex,
    fullscreenIndex,
  ]);

  const openFullscreen = (mode, index, zoom) => {
    setFullscreenMode(mode);
    setFullscreenIndex(index);
    setFullscreenZoom(zoom);
    setIsPanningFullscreen(false);
  };

  const closeFullscreen = () => {
    if (fullscreenMode === "mask") {
      setMaskIndex(fullscreenIndex);
      setMaskZoom(fullscreenZoom);
    }
    if (fullscreenMode === "patch") {
      setPatchIndex(fullscreenIndex);
      setPatchZoom(fullscreenZoom);
    }

    setIsPanningFullscreen(false);
    setFullscreenMode(null);
  };

  const changeFullscreenIndex = (delta) => {
    setFullscreenIndex((prev) => {
      const next = clampIndex(prev, delta, fullscreenCount);
      if (fullscreenMode === "mask") setMaskIndex(next);
      if (fullscreenMode === "patch") setPatchIndex(next);
      return next;
    });
  };

  const changeFullscreenZoom = (delta) => {
    setFullscreenZoom((prev) => {
      const next = clampZoom(prev + delta);
      if (fullscreenMode === "mask") setMaskZoom(next);
      if (fullscreenMode === "patch") setPatchZoom(next);
      return next;
    });
  };

  const handleCreatePatches = async () => {
    if (!file) {
      setPatchError("Upload an image first.");
      return;
    }
    if (!annotations.length) {
      setPatchError("Add at least one annotation before creating patches.");
      return;
    }
    setPatchLoading(true);
    setPatchError(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("annotations", JSON.stringify(
      annotations.map((a) => ({ bbox: a.bbox, class_name: a.class_name }))
    ));
    formData.append("patch_size", String(patchSettings.width));
    formData.append("padding_factor", "1.5");
    try {
      const response = await fetch("/api/detect/extract-patches", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setPatches(data.patches ?? []);
      setShowPatches(true);
      setPatchesCreated(true);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          patchViewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });
    } catch (err) {
      setPatchError(`Patch extraction failed: ${err.message}`);
    } finally {
      setPatchLoading(false);
    }
  };

  const onFullscreenMouseDown = (event) => {
    if (event.button !== 0 || !fullscreenViewportRef.current) return;

    const viewport = fullscreenViewportRef.current;
    setIsPanningFullscreen(true);
    panStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
    };
  };

  const onFullscreenMouseMove = (event) => {
    if (!isPanningFullscreen || !fullscreenViewportRef.current) return;

    const viewport = fullscreenViewportRef.current;
    const dx = event.clientX - panStartRef.current.x;
    const dy = event.clientY - panStartRef.current.y;
    viewport.scrollLeft = panStartRef.current.scrollLeft - dx;
    viewport.scrollTop = panStartRef.current.scrollTop - dy;
  };

  const endFullscreenPan = () => {
    setIsPanningFullscreen(false);
  };

  const onFullscreenWheel = (event) => {
    // Preserve natural scroll by default; use Cmd/Ctrl + wheel to zoom.
    if (!event.metaKey && !event.ctrlKey) return;

    event.preventDefault();
    const step = event.deltaY < 0 ? 0.1 : -0.1;
    changeFullscreenZoom(step);
  };

  useEffect(() => {
    if (!fullscreenMode) return;

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        closeFullscreen();
        return;
      }
      if (event.key === "ArrowLeft") {
        changeFullscreenIndex(-1);
        return;
      }
      if (event.key === "ArrowRight") {
        changeFullscreenIndex(1);
        return;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [fullscreenMode]);

  useEffect(() => {
    if (!fullscreenMode) return;
    if (fullscreenMode === "mask") {
      setFullscreenIndex(maskIndex);
    }
    if (fullscreenMode === "patch") {
      setFullscreenIndex(patchIndex);
    }
  }, [fullscreenMode, maskIndex, patchIndex]);

  useEffect(() => {
    if (!fullscreenMode) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = previousOverflow;
      setIsPanningFullscreen(false);
    };
  }, [fullscreenMode]);

  return (
    <>
      <aside className={`right-panel ${isCollapsed ? "collapsed" : ""}`}>
        <div className="tab-bar">
          <button
            type="button"
            className={`tab-item ${activeTab === "annotations" ? "active" : ""}`}
            onClick={() => setActiveTab("annotations")}
          >
            Annotations
          </button>
          <button
            type="button"
            className={`tab-item ${activeTab === "patches" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("patches");
              setShowPatches(true);
            }}
          >
            Export & Patches
          </button>
          <button
            type="button"
            className={`tab-arrow-btn ${isCollapsed ? "collapsed" : ""}`}
            title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            onClick={() => setIsCollapsed((prev) => !prev)}
          >
            <ChevronRight size={14} />
          </button>
        </div>

        {activeTab === "annotations" ? (
          <div className="tab-content">
            {detections.length === 0 ? (
              <div style={{ padding: "20px", textAlign: "center", color: "#6b7280", fontSize: 13 }}>
                No annotations yet. Run Defect Scan or draw on the image.
              </div>
            ) : (
              <div className="annotations-list">
                {detections.map((det) => {
                  const statusColor = det.status === "confirmed" ? "#22c55e" : det.status === "rejected" ? "#ef4444" : "#f59e0b";
                  return (
                    <div key={det.id} className="annotation-item">
                      <div className="annotation-item-info">
                        <span className="annotation-item-label">{det.class_name}</span>
                        {typeof det.confidence === "number" && (
                          <span className="annotation-item-confidence">{Math.round(det.confidence * 100)}%</span>
                        )}
                        <span className="annotation-item-status" style={{ color: statusColor }}>{det.status ?? "pending"}</span>
                      </div>
                      <div className="annotation-item-source">{det.source}</div>
                      <div className="annotation-item-actions">
                        <button type="button" className="annot-btn annot-btn--accept" title="Accept" onClick={() => onAccept?.(det.id)}>✓</button>
                        <button type="button" className="annot-btn annot-btn--reject" title="Reject" onClick={() => onReject?.(det.id)}>✗</button>
                        <button type="button" className="annot-btn annot-btn--delete" title="Delete" onClick={() => onDelete?.(det.id)}>🗑</button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="tab-content patches-content">
            <div className="right-panel-section create-patches-section">
              <div className="right-panel-section-header">Create Patches</div>
              <div className="form-group">
                <label>Width</label>
                <input
                  type="number"
                  value={patchSettings.width}
                  onChange={(e) =>
                    setPatchSettings((prev) => ({
                      ...prev,
                      width: Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="form-group">
                <label>Height</label>
                <input
                  type="number"
                  value={patchSettings.height}
                  onChange={(e) =>
                    setPatchSettings((prev) => ({
                      ...prev,
                      height: Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="form-group">
                <label>Horizontal stride</label>
                <input
                  type="number"
                  value={patchSettings.horizontalStride}
                  onChange={(e) =>
                    setPatchSettings((prev) => ({
                      ...prev,
                      horizontalStride: Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="form-group">
                <label>Vertical stride</label>
                <input
                  type="number"
                  value={patchSettings.verticalStride}
                  onChange={(e) =>
                    setPatchSettings((prev) => ({
                      ...prev,
                      verticalStride: Number(e.target.value),
                    }))
                  }
                />
              </div>
              <button
                type="button"
                className="primary-btn"
                style={{
                  fontWeight: "normal",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "8px",
                }}
                onClick={handleCreatePatches}
              >
                <Plus size={14} />
                Create Patches
              </button>
              {patchLoading && <div style={{ color: "#9ca3af", fontSize: 12, textAlign: "center" }}>Extracting patches…</div>}
              {patchError && <div style={{ color: "#f87171", fontSize: 12 }}>{patchError}</div>}
              {patches.length > 0 && <div style={{ color: "#6ee7b7", fontSize: 12 }}>{patches.length} patches extracted</div>}
            </div>

            {showPatches && (
              <>
                <div className="right-panel-section">
                  <button
                    type="button"
                    className="collapse-toggle"
                    style={{ border: "none" }}
                    onClick={() => setExpandPreview((prev) => !prev)}
                  >
                    <span>Preview</span>
                    {expandPreview ? (
                      <ChevronUp size={16} />
                    ) : (
                      <ChevronDown size={16} />
                    )}
                  </button>
                  {expandPreview && (
                    <div className="preview-actions">
                      <button
                        type="button"
                        className="secondary-btn"
                        disabled={!maskCount}
                        onClick={() =>
                          openFullscreen("mask", maskIndex, maskZoom)
                        }
                      >
                        <Eye size={14} />
                        Show Masks
                      </button>
                      <button
                        type="button"
                        className="secondary-btn"
                        disabled={!patchCount}
                        onClick={() =>
                          openFullscreen("patch", patchIndex, patchZoom)
                        }
                      >
                        <LayoutGrid size={14} />
                        Preview Patches
                      </button>
                      <div className="preview-disclaimer">
                        After adding/removing annotations, click 'Preview
                        Patches' again to refresh patch data.
                      </div>
                    </div>
                  )}
                </div>

                <div className="right-panel-section" ref={patchViewRef}>
                  <button
                    type="button"
                    className="collapse-toggle"
                    style={{ border: "none" }}
                    onClick={() => setExpandPatchView((prev) => !prev)}
                  >
                    <span>Patch View</span>
                    {expandPatchView ? (
                      <ChevronUp size={16} />
                    ) : (
                      <ChevronDown size={16} />
                    )}
                  </button>
                  {expandPatchView && (
                    <div
                      className="preview-header-actions"
                      style={{ justifyContent: "flex-end", width: "100%" }}
                    >
                      <button
                        type="button"
                        className="icon-btn"
                        disabled={!patchCount}
                        onClick={() =>
                          setPatchZoom((prev) => clampZoom(prev - 0.1))
                        }
                        title="Zoom out"
                      >
                        <ZoomOut size={14} />
                      </button>
                      <button
                        type="button"
                        className="icon-btn"
                        disabled={!patchCount}
                        onClick={() =>
                          setPatchZoom((prev) => clampZoom(prev + 0.1))
                        }
                        title="Zoom in"
                      >
                        <ZoomIn size={14} />
                      </button>
                      <button
                        type="button"
                        className="icon-btn"
                        disabled={!patchCount}
                        onClick={() =>
                          openFullscreen("patch", patchIndex, patchZoom)
                        }
                        title="Fullscreen"
                      >
                        <Maximize2 size={14} />
                      </button>
                    </div>
                  )}
                  {expandPatchView && (
                    <div className="preview-box">
                      <div
                        className="preview-content"
                        style={{ transform: `scale(${patchZoom})`, transformOrigin: "top center" }}
                      >
                        {patchItem ? (
                          <div className="preview-image">
                            <div className="preview-image-label">
                              {patchItem.class_name ?? `Patch ${patchIndex + 1}`}
                            </div>
                            {patchItem.image_base64 ? (
                              <img
                                src={`data:image/jpeg;base64,${patchItem.image_base64}`}
                                alt={patchItem.class_name ?? "patch"}
                                style={{ width: "100%", display: "block", borderRadius: 4 }}
                              />
                            ) : (
                              <div className="preview-image-placeholder" />
                            )}
                          </div>
                        ) : (
                          <div style={{ color: "#6b7280", fontSize: 12, textAlign: "center", padding: 16 }}>
                            No patches. Click "Create Patches" first.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div
                  className={`right-panel-section patches-gated ${patchesCreated ? "active" : "disabled"}`}
                >
                  <button
                    type="button"
                    className="collapse-toggle"
                    style={{ border: "none" }}
                    onClick={() => setExpandExportOptions((prev) => !prev)}
                  >
                    <span>Export Options</span>
                    {expandExportOptions ? (
                      <ChevronUp size={16} />
                    ) : (
                      <ChevronDown size={16} />
                    )}
                  </button>
                  {expandExportOptions && (
                    <div
                      className={`export-options ${patchesCreated ? "active" : "disabled"}`}
                    >
                      <label style={{ border: "none" }}>
                        <span>Mask export</span>
                        <input
                          type="checkbox"
                          checked={maskExport}
                          onChange={(e) => setMaskExport(e.target.checked)}
                        />
                      </label>
                      <label style={{ border: "none" }}>
                        <span>JSON export (Collective)</span>
                        <input
                          type="checkbox"
                          checked={jsonCollective}
                          onChange={(e) => setJsonCollective(e.target.checked)}
                        />
                      </label>
                      <label style={{ border: "none" }}>
                        <span>JSON export (Individual)</span>
                        <input
                          type="checkbox"
                          checked={jsonIndividual}
                          onChange={(e) => setJsonIndividual(e.target.checked)}
                        />
                      </label>
                      <label style={{ border: "none" }}>
                        <span>YOLO export</span>
                        <input
                          type="checkbox"
                          checked={yoloExport}
                          onChange={(e) => setYoloExport(e.target.checked)}
                        />
                      </label>
                      <label style={{ border: "none" }}>
                        <span>Pascal VOC export</span>
                        <input
                          type="checkbox"
                          checked={pascalExport}
                          onChange={(e) => setPascalExport(e.target.checked)}
                        />
                      </label>
                    </div>
                  )}
                </div>

                <div
                  className={`right-panel-section export-action-group patches-gated ${patchesCreated ? "active" : "disabled"}`}
                >
                  <div className="right-panel-section-header">
                    Export Patches
                  </div>
                  <button
                    type="button"
                    className="primary-btn"
                    onClick={() => {
                      const payload = {
                        filename: file?.name ?? "image",
                        annotations: annotations.map((a) => ({
                          id: a.id,
                          class_name: a.class_name,
                          confidence: a.confidence,
                          bbox: a.bbox,
                          status: a.status,
                          source: a.source,
                        })),
                        patches: patches.map((p) => ({
                          patch_id: p.patch_id,
                          class_name: p.class_name,
                          original_bbox: p.original_bbox,
                          patch_size: p.patch_size,
                        })),
                        exported_at: new Date().toISOString(),
                      };
                      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = `annotations_${Date.now()}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                  >
                    <Download size={14} style={{ marginRight: 6 }} />
                    Export All Patches
                  </button>
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={() => setShowExportModal(true)}
                  >
                    <Layers size={14} style={{ marginRight: 6 }} />
                    Export Selected Patches
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </aside>

      {fullscreenMode && (
        <div className="fullscreen-overlay" onClick={closeFullscreen}>
          <div className="fullscreen-card" onClick={(e) => e.stopPropagation()}>
            <div className="fullscreen-header">
              <span className="fullscreen-title">
                {fullscreenMode === "mask" ? "Mask" : "Patch"} Preview
                {fullscreenMode === "mask" && fullscreenCount
                  ? ` (${fullscreenIndex + 1}/${fullscreenCount})`
                  : ""}
              </span>
              <div className="fullscreen-actions">
                {fullscreenMode === "mask" && (
                  <>
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => changeFullscreenIndex(-1)}
                      disabled={!fullscreenCount}
                      title="Previous"
                    >
                      <ChevronLeft size={16} />
                    </button>
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => changeFullscreenIndex(1)}
                      disabled={!fullscreenCount}
                      title="Next"
                    >
                      <ChevronRight size={16} />
                    </button>
                  </>
                )}
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => changeFullscreenZoom(-0.1)}
                  disabled={!fullscreenCount}
                  title="Zoom out"
                >
                  <ZoomOut size={16} />
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => changeFullscreenZoom(0.1)}
                  disabled={!fullscreenCount}
                  title="Zoom in"
                >
                  <ZoomIn size={16} />
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  onClick={closeFullscreen}
                  title="Close"
                >
                  <X size={16} />
                </button>
              </div>
            </div>
            <div
              className="fullscreen-body"
              onMouseUp={endFullscreenPan}
              onMouseLeave={endFullscreenPan}
            >
              <div
                ref={fullscreenViewportRef}
                className={`fullscreen-viewport ${isPanningFullscreen ? "is-panning" : ""}`}
                onMouseDown={onFullscreenMouseDown}
                onMouseMove={onFullscreenMouseMove}
                onWheel={onFullscreenWheel}
              >
                <div
                  className="fullscreen-canvas"
                  style={{ width: "100%", height: "100%" }}
                >
                  {fullscreenItem ? (
                    <div className="preview-image" style={{ transform: `scale(${fullscreenZoom})`, transformOrigin: "center center" }}>
                      <div className="preview-image-label">
                        {fullscreenItem.label ?? fullscreenItem.class_name ?? "Patch"}
                      </div>
                      {fullscreenItem.image_base64 ? (
                        <img
                          src={`data:image/jpeg;base64,${fullscreenItem.image_base64}`}
                          alt={fullscreenItem.class_name ?? "patch"}
                          style={{ maxWidth: "100%", maxHeight: "70vh", display: "block", margin: "0 auto" }}
                        />
                      ) : (
                        <div className="preview-image-placeholder" />
                      )}
                    </div>
                  ) : null}
                </div>
              </div>

              {fullscreenMode === "mask" && (
                <>
                  <button
                    type="button"
                    className="fullscreen-nav-btn fullscreen-nav-btn--left"
                    onClick={() => changeFullscreenIndex(-1)}
                    disabled={!fullscreenCount || fullscreenIndex === 0}
                    title="Previous"
                  >
                    <ChevronLeft size={20} />
                  </button>
                  <button
                    type="button"
                    className="fullscreen-nav-btn fullscreen-nav-btn--right"
                    onClick={() => changeFullscreenIndex(1)}
                    disabled={
                      !fullscreenCount ||
                      fullscreenIndex === fullscreenCount - 1
                    }
                    title="Next"
                  >
                    <ChevronRight size={20} />
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {showExportModal && (
        <div
          className="export-modal-overlay"
          onClick={() => setShowExportModal(false)}
        >
          <div
            className="export-modal-card"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="export-modal-header">
              <h3>Export Selected Patches</h3>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setShowExportModal(false)}
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
            <div className="export-modal-body">
              <p className="export-modal-subtitle">
                Choose ratings for selected defects
              </p>
              <div className="export-modal-list">
                {exportModalDefects.map((item, index) => (
                  <div key={item.id} className="export-modal-item">
                    <div className="export-modal-item-main">
                      <span className="checkbox-text">{item.name}</span>
                      <div className="rating-options">
                        {[0, 1, 2].map((value) => {
                          const isSelected = ratingById[item.id] === value;
                          return (
                            <button
                              key={value}
                              type="button"
                              className={`rating-box ${isSelected ? "selected" : ""}`}
                              onClick={() => handleRatingChange(item.id, value)}
                              title={`Select rating ${value}`}
                              aria-label={`Rating ${value}`}
                            >
                              {value}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="export-modal-actions">
              <button
                type="button"
                className="secondary-btn"
                onClick={() => setShowExportModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary-btn"
                onClick={() => {
                  const selected = exportModalDefects.filter((d) => ratingById[d.id] !== null && ratingById[d.id] !== undefined);
                  const payload = {
                    filename: file?.name ?? "image",
                    exported_at: new Date().toISOString(),
                    selected_defects: selected.map((d) => ({
                      id: d.id,
                      name: d.name,
                      rating: ratingById[d.id],
                    })),
                    patches: patches
                      .filter((p) => selected.some((d) => d.id === p.patch_id || d.name === p.class_name))
                      .map((p) => ({
                        patch_id: p.patch_id,
                        class_name: p.class_name,
                        original_bbox: p.original_bbox,
                        patch_size: p.patch_size,
                      })),
                  };
                  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `selected_patches_${Date.now()}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                  setShowExportModal(false);
                }}
              >
                Export
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
