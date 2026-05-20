import { useEffect, useRef, useState } from "react";
import { jsPDF } from "jspdf";
import ImageUpload from "./ImageUpload";
import ImageCanvas from "./ImageCanvas";
import SidebarTools from "./SidebarTools";
import TopToolbar from "./TopToolbar";
import RightPanel from "./RightPanel";
import "./App.css";

function AnnotationScreen({
  file,
  onFileSelect,
  onBack,
  onLogoClick,
  isDark,
  setIsDark,
}) {
  const [zoom, setZoom] = useState(100);
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState(null);
  const [activeTool, setActiveTool] = useState("Pointer");
  const [selectedColor, setSelectedColor] = useState("#387df7");
  const [annotations, setAnnotations] = useState([]);
  const autoScannedRef = useRef(false);

  const clampZoom = (value) => Math.min(300, Math.max(25, value));
  const onZoomChange = (value) => setZoom(clampZoom(value));

  const [selectedId, setSelectedId] = useState(null);
  const [annotationsVisible, setAnnotationsVisible] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [settingsValues, setSettingsValues] = useState({
    marker1: "Defect 1",
    marker2: "Defect 2",
    marker3: "Defect 3",
    marker4: "Defect 4",
    marker5: "Defect 5",
    eraser: "Defect 6",
    edge: "Defect 7",
    surface: "Eraser",
    classCorrect: "Class Correct",
    classLeather: "Leather",
    patchWidth: "200",
    patchHeight: "200",
    patchHorizontalStride: "50",
    patchVerticalStride: "50",
    ratingTolerance: "95",
  });
  const defaultSettingsValues = {
    marker1: "Defect 1",
    marker2: "Defect 2",
    marker3: "Defect 3",
    marker4: "Defect 4",
    marker5: "Defect 5",
    eraser: "Defect 6",
    edge: "Defect 7",
    surface: "Eraser",
    classCorrect: "Class Correct",
    classLeather: "Leather",
    patchWidth: "200",
    patchHeight: "200",
    patchHorizontalStride: "50",
    patchVerticalStride: "50",
    ratingTolerance: "95",
  };

  useEffect(() => {
    if (!settingsOpen) return;

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setSettingsOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [settingsOpen]);

  const handleOverlayClick = () => {
    setSettingsOpen(false);
  };

  const handleModalClick = (event) => {
    event.stopPropagation();
  };

  const saveSettings = () => {
    // No backend hook yet. Just close modal.
    setSettingsOpen(false);
  };

  const runYolo = async () => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/api/upload-image", { method: "POST", body: formData });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const raw = data.detections ?? [];
    return raw.map((det, idx) => ({
      ...det,
      id: det.id ?? `${det.class_name}-${idx}`,
      source: "AI Detected",
      status: "pending",
    }));
  };

  // Defect Scan: run YOLO, replace AI detections but keep manual annotations
  const handleDefectScan = async () => {
    if (!file || scanning) return;
    setScanning(true);
    setScanError(null);
    try {
      const detected = await runYolo();
      setAnnotations((prev) => {
        const manual = prev.filter((a) => a.source === "Manual");
        return [...manual, ...detected];
      });
    } catch (err) {
      console.error("Defect scan failed:", err);
      setScanError(err.message.includes("502") || err.message.includes("503")
        ? "AI backend is still starting up. Wait a moment and try again."
        : `Defect scan failed: ${err.message}`);
    } finally {
      setScanning(false);
    }
  };

  // Auto Annotate: run YOLO, add only new detections not already present
  const handleAutoAnnotate = async () => {
    if (!file || scanning) return;
    setScanning(true);
    setScanError(null);
    try {
      const detected = await runYolo();
      if (detected.length) {
        setAnnotations((prev) => {
          const existingKeys = new Set(prev.map((a) => a.bbox?.join(",")));
          const novel = detected
            .filter((d) => !existingKeys.has(d.bbox?.join(",")))
            .map((d, i) => ({
              ...d,
              id: `auto-${Date.now()}-${i}`,
              source: "AI Detected",
              status: "pending",
            }));
          return [...prev, ...novel];
        });
      }
    } catch (err) {
      console.error("Auto annotate failed:", err);
      setScanError(err.message.includes("502") || err.message.includes("503")
        ? "AI backend is still starting up. Wait a moment and try again."
        : `Auto annotate failed: ${err.message}`);
    } finally {
      setScanning(false);
    }
  };

  const resetSettings = () => {
    setSettingsValues(defaultSettingsValues);
  };

  const updateSetting = (key, value) => {
    setSettingsValues((prev) => ({ ...prev, [key]: value }));
  };

  useEffect(() => {
    if (!file) {
      autoScannedRef.current = false;
      return;
    }
    if (autoScannedRef.current) return;
    autoScannedRef.current = true;
    handleDefectScan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);
  const [uploadedFiles, setUploadedFiles] = useState(new Set());

  useEffect(() => {
    if (file && file.name) {
      setUploadedFiles((prev) => {
        const next = new Set(prev);
        next.add(file.name);
        return next;
      });
    }
  }, [file]);

  const handleGenerateReport = () => {
    try {
      const doc = new jsPDF();
      const trunc = (str, n) => {
        const s = str ?? "";
        return s.length > n ? s.substring(0, n - 1) + "…" : s;
      };

      // Top Header styling
      doc.setFillColor(11, 11, 18);
      doc.rect(0, 0, 210, 40, "F");

      doc.setTextColor(255, 255, 255);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(22);
      doc.text("DefectDetect", 15, 18);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text("AI & Annotation Session Report", 15, 28);

      const dateStr = new Date().toLocaleString();
      doc.setFontSize(9);
      doc.text(`Generated: ${dateStr}`, 140, 28);

      // Section 1: Session Overview
      doc.setTextColor(11, 11, 18);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("1. Session Overview", 15, 55);

      doc.setDrawColor(200, 200, 200);
      doc.line(15, 58, 195, 58);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`Total Images Uploaded this Session: ${uploadedFiles.size}`, 20, 68);

      const fileList = Array.from(uploadedFiles).join(", ");
      doc.text(`Uploaded Images: ${fileList || "None"}`, 20, 75);

      // Section 2: Active Image & Annotation Statistics
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("2. Active Image Metadata", 15, 95);
      doc.line(15, 98, 195, 98);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`Active Image Name: ${file ? file.name : "None"}`, 20, 108);
      doc.text(`Image Dimensions: ${imageSize.width > 0 ? `${imageSize.width} x ${imageSize.height} px` : "Unknown"}`, 20, 115);

      const confirmedCount = annotations.filter((a) => a.status === "confirmed").length;
      const pendingCount = annotations.filter((a) => a.status === "pending" || !a.status).length;
      const rejectedCount = annotations.filter((a) => a.status === "rejected").length;

      doc.text(`Total Annotations/Masks: ${annotations.length}`, 20, 122);
      doc.text(`  - Confirmed: ${confirmedCount}`, 25, 129);
      doc.text(`  - Pending: ${pendingCount}`, 25, 136);
      doc.text(`  - Rejected: ${rejectedCount}`, 25, 143);

      // Section 3: Patch Configuration
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("3. Patch Configuration & Estimation", 15, 163);
      doc.line(15, 166, 195, 166);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`Patch Dimensions: ${settingsValues.patchWidth} x ${settingsValues.patchHeight} px`, 20, 176);
      doc.text(`Stride: Horizontal: ${settingsValues.patchHorizontalStride}px, Vertical: ${settingsValues.patchVerticalStride}px`, 20, 183);

      const pw = Number(settingsValues.patchWidth);
      const ph = Number(settingsValues.patchHeight);
      const hs = Number(settingsValues.patchHorizontalStride);
      const vs = Number(settingsValues.patchVerticalStride);
      const cols = imageSize.width > 0 && hs > 0
        ? Math.max(0, Math.floor((imageSize.width - pw) / hs) + 1)
        : 0;
      const rows = imageSize.height > 0 && vs > 0
        ? Math.max(0, Math.floor((imageSize.height - ph) / vs) + 1)
        : 0;
      doc.text(`Estimated Patches to Extract: ${cols * rows}`, 20, 190);

      // Section 4: AI Analysis Summary
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("4. AI Analysis Summary", 15, 210);
      doc.line(15, 213, 195, 213);

      const aiDetections = annotations.filter((a) => a.source === "AI Detected");
      const samDetections = annotations.filter((a) => a.source === "SAM Propagated");
      const manualAnnotations = annotations.filter((a) => a.source === "Manual" || !a.source);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`AI Detected: ${aiDetections.length}`, 20, 223);
      doc.text(`SAM Propagated: ${samDetections.length}`, 20, 230);
      doc.text(`Manual: ${manualAnnotations.length}`, 20, 237);

      let aiSectionY = 245;
      if (aiDetections.length === 0) {
        doc.setFont("helvetica", "italic");
        doc.text("No AI detections for this image.", 20, aiSectionY);
        aiSectionY += 7;
      } else {
        const confidences = aiDetections.map((a) => a.confidence ?? 0);
        const avgConf = confidences.reduce((s, v) => s + v, 0) / confidences.length;
        const minConf = Math.min(...confidences);
        const maxConf = Math.max(...confidences);
        doc.setFont("helvetica", "normal");
        doc.text(`Avg Confidence: ${(avgConf * 100).toFixed(1)}%  |  Min: ${(minConf * 100).toFixed(1)}%  |  Max: ${(maxConf * 100).toFixed(1)}%`, 20, aiSectionY);
        aiSectionY += 9;

        // Per-class breakdown
        const classMap = {};
        aiDetections.forEach((a) => {
          const cls = a.class_name || "Unknown";
          if (!classMap[cls]) classMap[cls] = { count: 0, total: 0 };
          classMap[cls].count += 1;
          classMap[cls].total += a.confidence ?? 0;
        });
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9);
        doc.text("Class", 20, aiSectionY);
        doc.text("Count", 90, aiSectionY);
        doc.text("Avg Conf.", 130, aiSectionY);
        doc.line(20, aiSectionY + 2, 175, aiSectionY + 2);
        aiSectionY += 8;
        doc.setFont("helvetica", "normal");
        Object.entries(classMap).forEach(([cls, { count, total }]) => {
          if (aiSectionY > 280) { doc.addPage(); aiSectionY = 20; }
          doc.text(trunc(cls, 20), 20, aiSectionY);
          doc.text(String(count), 90, aiSectionY);
          doc.text(`${((total / count) * 100).toFixed(1)}%`, 130, aiSectionY);
          aiSectionY += 7;
        });
      }

      // Section 5: Annotations Listing
      if (aiSectionY > 250) { doc.addPage(); aiSectionY = 20; }
      const listingY = aiSectionY + 10;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("5. Annotations Listing", 15, listingY);
      doc.line(15, listingY + 3, 195, listingY + 3);

      let yPos = listingY + 13;
      if (annotations.length === 0) {
        doc.setFont("helvetica", "italic");
        doc.setFontSize(10);
        doc.text("No annotations detected or created yet for the active image.", 20, yPos);
      } else {
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9);
        doc.text("ID", 20, yPos);
        doc.text("Class Name", 50, yPos);
        doc.text("Source", 100, yPos);
        doc.text("Status", 143, yPos);
        doc.text("Conf.", 170, yPos);
        doc.line(20, yPos + 2, 192, yPos + 2);

        yPos += 8;
        doc.setFont("helvetica", "normal");
        annotations.forEach((ann, index) => {
          if (yPos > 280) { doc.addPage(); yPos = 20; }
          const isAI = ann.source === "AI Detected" || ann.source === "SAM Propagated";
          const confStr = isAI && ann.confidence != null
            ? `${(ann.confidence * 100).toFixed(1)}%`
            : "—";
          doc.text(trunc(ann.id ? String(ann.id) : `Ann-${index}`, 12), 20, yPos);
          doc.text(trunc(ann.class_name || "Unknown", 14), 50, yPos);
          doc.text(trunc(ann.source || "Manual", 12), 100, yPos);
          doc.text(trunc(ann.status || "pending", 10), 143, yPos);
          doc.text(confStr, 170, yPos);
          yPos += 7;
        });
      }

      doc.save(`defect_detect_report_${Date.now()}.pdf`);
    } catch (err) {
      console.error("Failed to generate PDF report:", err);
      alert("Report generation failed. Please try again.");
    }
  };

  const annotationsCount = annotations.length;
  const confirmedCount = annotations.filter((a) => a.status === "confirmed").length;

  return (
    <div
      className={`annotation-screen ${isDark ? "theme-dark" : "theme-light"} ${
        sidebarExpanded ? "sidebar-expanded" : "sidebar-collapsed"
      }`}
      style={{ backgroundColor: "#0b0b12" }}
    >
      <TopToolbar
        zoom={zoom}
        onZoomChange={onZoomChange}
        onOpenSettings={() => setSettingsOpen(true)}
        onLogoClick={onLogoClick}
        darkMode={isDark}
        onToggleTheme={() => setIsDark && setIsDark((prev) => !prev)}
        annotationsVisible={annotationsVisible}
        onToggleAnnotationsVisibility={() =>
          setAnnotationsVisible((prev) => !prev)
        }
        selectedAnnotation={selectedId}
        onDeleteSelected={() => {
          if (selectedId) {
            setAnnotations((prev) => prev.filter((a) => a.id !== selectedId));
            setSelectedId(null);
          }
        }}
      />
      <div className="workspace">
        <aside
          className={`sidebar left ${sidebarExpanded ? "sidebar-expanded" : "sidebar-collapsed"}`}
        >
          <SidebarTools
            expanded={sidebarExpanded}
            onToggleExpand={() => setSidebarExpanded((prev) => !prev)}
            onDefectScan={handleDefectScan}
            onAutoAnnotate={handleAutoAnnotate}
            onGenerateReport={handleGenerateReport}
            scanning={scanning}
            activeTool={activeTool}
            onToolChange={setActiveTool}
            selectedColor={selectedColor}
            onColorChange={setSelectedColor}
          />
        </aside>

        <main className="workspace-main">
          {!file ? (
            <div className="workspace-upload-area">
              <ImageUpload
                onImageSelect={(f) => {
                  onFileSelect?.(f);
                }}
                onUpload={() => {}}
                loading={false}
              />
            </div>
          ) : (
            <>
              {scanning && (
                <div className="workspace-loading-overlay">
                  <div className="workspace-spinner" />
                  <span>Running defect detection…</span>
                </div>
              )}
              {!scanning && scanError && (
                <div className="workspace-loading-overlay" style={{ background: "rgba(11,11,18,0.82)" }}>
                  <span style={{ color: "#ff6b6b", fontWeight: 600, textAlign: "center", maxWidth: 360 }}>{scanError}</span>
                  <button
                    style={{ marginTop: 12, padding: "6px 18px", borderRadius: 6, border: "none", background: "#387df7", color: "#fff", cursor: "pointer" }}
                    onClick={() => setScanError(null)}
                  >
                    Dismiss
                  </button>
                </div>
              )}
              <ImageCanvas
                zoom={zoom}
                onZoomChange={onZoomChange}
                file={file}
                detections={annotations}
                selectedId={selectedId}
                onSelect={setSelectedId}
                annotationsVisible={annotationsVisible}
                onImageLoad={(width, height) => setImageSize({ width, height })}
                activeTool={activeTool}
                drawColor={selectedColor}
                onAddAnnotation={(partial) => {
                  const id = `manual-${Date.now()}`;
                  setAnnotations((prev) => [
                    ...prev,
                    {
                      id,
                      class_name: "Manual",
                      confidence: 1.0,
                      source: "Manual",
                      status: "confirmed",
                      ...partial,
                    },
                  ]);
                  setActiveTool("Pointer");
                  setSelectedId(id);
                }}
                onAccept={(id) => {
                  setAnnotations((prev) =>
                    prev.map((item) =>
                      item.id === id ? { ...item, status: "confirmed" } : item,
                    ),
                  );
                }}
                onReject={(id) => {
                  setAnnotations((prev) =>
                    prev.map((item) =>
                      item.id === id ? { ...item, status: "rejected" } : item,
                    ),
                  );
                }}
                onErase={(id) => {
                  setAnnotations((prev) => prev.filter((a) => a.id !== id));
                  setSelectedId((prev) => (prev === id ? null : prev));
                }}
              />
            </>
          )}
        </main>

        <RightPanel
          detections={annotations}
          file={file}
          annotations={annotations}
          onAccept={(id) =>
            setAnnotations((prev) =>
              prev.map((a) => (a.id === id ? { ...a, status: "confirmed" } : a))
            )
          }
          onReject={(id) =>
            setAnnotations((prev) =>
              prev.map((a) => (a.id === id ? { ...a, status: "rejected" } : a))
            )
          }
          onDelete={(id) => {
            setAnnotations((prev) => prev.filter((a) => a.id !== id));
            setSelectedId((prev) => (prev === id ? null : prev));
          }}
        />
      </div>

      <div className="bottom-status-bar">
        <div className="status-left">
          <span>{annotationsCount} annotations</span>
          <span>{confirmedCount} confirmed</span>
        </div>
      </div>

      {settingsOpen && (
        <div className="settings-modal-overlay" onClick={handleOverlayClick}>
          <div className="settings-modal-card" onClick={handleModalClick}>
            <div className="settings-modal-header">
              <h2>Settings</h2>
              <button
                className="icon-btn settings-close-button"
                onClick={() => setSettingsOpen(false)}
                title="Close"
              >
                ✕
              </button>
            </div>

            <div className="settings-modal-content">
              <div className="settings-section">
                <h3>Names</h3>
                <div className="settings-field">
                  <label>Marker 1</label>
                  <input
                    type="text"
                    value={settingsValues.marker1}
                    onChange={(e) => updateSetting("marker1", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Marker 2</label>
                  <input
                    type="text"
                    value={settingsValues.marker2}
                    onChange={(e) => updateSetting("marker2", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Marker 3</label>
                  <input
                    type="text"
                    value={settingsValues.marker3}
                    onChange={(e) => updateSetting("marker3", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Marker 4</label>
                  <input
                    type="text"
                    value={settingsValues.marker4}
                    onChange={(e) => updateSetting("marker4", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Marker 5</label>
                  <input
                    type="text"
                    value={settingsValues.marker5}
                    onChange={(e) => updateSetting("marker5", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Marker 6</label>
                  <input
                    type="text"
                    value={settingsValues.eraser}
                    onChange={(e) => updateSetting("eraser", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Marker 7</label>
                  <input
                    type="text"
                    value={settingsValues.edge}
                    onChange={(e) => updateSetting("edge", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Eraser</label>
                  <input
                    type="text"
                    value={settingsValues.surface}
                    onChange={(e) => updateSetting("surface", e.target.value)}
                  />
                </div>
                <div className="settings-field">
                  <label>Class Correct name</label>
                  <input
                    type="text"
                    value={settingsValues.classCorrect}
                    onChange={(e) =>
                      updateSetting("classCorrect", e.target.value)
                    }
                  />
                </div>
                <div className="settings-field">
                  <label>Class Leather name</label>
                  <input
                    type="text"
                    value={settingsValues.classLeather}
                    onChange={(e) =>
                      updateSetting("classLeather", e.target.value)
                    }
                  />
                </div>
              </div>

              <div className="settings-section">
                <h3>Patches</h3>
                <p className="settings-description">
                  Default patch dimensions and stride values
                </p>
                <div className="settings-field">
                  <label>Width</label>
                  <input
                    type="number"
                    min="1"
                    value={settingsValues.patchWidth}
                    onChange={(e) =>
                      updateSetting("patchWidth", e.target.value)
                    }
                  />
                </div>
                <div className="settings-field">
                  <label>Height</label>
                  <input
                    type="number"
                    min="1"
                    value={settingsValues.patchHeight}
                    onChange={(e) =>
                      updateSetting("patchHeight", e.target.value)
                    }
                  />
                </div>
                <div className="settings-field">
                  <label>Horizontal stride</label>
                  <input
                    type="number"
                    min="1"
                    value={settingsValues.patchHorizontalStride}
                    onChange={(e) =>
                      updateSetting("patchHorizontalStride", e.target.value)
                    }
                  />
                </div>
                <div className="settings-field">
                  <label>Vertical stride</label>
                  <input
                    type="number"
                    min="1"
                    value={settingsValues.patchVerticalStride}
                    onChange={(e) =>
                      updateSetting("patchVerticalStride", e.target.value)
                    }
                  />
                </div>
              </div>

              <div className="settings-section">
                <h3>Rating Tolerance</h3>
                <p className="settings-description">
                  Configure the tolerance for assigning rating 1 to a patch.
                  This tolerance represents the percentage of the patch covered
                  by the annotation in relation to its surface. It also limits
                  the minimum presence of the annotation for assigning a rating
                  of 1.
                </p>
                <div className="settings-field">
                  <label>Choose ratio (1–100)</label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={settingsValues.ratingTolerance}
                    onChange={(e) =>
                      updateSetting("ratingTolerance", e.target.value)
                    }
                  />
                </div>
              </div>
            </div>

            <div className="settings-modal-actions">
              <button className="settings-save-button" onClick={saveSettings}>
                Save
              </button>
              <button className="settings-reset-button" onClick={resetSettings}>
                Reset
              </button>
              <button
                className="settings-cancel-button"
                onClick={() => setSettingsOpen(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AnnotationScreen;
