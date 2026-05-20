import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import defaultImage from "./assets/pic.png";
import PanZoomBoard from "./PanZoomBoard";

const CLASS_STYLE_MAP = {
  bolt: { color: "#1fbf5b", badgeBg: "#1fbf5b", text: "#ffffff" },
  defect: { color: "#e23f3d", badgeBg: "#e23f3d", text: "#ffffff" },
  screw: { color: "#387df7", badgeBg: "#387df7", text: "#ffffff" },
  "circuit board": { color: "#e9b709", badgeBg: "#e9b709", text: "#ffffff" },
  manual: { color: "#a855f7", badgeBg: "#a855f7", text: "#ffffff" },
};

function getClassStyle(label) {
  const key = String(label || "").trim().toLowerCase();
  return CLASS_STYLE_MAP[key] || { color: "#387df7", badgeBg: "#387df7", text: "#ffffff" };
}

function getNormalizedRect(bbox, naturalSize) {
  if (!bbox) return null;
  const naturalWidth = Number(naturalSize?.width) || 0;
  const naturalHeight = Number(naturalSize?.height) || 0;
  let x, y, w, h;

  if (Array.isArray(bbox) && bbox.length >= 4) {
    const [a, b, c, d] = bbox.map((v) => Number(v));
    if ([a, b, c, d].some((v) => !Number.isFinite(v))) return null;
    if (c > a && d > b) { x = a; y = b; w = c - a; h = d - b; }
    else { x = a; y = b; w = c; h = d; }
  } else if (typeof bbox === "object") {
    const bx = Number(bbox.x ?? bbox.left ?? bbox.x1);
    const by = Number(bbox.y ?? bbox.top ?? bbox.y1);
    const bw = Number(bbox.w ?? bbox.width);
    const bh = Number(bbox.h ?? bbox.height);
    const bx2 = Number(bbox.x2);
    const by2 = Number(bbox.y2);
    if (Number.isFinite(bx) && Number.isFinite(by)) {
      x = bx; y = by;
      if (Number.isFinite(bw) && Number.isFinite(bh)) { w = bw; h = bh; }
      else if (Number.isFinite(bx2) && Number.isFinite(by2)) { w = bx2 - bx; h = by2 - by; }
    }
  }

  if (![x, y, w, h].every((v) => Number.isFinite(v))) return null;

  if (naturalWidth > 0 && naturalHeight > 0 && x >= 0 && y >= 0 && x <= 1 && y <= 1 && w <= 1 && h <= 1) {
    x *= naturalWidth; y *= naturalHeight; w *= naturalWidth; h *= naturalHeight;
  }
  if (w <= 0 || h <= 0) return null;
  return { x, y, w, h };
}

function clampRectToBounds(rect, bounds) {
  const maxWidth = Number(bounds?.width) || 0;
  const maxHeight = Number(bounds?.height) || 0;
  if (maxWidth <= 0 || maxHeight <= 0) return null;
  const left = Math.max(0, Math.min(rect.x, maxWidth));
  const top = Math.max(0, Math.min(rect.y, maxHeight));
  const width = Math.max(0, Math.min(rect.w, maxWidth - left));
  const height = Math.max(0, Math.min(rect.h, maxHeight - top));
  if (width <= 0 || height <= 0) return null;
  return { left, top, width, height };
}

export default function ImageCanvas({
  zoom = 100,
  onZoomChange,
  file,
  detections = [],
  selectedId,
  onSelect,
  onAccept,
  onReject,
  onErase,
  onImageLoad,
  annotationsVisible = true,
  activeTool = "Pointer",
  drawColor = "#387df7",
  onAddAnnotation,
}) {
  const imgRef = useRef(null);
  const containerRef = useRef(null);
  const initialFitAppliedRef = useRef(false);
  const drawingRef = useRef({ active: false });
  const freehandRef = useRef({ active: false, points: [] });

  const [drawPreview, setDrawPreview] = useState(null);
  const [freehandPoints, setFreehandPoints] = useState(null);
  const [hoveredId, setHoveredId] = useState(null);
  const [objectUrl, setObjectUrl] = useState(null);
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [boardScale, setBoardScale] = useState(zoom / 100);
  const [initialView, setInitialView] = useState({ id: 0, scale: 1, x: 0, y: 0 });
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [imageLoaded, setImageLoaded] = useState(false);

  const isRectTool = activeTool === "Rectangle" || activeTool === "Ellipse";
  const isFreehandTool = activeTool === "Freehand";
  const isDrawingTool = isRectTool || isFreehandTool;
  const isEraserTool = activeTool === "Eraser";

  useEffect(() => {
    setBoardScale(Math.max(0.25, Math.min(3, zoom / 100)));
  }, [zoom]);

  useEffect(() => {
    if (!file) { setObjectUrl(null); return; }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    initialFitAppliedRef.current = false;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const source = file ? objectUrl : defaultImage;

  useEffect(() => {
    if (!imgRef.current) return;
    const updateDimensions = () => {
      const img = imgRef.current;
      if (!img) return;
      setDimensions({ width: img.clientWidth || 0, height: img.clientHeight || 0 });
    };
    const observer = new ResizeObserver(updateDimensions);
    observer.observe(imgRef.current);
    window.addEventListener("resize", updateDimensions);
    return () => { observer.disconnect(); window.removeEventListener("resize", updateDimensions); };
  }, [source]);

  useEffect(() => { setImageLoaded(false); }, [source]);
  useEffect(() => { initialFitAppliedRef.current = false; }, [source]);

  const scale = useMemo(() => {
    if (!naturalSize.width || !naturalSize.height) return { x: 1, y: 1 };
    return { x: dimensions.width / naturalSize.width, y: dimensions.height / naturalSize.height };
  }, [dimensions, naturalSize]);

  const scaleRef = useRef({ x: 1, y: 1 });
  useEffect(() => { scaleRef.current = scale; }, [scale]);

  const naturalSizeRef = useRef(naturalSize);
  useEffect(() => { naturalSizeRef.current = naturalSize; }, [naturalSize]);

  const getCoords = useCallback((e) => {
    const rect = imgRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return null;
    const ns = naturalSizeRef.current;
    const sc = scaleRef.current;
    const imgX = ((e.clientX - rect.left) / rect.width) * ns.width;
    const imgY = ((e.clientY - rect.top) / rect.height) * ns.height;
    return { imgX, imgY, localX: imgX * sc.x, localY: imgY * sc.y };
  }, []);

  // Keyboard shortcuts: Delete/Backspace = delete selected, Escape = deselect
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "Escape") {
        onSelect?.(null);
        return;
      }
      if ((e.key === "Delete" || e.key === "Backspace") && selectedId != null) {
        // Don't interfere with text inputs
        if (document.activeElement?.tagName === "INPUT" || document.activeElement?.tagName === "TEXTAREA") return;
        onErase?.(selectedId);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [selectedId, onSelect, onErase]);

  // Rectangle / Ellipse draw handlers
  const handleRectMouseDown = useCallback((e) => {
    if (!isRectTool || !imgRef.current) return;
    e.stopPropagation();
    e.preventDefault();
    const c = getCoords(e);
    if (!c) return;
    drawingRef.current = { active: true, startImgX: c.imgX, startImgY: c.imgY, startLocalX: c.localX, startLocalY: c.localY };
  }, [isRectTool, getCoords]);

  // Freehand draw handlers
  const handleFreehandMouseDown = useCallback((e) => {
    if (!isFreehandTool || !imgRef.current) return;
    e.stopPropagation();
    e.preventDefault();
    const c = getCoords(e);
    if (!c) return;
    freehandRef.current = { active: true, points: [{ lx: c.localX, ly: c.localY, ix: c.imgX, iy: c.imgY }] };
    setFreehandPoints([{ lx: c.localX, ly: c.localY }]);
  }, [isFreehandTool, getCoords]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      // Rect/Ellipse preview
      if (drawingRef.current.active) {
        const c = getCoords(e);
        if (!c) return;
        const d = drawingRef.current;
        setDrawPreview({
          left: Math.min(d.startLocalX, c.localX),
          top: Math.min(d.startLocalY, c.localY),
          width: Math.abs(c.localX - d.startLocalX),
          height: Math.abs(c.localY - d.startLocalY),
          isEllipse: activeTool === "Ellipse",
        });
        return;
      }
      // Freehand preview
      if (freehandRef.current.active) {
        const c = getCoords(e);
        if (!c) return;
        freehandRef.current.points.push({ lx: c.localX, ly: c.localY, ix: c.imgX, iy: c.imgY });
        setFreehandPoints([...freehandRef.current.points.map((p) => ({ lx: p.lx, ly: p.ly }))]);
      }
    };

    const handleMouseUp = (e) => {
      // Rect/Ellipse finish
      if (drawingRef.current.active) {
        const d = drawingRef.current;
        drawingRef.current = { active: false };
        setDrawPreview(null);
        const c = getCoords(e);
        if (!c) return;
        const x = Math.min(d.startImgX, c.imgX);
        const y = Math.min(d.startImgY, c.imgY);
        const w = Math.abs(c.imgX - d.startImgX);
        const h = Math.abs(c.imgY - d.startImgY);
        if (w > 3 && h > 3 && onAddAnnotation) {
          onAddAnnotation({ bbox: [x, y, w, h], shape: activeTool === "Ellipse" ? "ellipse" : "rect" });
        }
        return;
      }
      // Freehand finish
      if (freehandRef.current.active) {
        const pts = freehandRef.current.points;
        freehandRef.current = { active: false, points: [] };
        setFreehandPoints(null);
        if (pts.length < 2) return;
        const xs = pts.map((p) => p.ix);
        const ys = pts.map((p) => p.iy);
        const x = Math.min(...xs);
        const y = Math.min(...ys);
        const w = Math.max(...xs) - x;
        const h = Math.max(...ys) - y;
        if (w > 3 && h > 3 && onAddAnnotation) {
          onAddAnnotation({ bbox: [x, y, w, h], shape: "freehand" });
        }
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [activeTool, getCoords, onAddAnnotation]);

  const boxes = useMemo(() => {
    return detections.map((det, index) => {
      const rect = getNormalizedRect(det.bbox, naturalSize);
      if (!rect) return null;
      const scaled = { x: rect.x * scale.x, y: rect.y * scale.y, w: rect.w * scale.x, h: rect.h * scale.y };
      const bounded = clampRectToBounds(scaled, dimensions);
      if (!bounded) return null;
      const classStyle = getClassStyle(det.class_name);
      const isDashed = det.source === "SAM Propagated";
      const confidenceRank = typeof det.confidence === "number" ? Math.round(det.confidence * 100) : 0;
      return {
        key: det.id || index,
        id: det.id || index,
        index,
        left: bounded.left,
        top: bounded.top,
        width: bounded.width,
        height: bounded.height,
        label: det.class_name || "",
        confidence: det.confidence,
        source: det.source,
        status: det.status,
        shape: det.shape,
        classStyle,
        isDashed,
        zIndex: 200 + confidenceRank + index,
      };
    }).filter(Boolean);
  }, [detections, naturalSize, scale, dimensions]);

  const freehandSvgPath = useMemo(() => {
    if (!freehandPoints || freehandPoints.length < 2) return null;
    const d = freehandPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.lx} ${p.ly}`).join(" ");
    return d;
  }, [freehandPoints]);

  return (
    <div
      className="image-canvas"
      style={{
        width: "100%",
        height: "100%",
        minHeight: "100vh",
        backgroundColor: "#0b0b12",
        backgroundImage: "radial-gradient(#1F2937 1px, transparent 1px)",
        backgroundSize: "20px 20px",
      }}
    >
      <div className="canvas-wrapper" ref={containerRef}>
        {!source ? (
          <div className="workspace-empty">No image loaded</div>
        ) : (
          <PanZoomBoard
            key={`${source}-${initialView?.id ?? 0}`}
            minZoom={0.25}
            maxZoom={3}
            scale={boardScale}
            disabled={isDrawingTool || isEraserTool}
            centerContent={true}
            initialScale={initialView?.scale ?? boardScale}
            initialX={initialView?.x ?? 0}
            initialY={initialView?.y ?? 0}
            onScaleChange={(nextScale) => {
              setBoardScale(nextScale);
              onZoomChange?.(Math.round(nextScale * 100));
            }}
            style={{
              width: "100%",
              height: "100%",
              minHeight: "100%",
              background: "transparent",
              backgroundColor: "transparent",
              backgroundImage: "none",
              boxShadow: "none",
            }}
          >
            <div
              className="image-canvas-inner"
              style={{
                position: "relative",
                background: "transparent",
                cursor: isDrawingTool ? "crosshair" : isEraserTool ? "cell" : "default",
              }}
              onMouseDown={(e) => {
                if (isRectTool) handleRectMouseDown(e);
                else if (isFreehandTool) handleFreehandMouseDown(e);
              }}
              onClick={(e) => {
                if (e.target === e.currentTarget) onSelect?.(null);
              }}
            >
              {!imageLoaded && <div className="image-skeleton" aria-hidden="true" />}

              <img
                ref={imgRef}
                src={source}
                alt="Annotation"
                className={`canvas-image ${imageLoaded ? "" : "canvas-image--hidden"}`}
                draggable={false}
                onLoad={(event) => {
                  const width = event.target.naturalWidth;
                  const height = event.target.naturalHeight;
                  setNaturalSize({ width, height });
                  setImageLoaded(true);
                  const container = containerRef.current;
                  const containerWidth = (container && container.clientWidth) || event.target.clientWidth || 0;
                  const containerHeight = (container && container.clientHeight) || event.target.clientHeight || 0;
                  if (!initialFitAppliedRef.current && width > 0 && height > 0 && containerWidth > 0 && containerHeight > 0) {
                    const fitScale = Math.max(0.25, Math.min(3, Math.min(containerWidth / width, containerHeight / height) * 0.82));
                    const x = (containerWidth - width * fitScale) / 2;
                    const y = (containerHeight - height * fitScale) / 2;
                    setInitialView({ id: Date.now(), scale: fitScale, x, y });
                    onZoomChange?.(Math.round(fitScale * 100));
                    initialFitAppliedRef.current = true;
                  }
                  onImageLoad?.(width, height);
                }}
              />

              {/* Drawing overlay — rect/ellipse preview */}
              {isRectTool && imageLoaded && drawPreview && (
                <div
                  style={{
                    position: "absolute",
                    left: drawPreview.left,
                    top: drawPreview.top,
                    width: drawPreview.width,
                    height: drawPreview.height,
                    border: `2px dashed ${drawColor}`,
                    borderRadius: drawPreview.isEllipse ? "50%" : 0,
                    boxSizing: "border-box",
                    pointerEvents: "none",
                    background: `${drawColor}22`,
                    zIndex: 500,
                  }}
                />
              )}

              {/* Freehand SVG preview */}
              {isFreehandTool && imageLoaded && freehandSvgPath && (
                <svg
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: dimensions.width || "100%",
                    height: dimensions.height || "100%",
                    pointerEvents: "none",
                    zIndex: 500,
                    overflow: "visible",
                  }}
                >
                  <path d={freehandSvgPath} fill="none" stroke={drawColor} strokeWidth="2" strokeDasharray="4 2" />
                </svg>
              )}

              {/* Annotation boxes */}
              {imageLoaded && annotationsVisible && (
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: dimensions.width || "100%",
                    height: dimensions.height || "100%",
                    pointerEvents: "none",
                  }}
                >
                  {boxes.map((box) => {
                    const isSelected = box.id === selectedId;
                    const isHovered = box.id === hoveredId;
                    const showLabel = isSelected || isHovered;
                    const borderColor = isEraserTool ? "#ef4444" : box.classStyle.color;
                    const isEllipse = box.shape === "ellipse";
                    return (
                      <div
                        key={box.key}
                        style={{
                          position: "absolute",
                          left: box.left,
                          top: box.top,
                          width: box.width,
                          height: box.height,
                          border: `2px ${box.isDashed ? "dashed" : "solid"} ${borderColor}`,
                          borderRadius: isEllipse ? "50%" : 2,
                          boxSizing: "border-box",
                          zIndex: isSelected ? 999 : isHovered ? 998 : box.zIndex,
                          cursor: isDrawingTool ? "crosshair" : isEraserTool ? "cell" : "pointer",
                          outline: isSelected ? `2px solid ${box.classStyle.color}` : "none",
                          outlineOffset: 2,
                          pointerEvents: isDrawingTool ? "none" : "all",
                          opacity: box.status === "rejected" ? 0.35 : 1,
                          background: isSelected ? `${box.classStyle.color}18` : "transparent",
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (isEraserTool) onErase?.(box.id);
                          else onSelect?.(box.id);
                        }}
                        onMouseEnter={() => setHoveredId(box.id)}
                        onMouseLeave={() => setHoveredId(null)}
                      >
                        {showLabel && (
                          <div
                            style={{
                              position: "absolute",
                              top: box.top < 28 ? box.height + 2 : -26,
                              left: 0,
                              background: box.classStyle.badgeBg,
                              color: box.classStyle.text,
                              padding: "2px 6px",
                              borderRadius: 3,
                              fontSize: 11,
                              fontWeight: 600,
                              whiteSpace: "nowrap",
                              lineHeight: 1.4,
                              pointerEvents: "none",
                              zIndex: 1000,
                            }}
                          >
                            {box.label}
                            {typeof box.confidence === "number" ? ` ${Math.round(box.confidence * 100)}%` : ""}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </PanZoomBoard>
        )}

        <div className="image-info-overlay">
          {naturalSize?.width || 0} x {naturalSize?.height || 0} px ·{" "}
          {Math.round((boardScale || 0) * 100)}%
        </div>
      </div>
    </div>
  );
}
