import { useCallback, useEffect, useRef, useState } from "react";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export default function PanZoomBoard({
  minZoom = 0.5,
  maxZoom = 3,
  zoomStep = 0.1,
  scale,
  onScaleChange,
  centerContent = true,
  initialX = 0,
  initialY = 0,
  initialScale = 1,
  className,
  style,
  children,
  disabled = false,
}) {
  const isScaleControlled = typeof scale === "number";
  const initialResolvedScale = clamp(
    isScaleControlled ? scale : initialScale,
    minZoom,
    maxZoom,
  );

  const [isDragging, setIsDragging] = useState(false);
  const [view, setView] = useState({
    x: -740,
    y: 150,
    scale: initialResolvedScale,
  });

  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const isDraggingRef = useRef(false);
  const hasInteractedRef = useRef(false);
  const dragStartRef = useRef({ mouseX: 0, mouseY: 0, viewX: 0, viewY: 0 });

  const rafRef = useRef(null);
  const pendingViewRef = useRef(null);
  const latestViewRef = useRef(view);

  useEffect(() => {
    latestViewRef.current = view;
  }, [view]);

  useEffect(() => {
    if (!isScaleControlled) return;
    const nextScale = clamp(scale, minZoom, maxZoom);
    setView((prev) =>
      prev.scale === nextScale ? prev : { ...prev, scale: nextScale },
    );
  }, [isScaleControlled, maxZoom, minZoom, scale]);

  // Centering logic disabled - using hardcoded initial x/y values instead

  const flushViewUpdate = useCallback(() => {
    rafRef.current = null;
    if (!pendingViewRef.current) return;
    setView(pendingViewRef.current);
    pendingViewRef.current = null;
  }, []);

  const scheduleViewUpdate = useCallback(
    (nextView) => {
      pendingViewRef.current = nextView;
      if (rafRef.current !== null) return;
      rafRef.current = requestAnimationFrame(flushViewUpdate);
    },
    [flushViewUpdate],
  );

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const handleMouseDown = useCallback((event) => {
    if (disabled || event.button !== 0) return;

    event.preventDefault();
    hasInteractedRef.current = true;
    isDraggingRef.current = true;
    setIsDragging(true);

    const current = latestViewRef.current;
    dragStartRef.current = {
      mouseX: event.clientX,
      mouseY: event.clientY,
      viewX: current.x,
      viewY: current.y,
    };
  }, []);

  const handleMouseMove = useCallback(
    (event) => {
      if (!isDraggingRef.current) return;

      const start = dragStartRef.current;
      const dx = event.clientX - start.mouseX;
      const dy = event.clientY - start.mouseY;

      scheduleViewUpdate({
        x: start.viewX + dx,
        y: start.viewY + dy,
        scale: latestViewRef.current.scale,
      });
    },
    [scheduleViewUpdate],
  );

  const stopDragging = useCallback(() => {
    isDraggingRef.current = false;
    setIsDragging(false);
  }, []);

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", stopDragging);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", stopDragging);
    };
  }, [handleMouseMove, stopDragging]);

  const handleWheel = useCallback(
    (event) => {
      event.preventDefault();

      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const cursorX = event.clientX - rect.left;
      const cursorY = event.clientY - rect.top;

      const current = latestViewRef.current;
      const zoomDirection = event.deltaY < 0 ? 1 : -1;
      const zoomFactor = 1 + zoomDirection * zoomStep;
      const nextScale = clamp(current.scale * zoomFactor, minZoom, maxZoom);

      if (nextScale === current.scale) return;

      // Keep zoom centered around the cursor point in the viewport.
      const worldX = (cursorX - current.x) / current.scale;
      const worldY = (cursorY - current.y) / current.scale;
      const nextX = cursorX - worldX * nextScale;
      const nextY = cursorY - worldY * nextScale;

      hasInteractedRef.current = true;
      if (onScaleChange) {
        onScaleChange(nextScale);
      }
      scheduleViewUpdate({ x: nextX, y: nextY, scale: nextScale });
    },
    [maxZoom, minZoom, onScaleChange, scheduleViewUpdate, zoomStep],
  );

  return (
    <div
      ref={containerRef}
      className={className}
      onMouseDown={handleMouseDown}
      onWheel={handleWheel}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        cursor: disabled ? "crosshair" : isDragging ? "grabbing" : "grab",
        touchAction: "none",
        userSelect: "none",
        backgroundImage:
          "linear-gradient(to right, #1F2937 1px, transparent 1px), linear-gradient(to bottom, #1F2937 1px, transparent 1px)",
        backgroundSize: "24px 24px",
        ...style,
      }}
    >
      <div
        ref={contentRef}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          transformOrigin: "50% 50%",
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
          width: "max-content",
          height: "max-content",
          willChange: "transform",
        }}
      >
        {children}
      </div>
    </div>
  );
}
