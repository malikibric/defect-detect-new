import { useEffect, useMemo, useRef, useState } from "react";
import {
  MousePointer2,
  Square,
  Circle,
  PenTool,
  Hexagon,
  Eraser,
  Sparkles,
  FileText,
  Search,
  Plus,
  Minus,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import "./App.css";

function SidebarTools({
  expanded = true,
  onToggleExpand,
  onDefectScan,
  onAutoAnnotate,
  onGenerateReport,
  scanning = false,
  activeTool: activeToolProp,
  onToolChange,
  selectedColor: selectedColorProp,
  onColorChange,
  thickness: thicknessProp,
  onThicknessChange,
}) {
  const [localActiveTool, setLocalActiveTool] = useState("Pointer");
  const [localSelectedColor, setLocalSelectedColor] = useState("#1fbf5b");
  const [localThickness, setLocalThickness] = useState("20px");
  const [thicknessOpen, setThicknessOpen] = useState(false);

  const activeTool = activeToolProp ?? localActiveTool;
  const selectedColor = selectedColorProp ?? localSelectedColor;
  const thickness = thicknessProp ?? localThickness;

  const setActiveTool = (v) => { setLocalActiveTool(v); onToolChange?.(v); };
  const setSelectedColor = (v) => { setLocalSelectedColor(v); onColorChange?.(v); };
  const setThickness = (v) => { setLocalThickness(v); onThicknessChange?.(v); };
  const pickerRef = useRef(null);

  const toolSections = useMemo(
    () => [
      {
        title: "ANNOTATE",
        items: [
          {
            key: "Pointer",
            label: "Pointer",
            icon: <MousePointer2 size={18} />,
          },
          { key: "Rectangle", label: "Rectangle", icon: <Square size={18} /> },
          { key: "Ellipse", label: "Ellipse", icon: <Circle size={18} /> },
          { key: "Freehand", label: "Freehand", icon: <PenTool size={18} /> },
          { key: "Eraser", label: "Eraser", icon: <Eraser size={18} /> },
        ],
      },
      {
        title: "PALETTE",
        palette: true,
        paletteColors: [
          "#1fbf5b",
          "#e23f3d",
          "#387df7",
          "#e9b709",
          "#f48924",
          "#7552cc",
          "#ff66c4",
        ],
      },
      {
        title: "THICKNESS",
        picker: true,
        value: "5px",
      },
      {
        title: "AI",
        items: [
          {
            key: "SmartPropagate",
            label: "Auto Annotate",
            icon: <Sparkles size={18} />,
          },
          {
            key: "DefectScan",
            label: "Defect Scan",
            icon: <Search size={18} />,
          },
          {
            key: "PatchExtract",
            label: "Report",
            icon: <FileText size={18} />,
          },
        ],
      },
    ],
    [],
  );

  const topSections = useMemo(() => toolSections.slice(0, 4), [toolSections]);
  const bottomSections = useMemo(() => toolSections.slice(4), [toolSections]);

  const renderSection = (section) => (
    <div key={section.title} className="tool-section">
      <div className="tool-section-title">{section.title}</div>
      <div className="tools-group">
        {section.items?.map((tool) => {
          const isActive = tool.key === activeTool;
          return (
            <button
              key={tool.key}
              type="button"
              className={`tool-button ${isActive ? "active" : ""}`}
              onClick={() => {
                if (tool.key === "PatchExtract") {
                  if (onGenerateReport) onGenerateReport();
                  return;
                }
                setActiveTool(tool.key);
                if (tool.key === "DefectScan" && onDefectScan) onDefectScan();
                if (tool.key === "SmartPropagate" && onAutoAnnotate) onAutoAnnotate();
              }}
              disabled={scanning && (tool.key === "DefectScan" || tool.key === "SmartPropagate")}
              title={expanded ? undefined : tool.label}
            >
              <span className="tool-icon" aria-hidden="true">
                {tool.icon}
              </span>
              {expanded ? (
                <span className="tool-label">{tool.label}</span>
              ) : null}
            </button>
          );
        })}

        {section.palette ? renderPalette(section.paletteColors) : null}
        {section.picker ? renderPicker(section.value) : null}
      </div>
    </div>
  );

  const thicknessOptions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65];

  const renderPalette = (colors) => {
    return (
      <div className="palette">
        {colors.map((color) => {
          const isSelected = color === selectedColor;
          return (
            <button
              key={color}
              type="button"
              className={`palette-swatch${isSelected ? " selected" : ""}`}
              style={{ background: color }}
              aria-label={`Color ${color}`}
              onClick={() => setSelectedColor(color)}
            />
          );
        })}
      </div>
    );
  };

  const renderPicker = () => {
    const numValue = parseInt(thickness, 10);
    const currentIndex = thicknessOptions.indexOf(numValue);

    const handleDecrement = (e) => {
      e.stopPropagation();
      if (currentIndex > 0) {
        setThickness(`${thicknessOptions[currentIndex - 1]}px`);
      }
    };

    const handleIncrement = (e) => {
      e.stopPropagation();
      if (currentIndex < thicknessOptions.length - 1) {
        setThickness(`${thicknessOptions[currentIndex + 1]}px`);
      }
    };

    if (!expanded) {
      return (
        <div className="picker-control-group-collapsed">
          <button
            type="button"
            className="picker-arrow-button picker-arrow-top"
            onClick={handleIncrement}
            title="Increase thickness"
            aria-label="Increase thickness"
            disabled={currentIndex === thicknessOptions.length - 1}
          >
            <Plus size={16} />
          </button>
          <div
            ref={pickerRef}
            className="picker-control-collapsed"
            onClick={() => setThicknessOpen((prev) => !prev)}
            role="button"
            aria-haspopup="listbox"
            aria-expanded={thicknessOpen}
            tabIndex={0}
          >
            <span
              className="picker-value-collapsed"
              style={{ fontWeight: 700 }}
            >
              {thickness}
            </span>
            {thicknessOpen ? (
              <div className="picker-dropdown" role="listbox">
                {thicknessOptions.map((size) => {
                  const value = `${size}px`;
                  const isSelected = value === thickness;
                  return (
                    <button
                      key={value}
                      type="button"
                      className={`picker-item${isSelected ? " selected" : ""}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        setThickness(value);
                        setThicknessOpen(false);
                      }}
                    >
                      <span className="picker-item-label">{value}</span>
                      {isSelected ? (
                        <span className="picker-item-check">✓</span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
          <button
            type="button"
            className="picker-arrow-button picker-arrow-bottom"
            onClick={handleDecrement}
            title="Decrease thickness"
            aria-label="Decrease thickness"
            disabled={currentIndex === 0}
          >
            <Minus size={16} />
          </button>
        </div>
      );
    }

    return (
      <div className="picker-control-group">
        <button
          type="button"
          className="picker-arrow-button picker-arrow-left"
          onClick={handleDecrement}
          title="Decrease thickness"
          aria-label="Decrease thickness"
          disabled={currentIndex === 0}
        >
          <Minus size={16} />
        </button>
        <div
          ref={pickerRef}
          className="picker-control"
          onClick={() => setThicknessOpen((prev) => !prev)}
          role="button"
          aria-haspopup="listbox"
          aria-expanded={thicknessOpen}
          tabIndex={0}
        >
          <span className="picker-value" style={{ fontWeight: 400 }}>
            {thickness}
          </span>
          <span className="picker-arrows" aria-hidden>
            {thicknessOpen ? (
              <ChevronUp size={16} />
            ) : (
              <ChevronDown size={16} />
            )}
          </span>
          {thicknessOpen ? (
            <div className="picker-dropdown" role="listbox">
              {thicknessOptions.map((size) => {
                const value = `${size}px`;
                const isSelected = value === thickness;
                return (
                  <button
                    key={value}
                    type="button"
                    className={`picker-item${isSelected ? " selected" : ""}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      setThickness(value);
                      setThicknessOpen(false);
                    }}
                  >
                    <span className="picker-item-label">{value}</span>
                    {isSelected ? (
                      <span className="picker-item-check">✓</span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          ) : null}
        </div>
        <button
          type="button"
          className="picker-arrow-button picker-arrow-right"
          onClick={handleIncrement}
          title="Increase thickness"
          aria-label="Increase thickness"
          disabled={currentIndex === thicknessOptions.length - 1}
        >
          <Plus size={16} />
        </button>
      </div>
    );
  };

  useEffect(() => {
    if (!thicknessOpen) return;

    const handleClickOutside = (event) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target)) {
        setThicknessOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [thicknessOpen]);

  return (
    <div
      className={`tools-panel ${expanded ? "expanded" : "collapsed"}`}
      role="toolbar"
      aria-label="Annotation tools"
    >
      <div className="tools-header">
        <button
          type="button"
          className="tools-toggle"
          aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
          onClick={onToggleExpand}
          title={expanded ? "Collapse sidebar" : "Expand sidebar"}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path
              d={expanded ? "M12 6l-6 6 6 6" : "M12 6l6 6-6 6"}
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>

      <div className="sidebar-top">
        {topSections.map((section) => renderSection(section))}
      </div>

      <div className="sidebar-bottom">
        {bottomSections.map((section) => renderSection(section))}
      </div>
    </div>
  );
}

export default SidebarTools;
