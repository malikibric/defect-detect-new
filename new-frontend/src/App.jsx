import { useState, useEffect, useRef } from "react";
import HomeScreen from "./HomeScreen";
import AnnotationScreen from "./AnnotationScreen";
import "./App.css";

function App() {
  const [view, setView] = useState("home");
  const [isDark, setIsDark] = useState(
    () => localStorage.getItem("theme") !== "light",
  );
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem("theme", isDark ? "dark" : "light");
    } catch (e) {
      // ignore storage errors
    }
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  const handleBackHome = () => {
    setView("home");
    setSelectedFile(null);
  };

  const handleFilePickerChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = "";
    setSelectedFile(file);
    setView("annotation");
  };

  const handleImageSelect = (file) => {
    setSelectedFile(file);
  };

  return (
    <div className="app" style={{ backgroundColor: "#0b0b12" }}>
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFilePickerChange}
      />
      {view === "home" ? (
        <HomeScreen
          onLoadImage={() => fileInputRef.current?.click()}
          onLoadMultiple={() => fileInputRef.current?.click()}
          onLoadSession={() => fileInputRef.current?.click()}
          isDark={isDark}
          setIsDark={setIsDark}
        />
      ) : (
        <AnnotationScreen
          file={selectedFile}
          onBack={handleBackHome}
          onLogoClick={handleBackHome}
          onFileSelect={handleImageSelect}
          isDark={isDark}
          setIsDark={setIsDark}
        />
      )}
    </div>
  );
}

export default App;
