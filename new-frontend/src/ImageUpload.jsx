import { useState } from "react";

function ImageUpload({ onImageSelect, onUpload, loading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  
  const preloadImage = (file) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => URL.revokeObjectURL(url);
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setError(null);
      preloadImage(file);
      onImageSelect(file);
    } else {
      setError("Please select a valid image file");
      setSelectedFile(null);
      onImageSelect(null);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div className="upload-section">
      <div className="file-input-container">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          id="file-input"
          className="file-input"
        />
        <label htmlFor="file-input" className="file-input-label">
          {selectedFile ? selectedFile.name : "Choose an image file"}
        </label>
      </div>

      <button
        onClick={handleUpload}
        disabled={!selectedFile || loading}
        className="upload-button"
      >
        {loading ? "Processing..." : "Detect Defects"}
      </button>

      {error && <div className="error-message">{error}</div>}
    </div>
  );
}

export default ImageUpload;
