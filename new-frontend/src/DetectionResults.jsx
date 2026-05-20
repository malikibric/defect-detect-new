function DetectionResults({ results }) {
  if (!results) return null;

  return (
    <div className="results">
      <h2>Detection Results</h2>
      <div className="results-summary">
        <p>Total detections: {results.total_detections || 0}</p>
        <p>Processing time: {results.processing_time_seconds?.toFixed(2)}s</p>
      </div>

      {results.detections && results.detections.length > 0 ? (
        <div className="detections-list">
          <h3>Detected Defects:</h3>
          {results.detections.map((detection, index) => (
            <div key={index} className="detection-item">
              <p>
                <strong>Class:</strong> {detection.class_name}
              </p>
              <p>
                <strong>Confidence:</strong>{" "}
                {(detection.confidence * 100).toFixed(1)}%
              </p>
              <p>
                <strong>Bounding Box:</strong> [{detection.bbox.join(", ")}]
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p>No defects detected in the image.</p>
      )}
    </div>
  );
}

export default DetectionResults;
