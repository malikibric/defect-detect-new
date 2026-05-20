function ImagePreview({ imageFile }) {
  if (!imageFile) return null;

  return (
    <div className="image-preview">
      <h3>Selected Image:</h3>
      <img
        src={URL.createObjectURL(imageFile)}
        alt="Selected file"
        className="preview-image"
      />
    </div>
  );
}

export default ImagePreview;
