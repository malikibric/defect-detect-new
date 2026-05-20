import "./App.css";
import logoColor from "./assets/logo color.png";
import nameColor from "./assets/name color.png";

function HomeScreen({
  onLoadImage,
  onLoadMultiple,
  onLoadSession,
  error,
  isDark,
  setIsDark,
}) {
  const toggleTheme = () => {
    if (setIsDark) setIsDark((current) => !current);
  };

  const Card = ({ title, description, onClick, icon }) => (
    <button type="button" className="home-card" onClick={onClick}>
      <div className="home-card-icon" aria-hidden="true">
        {icon}
      </div>
      <div className="home-card-body">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </button>
  );

  return (
    <div className={`home-screen home-screen--${isDark ? "dark" : "light"}`}>
      <button
        type="button"
        className="home-theme-toggle"
        onClick={toggleTheme}
        aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
      >
        {isDark ? (
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M12 4V2M12 22v-2M4 12H2m20 0h-2M6.34 6.34 4.93 4.93m14.14 14.14-1.41-1.41M17.66 6.34l1.41-1.41M6.34 17.66l-1.41 1.41"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
            <circle
              cx="12"
              cy="12"
              r="4"
              stroke="currentColor"
              strokeWidth="1.8"
            />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </button>

      <div className="home-hero">
        <div className="home-logo" aria-hidden="true">
          <img src={logoColor} alt="" className="home-logo-image" />
        </div>
        <img src={nameColor} alt="Annotic" className="home-title-image" />
        <p className="home-subtitle">AI-Assisted Annotation Platform</p>
      </div>

      <div className="home-cards">
        <Card
          title="Load Image"
          description="Open a single image for annotation"
          onClick={onLoadImage}
          icon={
            <svg viewBox="0 0 24 24" fill="none" className="home-icon-svg">
              <rect
                x="4"
                y="5"
                width="16"
                height="14"
                rx="2"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <path
                d="m7.5 15 3.2-3.6 2.6 2.7 3.2-3.6"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle cx="9" cy="9" r="1.2" fill="currentColor" />
            </svg>
          }
        />
        <Card
          title="Load Multiple Images"
          description="Open a folder with multiple images"
          onClick={onLoadMultiple}
          icon={
            <svg viewBox="0 0 24 24" fill="none" className="home-icon-svg">
              <path
                d="M3.5 8.5A2.5 2.5 0 0 1 6 6h3.2l1.6 1.8H18A2.5 2.5 0 0 1 20.5 10.3V16A2.5 2.5 0 0 1 18 18.5H6A2.5 2.5 0 0 1 3.5 16Z"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinejoin="round"
              />
            </svg>
          }
        />
      </div>

      {error ? <div className="home-error">{error}</div> : null}

      <div className="home-hint">
        Supports PNG, JPG, TIFF, BMP · Drag and drop anywhere to begin
      </div>
    </div>
  );
}

export default HomeScreen;
