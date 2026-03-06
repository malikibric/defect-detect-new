# ML Backend for DefectDetect Application

Advanced AI-powered backend service providing automated label propagation, quality assurance, smart patching, and synthetic data generation for defect detection workflows.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Endpoints Reference](#api-endpoints-reference)
- [Service Documentation](#service-documentation)
- [Model Downloads](#model-downloads)
- [Configuration](#configuration)
- [Testing](#testing)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                           (main.py)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬──────────────┐
         │               │               │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
    │   SAM   │    │   QA    │    │  Patch  │   │Synthetic│
    │ Router  │    │ Router  │    │ Router  │   │ Router  │
    └────┬────┘    └────┬────┘    └────┬────┘   └────┬────┘
         │               │               │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
    │   SAM   │    │   QA    │    │  Patch  │   │Synthetic│
    │ Service │    │ Service │    │ Service │   │ Service │
    └────┬────┘    └────┬────┘    └────┬────┘   └────┬────┘
         │               │               │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
    │  SAM    │    │ YOLO26  │    │  CLIP   │   │ Stable  │
    │  Model  │    │  Model  │    │  Model  │   │Diffusion│
    └─────────┘    └─────────┘    └─────────┘   └─────────┘
```

### Components

- **Routers**: FastAPI route handlers for each AI service
- **Services**: Business logic implementing ML algorithms
- **Models**: Pydantic schemas for request/response validation
- **ML Models**: Pre-trained deep learning models (loaded on demand)

---

## ✨ Features

### 1. **SAM Label Propagation** 🎯
Automatically propagate annotations across images using Meta's Segment Anything Model with few-shot learning.

### 2. **AI-Driven Quality Assurance** ✅
Validate human annotations using YOLO26, identifying missed defects and sizing inconsistencies.

### 3. **Smart Patching & Clustering** 🧩
Extract defect patches with AI-suggested sizing and cluster by severity using CLIP embeddings.

### 4. **Synthetic Data Generation** 🎨
Generate realistic defect variations with different lighting and severity using Stable Diffusion.

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for performance)
- 16GB+ RAM (32GB recommended for Stable Diffusion)

### Step 1: Clone Repository

```bash
cd ml_backend
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 10-15 minutes due to large ML libraries (PyTorch, Transformers, etc.)

---

## 🎮 Running the Server

### Development Mode

```bash
# From ml_backend directory
python main.py
```

or

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/api/health

---

## 📚 API Endpoints Reference

| Endpoint | Method | Description | Input | Output |
|----------|--------|-------------|-------|--------|
| `/` | GET | Health check | None | Status message |
| `/api/health` | GET | Detailed system health | None | System info + model status |
| `/api/sam/propagate` | POST | Propagate annotations with SAM | Image path + seed annotations | Proposed annotations |
| `/api/qa/check` | POST | Run QA validation | Image path + human annotations | QA report (missed/confirmed/warnings) |
| `/api/patch/extract` | POST | Extract defect patches | Image path + annotations | Base64 patches with metadata |
| `/api/patch/cluster` | POST | Cluster patches by severity | Patches from /extract | Patches grouped by severity |
| `/api/synthetic/generate` | POST | Generate synthetic variations | Image path + annotation | Paths to generated images |

---

## 🔬 Service Documentation

### 1. SAM Label Propagation Service

**Purpose**: Automatically find similar defects across an image using 2-3 manual examples.

**How it works**:
1. User annotates 2-3 examples of a defect type
2. SAM extracts features from these seed examples
3. System generates candidate regions using SAM's segmentation
4. Each candidate is compared with seeds using similarity metrics
5. Only regions with similarity > 75% are proposed

**Key Parameters**:
- `seed_annotations`: 2-10 seed examples (more = better accuracy)
- `similarity_threshold`: 0.0-1.0 (default 0.75, higher = stricter)

**Use Cases**:
- Rapid annotation of repetitive defects (cracks, scratches)
- Reducing manual annotation time by 70-90%
- Ensuring consistent annotation style

---

### 2. Quality Assurance Service

**Purpose**: Validate human annotations using AI to catch errors and omissions.

**How it works**:
1. Runs YOLO26 defect detection independently
2. Compares YOLO predictions with human annotations using IoU
3. Identifies three categories:
   - **Missed Defects**: AI found, human didn't (potential errors)
   - **Size Warnings**: Human bbox size deviates >40% from median (inconsistency)
   - **Confirmed**: High IoU match with AI (validated correct)

**Key Metrics**:
- **IoU Threshold**: 0.5 default (50% overlap = match)
- **Size Deviation**: 40% from median triggers warning

**Use Cases**:
- Quality control for annotation teams
- Training data validation before ML training
- Identifying annotator bias or fatigue

---

### 3. Smart Patching & Clustering Service

**Purpose**: Extract defect patches and automatically group by severity.

**Patch Extraction**:
- Analyzes all annotation dimensions
- Calculates optimal patch size = median_diagonal × 1.5
- Extracts patches with padding for context
- Returns base64-encoded images

**Clustering Algorithm**:
1. Extract CLIP embeddings (512-dim semantic vectors)
2. Run K-Means clustering (k=3 by default)
3. Map clusters to severity based on average defect size:
   - Largest → Severe Defect
   - Medium → Minor Defect  
   - Smallest → Clean/No Defect

**Use Cases**:
- Creating defect severity datasets
- Prioritizing critical defects for review
- Balancing training data across severity levels

---

### 4. Synthetic Data Generation Service

**Purpose**: Generate realistic defect variations for data augmentation.

**How it works**:
1. Extracts defect region from annotation
2. Creates inpainting mask with 10% expansion
3. Builds detailed prompts: `"{severity} {defect_type} on {material}, {lighting}, photorealistic"`
4. Runs Stable Diffusion inpainting with:
   - 50 inference steps
   - Guidance scale 7.5
   - Unique seed per variation
5. Saves generated images to `output/synthetic/`

**Variation Strategy**:
- Cycles through lighting: dark, bright, side-lit, natural
- Cycles through severity: minor, moderate, severe
- Generates diverse combinations

**Use Cases**:
- Augmenting rare defect classes (corrosion, delamination)
- Creating balanced datasets (equal examples per class)
- Testing model robustness across lighting conditions
- Simulating defects that are expensive to reproduce

---

## 📦 Model Downloads

The backend requires several pre-trained models. Most download automatically via HuggingFace, but SAM requires manual download:

### Required Downloads

1. **Segment Anything Model (SAM)**:
   ```bash
   mkdir -p models
   cd models
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

2. **YOLO26** (auto-downloads on first use):
    - Pretrained: `yolo26n.pt`
    - Or set `YOLO_MODEL=models/yolo26n.pt` in `.env`

3. **CLIP** (auto-downloads):
   - Model: `openai/clip-vit-base-patch32`

4. **Stable Diffusion** (auto-downloads, ~5GB):
   - Model: `runwayml/stable-diffusion-inpainting`

### Model Storage

- Default location: `models/`
- Huggingface cache: `~/.cache/huggingface/`
- Total disk space required: ~15GB

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Device configuration
DEVICE=cuda  # or "cpu"

# Model paths
SAM_CHECKPOINT=models/sam_vit_h_4b8939.pth
YOLO_MODEL=models/yolo26n.pt

# API settings
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Output directories
SYNTHETIC_OUTPUT_DIR=output/synthetic
PATCH_OUTPUT_DIR=output/patches
```

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU) | NVIDIA RTX 3060+ (8GB VRAM) |
| RAM | 16GB | 32GB |
| Storage | 20GB | 50GB |
| CPU | 4 cores | 8+ cores |

### Performance Notes

- **GPU vs CPU**: 
  - SAM: 2s (GPU) vs 15s (CPU)
  - YOLO: 0.5s (GPU) vs 3s (CPU)
  - CLIP: 0.1s (GPU) vs 1s (CPU)
  - Stable Diffusion: 5s/image (GPU) vs 60s/image (CPU)

---

## 🧪 Testing

### Run All Tests

```bash
cd tests
pytest -v
```

### Run Specific Test Suite

```bash
pytest tests/test_sam_service.py -v
pytest tests/test_qa_service.py -v
pytest tests/test_patch_service.py -v
pytest tests/test_synthetic_service.py -v
```

### Test Coverage

```bash
pytest --cov=services --cov=routers --cov-report=html
```

View coverage report: `htmlcov/index.html`

---

## 📊 Example Usage

### 1. Propagate Annotations with SAM

```python
import requests

response = requests.post(
    "http://localhost:8000/api/sam/propagate",
    json={
        "image_path": "data/defect_image.jpg",
        "seed_annotations": [
            {
                "bbox": [100, 100, 50, 50],
                "class_name": "crack",
                "confidence": 1.0
            },
            {
                "bbox": [300, 200, 45, 52],
                "class_name": "crack",
                "confidence": 1.0
            }
        ],
        "similarity_threshold": 0.75
    }
)

result = response.json()
print(f"Proposed {result['total_proposed']} annotations")
```

### 2. Run QA Check

```python
response = requests.post(
    "http://localhost:8000/api/qa/check",
    json={
        "image_path": "data/defect_image.jpg",
        "human_annotations": [...],  # Your annotations
        "iou_threshold": 0.5
    }
)

qa_report = response.json()
print(f"Confirmed: {len(qa_report['confirmed'])}")
print(f"Missed: {len(qa_report['missed_defects'])}")
print(f"Warnings: {len(qa_report['size_warnings'])}")
```

### 3. Extract and Cluster Patches

```python
# Step 1: Extract patches
extract_response = requests.post(
    "http://localhost:8000/api/patch/extract",
    json={
        "image_path": "data/defect_image.jpg",
        "annotations": [...]
    }
)

patches = extract_response.json()['patches']

# Step 2: Cluster by severity
cluster_response = requests.post(
    "http://localhost:8000/api/patch/cluster",
    json={
        "patches": patches,
        "num_clusters": 3
    }
)

results = cluster_response.json()
print(f"Severe: {len(results['severe'])}")
print(f"Minor: {len(results['minor'])}")
print(f"Clean: {len(results['clean'])}")
```

### 4. Generate Synthetic Defects

```python
response = requests.post(
    "http://localhost:8000/api/synthetic/generate",
    json={
        "image_path": "data/defect_image.jpg",
        "annotation": {
            "bbox": [150, 150, 60, 60],
            "class_name": "crack"
        },
        "num_variations": 15,
        "lighting_conditions": ["dark", "bright", "side-lit"],
        "severity_levels": ["minor", "moderate", "severe"]
    }
)

result = response.json()
print(f"Generated {result['total_generated']} variations")
print(f"Output directory: {result['output_directory']}")
```

---

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size or use CPU mode
   - Enable attention slicing (already enabled for Stable Diffusion)

2. **SAM model not loading**:
   - Verify checkpoint downloaded to `models/sam_vit_h_4b8939.pth`
   - Check file size: should be ~2.4GB

3. **Slow performance**:
   - Ensure GPU is detected: check `/api/health` endpoint
   - Update CUDA drivers
    - Use smaller models (e.g., YOLO26n for lower memory usage)

4. **Import errors**:
   - Reinstall dependencies: `pip install -r requirements.txt --upgrade`
   - Check Python version: requires 3.8+

---

## 📝 License

This ML backend is part of the DefectDetect Application project.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📞 Support

For issues and questions:
- GitHub Issues: [DefectDetect Repository]
- Documentation: http://localhost:8000/api/docs (when running)

---

**Built with ❤️ using FastAPI, PyTorch, and Hugging Face Transformers**
