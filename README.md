# Facial Expression Classifier

Deep learning model for classifying facial expressions into 7 categories: angry, disgust, fear, happy, neutral, sad, and surprise.

## Quick Start

```bash
# Install dependencies
uv sync

# Run Jupyter notebook
uv run jupyter notebook src/facial-expression-classifier.ipynb
```

## Data Structure

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Tech Stack

- **PyTorch** - Deep learning framework
- **Jupyter** - Interactive development
- **OpenCV/Pillow** - Image processing
- **Matplotlib/Seaborn** - Visualization

## Why uv?

| **With uv**                      | **Without uv**                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `uv sync`                        | `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| `uv run jupyter`                 | `source .venv/bin/activate && jupyter`                                                 |
| Automatic environment management | Manual activation/deactivation                                                         |
| Faster installation              | Slower dependency resolution                                                           |
