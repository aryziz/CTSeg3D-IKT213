# 🚀 Quickstart

hallo

1. Clone the repo

```bash
git clone https://github.com/aryziz/CTSeg3D-IKT213
cd CTSeg3D-IKT213
```

2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or
.venv\Scripts\activate      # Windows
```

3. Install deps

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Install pre-commit hooks

```
pip install pre-commit
pre-commit install
```


# 📂 Project Structure

```
CTSeg3D-IKT213/
├── README.md
├── requirements.txt
├── run_pipeline.py         # CLI entrypoint
├── .pre-commit-config.yaml # Pre-commit setup file
├── setup.cfg               # Configuring defaults & metadata
├── config/
│   └── default.yaml        # default params
├── xctp/                   # source package
│   ├── preprocess.py       # normalization, denoise
│   ├── seeds.py            # auto-seeding
│   ├── segment.py          # graph-cut, random walker
│   ├── postprocess.py      # cleanup
│   ├── metrics.py          # porosity, tortuosity
│   └── pipeline.py         # orchestrator
├── notebooks/              # experiments + benchmarks
└── tests/                  # pytest unit tests
```
