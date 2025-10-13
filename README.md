# 🚀 Quickstart - Installation Guide


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

4. Add a data directory with tif data and a results subdirectory in it.
5. Run `python xctp/preprocess.py` to test the preprocessing as it is implemented now.

6. Install pre-commit hooks (Before committing changes, for developers)

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
│   └── pipeline.py         # orchestrator
└── notebooks/              # experiments + benchmarks
```
