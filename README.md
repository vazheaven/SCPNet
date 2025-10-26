# Statistics-Enhanced Cross-Domain Perception Network for Spaceborne Infrared Tiny Ship Detection

This repository contains **SCPNet**, our lightweight detection framework for spaceborne thermal infrared tiny-ship detection.  
It ships with a **local, modified `ultralytics` package** (installed in editable mode) that includes our custom modules (e.g., `SEM`, `CDSAM`, `SFCAM`), model configs, and dataset configs.

> If you can run Ultralytics YOLO, you can run SCPNet — the CLI is the same (`yolo ...`), but it loads this repo’s local sources.

# Dataset
SDG-TISD: https://doi.org/10.6084/m9.figshare.30104404.v1

---

## 1) Requirements

- **OS:** Windows 10/11 or Linux  
- **Python:** 3.8–3.11 (recommended: **3.10**)  
- **GPU:** NVIDIA (CUDA) recommended  
- **CUDA:** 11.8 or 12.x (match your driver)

### 1.1 Create a fresh Python/Conda env (recommended)

```bash
# Conda example
conda create -n scpnet python=3.10 -y
conda activate scpnet
```

> If you prefer plain `venv`, that works too.

### 1.2 Install PyTorch (match CUDA)

Pick the wheel that matches your CUDA. Examples:

```bash
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# or CUDA 11.8
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# or CPU-only (for testing only)
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

---

## 2) Get the code & Python deps

```bash
git clone https://github.com/vazheaven/SCPNet.git
cd SCPNet
pip install -r requirements.txt
```

> `requirements.txt` contains the minimal scientific/vision stack.  
> We intentionally **do not** pin `torch` here because it depends on your CUDA.

---

## 3) Install the local Ultralytics fork (editable)

This step installs the CLI **`yolo`** that points to the code in this repo.

```bash
pip install -e .
```

**Verify it’s using the local sources:**
```bash
python -c "import ultralytics; print('ultralytics path ->', ultralytics.__file__)"
# Expect a path like: .../SCPNet/ultralytics/__init__.py

```

---

## 4) Datasets

Default dataset config lives at:
```
ultralytics/cfg/datasets/SDG-TISD.yaml
```
Edit `train: / val: / test:` paths inside the YAML to point to your data.  
Standard YOLO txt label format is expected.

---

## 5) Training / Validation / Inference

### 5.1 Train

Example (SCPNet-n + 256px training):

```bash
yolo detect train model=ultralytics/cfg/models/SCPNet/SCPNet-yolo12n.yaml data=ultralytics/cfg/datasets/SDG-TISD.yaml batch=16 epochs=500 imgsz=256 device=0,1
```


### 5.2 Validate

```bash
yolo detect val   model=runs/detect/train/weights/best.pt   data=ultralytics/cfg/datasets/SDG-TISD.yaml   imgsz=256
```

### 5.3 Inference (predict)

```bash
# single image or directory
yolo detect predict   model=runs/detect/train/weights/best.pt   source=path/to/images_or_dir   imgsz=256 save=True
```

### 5.4 Export (optional)

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
# formats also include: torchscript, engine (TensorRT), openvino, etc.
```

---

## 6) Troubleshooting

- **`KeyError: 'SEM'`**  
  You’re not running this repo’s local `ultralytics`. Make sure you:
  1) ran `pip install -e .` in the repo root;  
  2) `python -c "import ultralytics; print(ultralytics.__file__)"` points to this repo;  
  3) your YAML uses the exact class names (`SEM`, case-sensitive).

- **`yolo` command not found**  
  Re-run `pip install -e .`. As a fallback:
  ```bash
  python -c "from ultralytics.cfg import entrypoint; entrypoint()" detect val ...
  ```
  (but using the `yolo` CLI is strongly recommended).

---

## 7) Project Structure (key parts)

```
SCPNet/
├─ ultralytics/                 # Local fork (models, nn modules incl. SEM/CDSAM/SFCAM, cfg)
│  ├─ cfg/
│  ├─ models/
│  ├─ nn/
│  └─ ...
├─ requirements.txt
├─ pyproject.toml               # installs local fork & registers 'yolo' entrypoint
└─ README.md
```


---

## 8) Acknowledgements

This project is built on top of (and deeply inspired by) **[Ultralytics](https://ultralytics.com/)** and the YOLO family.  
We sincerely thank the Ultralytics team and community for their outstanding open-source work.

---

## 10) License

Please see the license files in this repository and respect the upstream Ultralytics license where applicable.

---

### Reproduce in 30 seconds (summary)

```bash
conda create -n scpnet python=3.10 -y && conda activate scpnet
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
git clone https://github.com/<your-username>/SCPNet.git && cd SCPNet
pip install -r requirements.txt
pip install -e .
yolo detect train model=ultralytics/cfg/models/SCPNet/SCPNet-yolo12n.yaml data=ultralytics/cfg/datasets/SDG-TISD.yaml batch=16 epochs=500 imgsz=256
```

