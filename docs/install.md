# Installation

## Supported environment

- Linux is recommended for CUDA and the rotated-NMS extension.
- Python 3.12, CUDA 12.4, and PyTorch 2.5 are the validated cloud
  combination. Package metadata accepts Python 3.8 or newer.
- Install a PyTorch wheel matching the host CUDA driver from the
  [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Install the package

```bash
git clone https://github.com/DexZane/MSDYOLO.git
cd MSDYOLO
python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1"
python -m pip install -e .
python -m pip install "setuptools==69.5.1"  # restore the pin after dependency resolution
```

The setuptools pin keeps Python 3.12 environments that still depend on
`pkgutil.ImpImporter` from failing during installation. `setup.sh` applies
the same pin automatically on a cloud instance and also installs the
OpenDataLab SDK used for DOTA v1.5.

The package metadata is GPL-3.0 and includes the DOTA YAML/hyperparameter
assets and the rotated-NMS C++/CUDA sources in wheel builds. The four console
commands are available after installation:

```bash
msdyolo-train --help
msdyolo-val --help
msdyolo-detect --help
msdyolo-export --help
```

## Optional rotated-NMS extension

The Python fallback is importable without compiling an extension. For CUDA
NMS acceleration, build the extension from the canonical source directory:

```bash
cd msdyolo/utils/nms_rotated
python setup.py build_ext --inplace
```

The source files are deliberately shipped in the wheel so a deployed
environment can build the extension after installation.

## Verify the installation

```bash
python -m pytest -q
python -m msdyolo.train --help
```

For a real DOTA cloud run, use `bash scripts/setup.sh`; it performs package
installation, SDK setup, raw-data normalization, atomic patch preparation,
weight validation, and training launch. See [GetStart.md](GetStart.md) for
the operational flags and PID/log locations.
