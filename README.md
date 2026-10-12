
# PS-NSP-FewShot-AD

Official implementation of the paper:

**Physics- and Statistics-guided Normal Subspace Purification for Training-Free Few-Shot Industrial Anomaly Detection**

**Journal:** IET Image Processing

**DOI:** https://doi.org/10.1049/ipr2.70493

## Overview

This repository provides the implementation of a physics- and statistics-guided normal subspace purification method for training-free few-shot industrial anomaly detection.

## Requirements

The Python dependencies are provided in `requirements.txt` and `environment.yml`.

```bash
pip install -r requirements.txt
```

## Datasets

Experiments use the MVTec AD and VisA datasets. Please obtain the datasets from their official sources and configure the dataset paths before running the scripts.

## Code Structure

- `main.py`: Main experiment program
- `src/subspacead/`: Core implementation
- `scripts/`: Experiment scripts
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment

## Experiments

The experiment scripts are located in `scripts/`:

- `run_full_benchmark.sh`
- `run_visa_benchmark.sh`
- `run_672_ablations.sh`

Please configure the dataset paths and runtime environment in the scripts before execution.

## Citation

If you use this work, please cite the associated paper.

DOI: https://doi.org/10.1049/ipr2.70493
