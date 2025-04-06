# In-Context Learning (ICL) Implementation

This repository contains implementations of In-Context Learning for classification experiments using both JAX and PyTorch frameworks.

## Project Structure

- `main.py` - Main PyTorch implementation
- `main_jax.py` - Legacy JAX implementation
- `model.py` - PyTorch model implementation
- `model_jax.py` - JAX model implementation
- `dataset.py` - PyTorch dataset implementation
- `dataset_jax.py` - JAX dataset implementation
- `visualize.ipynb` - Visualization notebooks for analysis (old results)
- Various run scripts for different experiments

## Setup

1. Install dependencies:
```bash
pip install torch numpy wandb tqdm
```

2. For JAX implementation (legacy):
```bash
pip install jax jaxlib
```

## Usage

### PyTorch Implementation

The main PyTorch implementation can be run using various scripts:

- Basic run: `./run_torch.sh`

### JAX Implementation (Legacy)

For the legacy JAX implementation:
```bash
./run.sh
```

## Visualization

Use the provided Jupyter notebooks for visualization and analysis:
- `visualize.ipynb` - Main visualization notebook
- `visualize_layer4.ipynb` - Layer 4 specific visualizations

## Outputs

Results are stored in:
- `outs_torch/` - PyTorch implementation outputs
- `outs/` - JAX implementation outputs

## Notes

- The PyTorch implementation (`main.py`) is the current active codebase, WANDB report see: https://wandb.ai/explainableml/ICL_torch/reports/Results---VmlldzoxMjE1Mjk0Nw?accessToken=l7pw9osgk32n02dgt2ysd8dr12mfmpb6v8pblg6gn54ph7ywxbu7kdgk1bq57r6m
- The JAX implementation (`main_jax.py`) is maintained for reference, WANDB report see: https://api.wandb.ai/links/explainableml/iawqfvdf
- WandB integration is available for experiment tracking (set `WANDB = True` in main.py)