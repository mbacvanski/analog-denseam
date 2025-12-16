## Getting Started 

### Install `uv` for package management

https://docs.astral.sh/uv/getting-started/installation/

### Install and activate virtual environment

```
uv sync
source .venv/bin/activate
```

### Train model

`model_energy_train.py` creates the training and validation data and saves them to the `data` folder. It also saved the best model weights and the weights from the last epoch to the `data` folder.

```
uv run model_energy_train.py
```

### Visualization notebook

These notebooks use the pretrained weights stored in `data/model_best.npz`.

`make_plots.ipynb` contains various plots, including sample energy and visible neuron trajectories and training/validation loss/accuracy.
