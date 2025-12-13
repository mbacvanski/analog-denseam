## Install `uv` for package management

https://docs.astral.sh/uv/getting-started/installation/

## Install and activate virtual environment

```
uv sync
source .venv/bin/activate
```

## Visualization notebook

These notebooks use the pretrained weights stored in `data/model_best.npz`.

Sample energy and visible neuron trajectories: `make_plots.ipynb`
Training loss/accuracy: `plot_training.ipynb`

## Train model

`model_energy_train.py` creates the training and validation data and saves them to the `data` folder. It also saved the best model weights and the weights from the last epoch to the `data` folder.

```
uv run model_energy_train.py
```
