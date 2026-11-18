# INFLECT

Code for the paper:

> Elena Tiukhova, Emiliano Penaloza, María Óskarsdóttir, Bart Baesens, Monique Snoeck, Cristián Bravo (2024). INFLECT-DGNN: Influencer Prediction With Dynamic Graph Neural Networks. *IEEE Access*, 12, 115026–115041. <https://doi.org/10.1109/ACCESS.2024.3443533>

Preprint: <https://arxiv.org/abs/2307.08131>.

INFLECT-DGNN predicts influencers in a growing customer network: customers who refer others who
then obtain a credit card. A graph neural network encoder (GIN or GAT) embeds each monthly snapshot
of the network, a recurrent network (GRU or LSTM) carries the embeddings through a three-month
rolling window, and a feed-forward decoder gives the influencer score. Weighted binary
cross-entropy or class-balanced focal loss, and SMOTE adapted to graphs, handle the class imbalance.
This repository holds the model code and the training script for these models and for the paper's
baselines (Features-RNN, Features+PR-RNN and static GNN). The script evaluates each model on seen
and unseen nodes and writes the per-node scores that the profit analysis of the paper uses. The data
and the script that built the networks are not included (see [Data](#data)).

## Requirements

The Python version used for the paper is not documented, but the pinned versions need Python 3.8:
pandas 1.1.0 and scikit-learn 0.23.1 have PyPI wheels only up to Python 3.8, and matplotlib 3.7.0
needs Python 3.8 or later. `train.py` places the model on `cuda:0`, so it needs a CUDA GPU. The
paper reports training on an NVIDIA A100 GPU.

```bash
pip install -r requirements.txt
```

`requirements.txt` holds the environment of the paper with the same version numbers from PyPI. Its
first line points pip to the PyTorch Geometric wheel index for `torch-scatter` and `torch-sparse`
with torch 1.13 and CUDA 11.7. For a CPU-only machine, change `cu117` to `cpu` in that line. The
set resolves without conflicts for Python 3.8 on Linux. `requirements-alliance.txt` is the original
freeze from the Digital Research Alliance of Canada cluster, where most versions carry the
`+computecanada` label of the Alliance wheelhouse. The packages that the code imports are
`torch`, `torch-geometric`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `tqdm`, `wandb` and
`carbontracker`.

`train.py` logs to Weights & Biases under the entity `INFLECT` (line 102 of `train.py`). To run
without access to that entity, set the environment variable `WANDB_MODE=offline`, or change the
entity in that line.

## Data

The data come from a Super-App company in Latin America with a delivery app and credit cards: nine
months of monthly customer networks in three cities. The provider is anonymous, and the data are
not distributed with this repository. The script that built the network files (`make_data.py`,
named in an earlier version of this README) is not in the repository either.
[`data/README.md`](data/README.md) describes the data and lists the files that `train.py` expects,
with their paths.

## Training

Run every command from the repository root: `train.py` imports `models` and `utils` as packages and
reads the data from paths relative to the working directory.

`train.py` trains one model on one city, keeps the epoch with the lowest validation loss, and
evaluates it on the validation and test windows. `--GNN` and `--RNN` choose the model, and
`--data_name` chooses the city through the names of its data files. The repository has no launch
script for the paper's runs. The flags for each model of the paper, read from `models/models.py`
and `models/RNNs.py`, are:

| Model in the paper | Flags |
|---|---|
| GIN-GRU, GIN-LSTM | `--GNN GIN --RNN GRU`, `--GNN GIN --RNN LSTM` |
| GAT-GRU, GAT-LSTM | `--GNN GAT --RNN GRU`, `--GNN GAT --RNN LSTM` |
| Features-GRU, Features-LSTM | `--RNN GRU`, `--RNN LSTM`, without `--GNN` |
| Features+PR-GRU, Features+PR-LSTM | `--page_rank --RNN GRU`, `--page_rank --RNN LSTM`, without `--GNN` |
| Static GIN, Static GAT | `--GNN GIN --full_windows`, `--GNN GAT --full_windows`, without `--RNN` |

The static models need `--full_windows`, because the GNN-only model in `models/models.py` accepts
single months and not windows. For example, for GIN-GRU:

```bash
python train.py --GNN GIN --RNN GRU --epochs 300 --dropout_rate 0.5 \
  --GNN_layers <n> --gnn_output_dim <d> --embedding_dim <d> --upsample_rate <rate> \
  --RNN_layers <n> --RNN_hidden_dim <d> --loss <bce|focal> \
  --data_name <city> --run_name <run> --wandb_name <project> --output_name <folder>
```

Tables 7 to 9 of the paper give the configuration of each model in each city. Their columns map to
the flags as follows, by the help strings in `train.py`: GNN layers to `--GNN_layers`, GNN
embeddings to `--gnn_output_dim`, GNN hidden dimensions to `--embedding_dim`, SMOTE rate to
`--upsample_rate`, RNN layers to `--RNN_layers`, RNN hidden dimensions to `--RNN_hidden_dim`, and
loss to `--loss`. The paper trains for 300 epochs with a learning rate of 0.0001, two GAT heads,
focal loss with γ = 2 and β = 0.999, and 50% dropout. The defaults of `train.py` match these except
for the epochs (default 5) and the dropout (default 0.2), so the command above sets both.
`--gnn_input_dim` must equal the number of node features (default 31).

### Code layout

```
 |-Appendices
 | |-Online appendices.pdf
 |-data
 | |-README.md
 |-models
 | |-GNNs.py
 | |-RNNs.py
 | |-decoder.py
 | |-models.py
 |-utils
 | |-utils.py
 |-train.py
 |-requirements.txt
 |-requirements-alliance.txt
 |-CITATION.cff
 |-LICENSE
```

- `models/`: the encoders (`GNNs.py`), the recurrent networks (`RNNs.py`) and the decoder
  (`decoder.py`), which `models.py` combines into the dynamic GNN models and the baseline models.
- `utils/utils.py`: functions used by the other files: the losses, the metrics and their bootstrap,
  the oversampling of embeddings, the logging and the profit output.
- `train.py`: the script that runs the experiments.
- `requirements.txt`: the project's dependencies, with the versions of the paper from PyPI.
- `requirements-alliance.txt`: the same environment as frozen on the Alliance cluster.

## Evaluation

There is no separate evaluation script. After training, `train.py` reloads the weights of the epoch
with the lowest validation loss and evaluates them on the validation window and then on the test
window. It reports AUC and AUPRC on all nodes, seen nodes and unseen nodes, and the bootstrap mean
and standard deviation of these metrics on seen and unseen nodes (`--boot_sample` resamples, default
10000). CarbonTracker reports the energy use and carbon footprint of the training. The outputs are:

- standard output: the metrics of every epoch and of the final model;
- Weights & Biases: the training and validation metrics of every epoch, in the project named by
  `--wandb_name`, with local files in `wandb_<wandb_name>/`;
- one CSV row per run with the final metrics, appended to the path `./INFLECT` followed by
  `--log_file`. With the default `--log_file logs.csv`, the file is `./INFLECTlogs.csv`;
- `INFLECT/output_<output_name>/<run_name>_val_for_profit.csv` and `<run_name>_test_for_profit.csv`:
  for each node with an edge in the last month of the window, the label, the score, the summed profit of the
  linked customers in `dict_v.pkl` over six months, and the predicted label. These files feed the
  profit analysis. The folder must exist before the run.

## Pretrained models

Not released. `train.py` keeps the best weights in memory and does not save them to disk.

## Results

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Table 4: AUC and AUPRC on seen and unseen test nodes, time and CO2, per model and city | `python train.py` once per model and city, with the configuration of Tables 7 to 9 (see [Training](#training)). How the runs were collected into the table: Not documented. | Table 4 of the paper |
| Tables 7 to 9: best configuration per model and city, from the grid of Table 3 | Not documented. The repository has no grid-search script. | Tables 7 to 9 of the paper |
| Figure 4 and Online Appendix A: convergence plots | Not documented. `train.py` logs the metrics of every epoch to Weights & Biases. | Figure 4, and [`Appendices/Online appendices.pdf`](Appendices/Online%20appendices.pdf) |
| Figures 5 and 6: profit analysis | Not documented. `train.py` writes the `*_for_profit.csv` files; the threshold and profit analysis is not in the repository. | Figures 5 and 6 of the paper |
| Table 6 and Figure 7: node features and network characteristics | Not documented. The data preparation is not in the repository. | Table 6 and Figure 7 of the paper |

[`Appendices/Online appendices.pdf`](Appendices/Online%20appendices.pdf) (1.2 MB) holds Online
Appendix A of the paper: the convergence plots of the ten models in the three cities.

## Citation

```bibtex
@article{tiukhova2024inflect,
  title   = {{INFLECT-DGNN}: Influencer Prediction With Dynamic Graph Neural Networks},
  author  = {Tiukhova, Elena and Penaloza, Emiliano and Óskarsdóttir, María and Baesens, Bart and Snoeck, Monique and Bravo, Cristián},
  year    = {2024},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {115026--115041},
  doi     = {10.1109/ACCESS.2024.3443533},
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

GPL-3.0 (GNU General Public License, version 3). See `LICENSE`.

## Acknowledgements

From the paper: the research was sponsored by the ING Chair on Applying Deep Learning on Metadata as
a Competitive Accelerator. It was supported by the NSERC Discovery Grant program
[RGPIN-2020-07114], the Canada Research Chairs program [CRC-2018-00082] and the Icelandic Research
Fund [228511-051]. Compute Ontario, Calcul Québec and the Digital Research Alliance of Canada
provided computing support. The authors thank the anonymous company that provided the data.
