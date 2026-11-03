# scLUMEN

Mass-aware latent unbalanced flow matching for continuous single-cell dynamics.

## Contents

- `scLUMEN/`: model, training and evaluation code.
- `scripts/`: demo notebooks (`ZB.ipynb`, `PA.ipynb`) and the command-line entry point (`run_sclumen.py`).
- `data/`: see `data/README.md` for the expected files.

## Installation

```bash
conda create -n sclumen python=3.10
conda activate sclumen
pip install -r requirements.txt
```

For CUDA 12.8, use `conda env create -f environment.yml` instead.

## Usage

Open `scripts/ZB.ipynb` for the zebrafish demo, or run:

```bash
python scripts/run_sclumen.py --dataset zebrafish --data-dir data/zebrafish --seed 0
```

Datasets: `zebrafish`, `drosophila`, `schiebinger`, `pancreatic`. Results are written to `results/<dataset>/seed_<seed>/`.

Add `--refine-modes none static dyn both` for the decoder ablation and `--latent pca` for the PCA ablation.

## Dataset

[https://zenodo.org/records/23926311](https://zenodo.org/records/23926311)

## License

[MIT](LICENSE)
