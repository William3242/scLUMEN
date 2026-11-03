from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    short_name: str
    expression_file: str
    metadata_file: str
    stage_column: str
    test_tps: tuple[int, ...]
    vae_hidden_dim: int
    vae_n_hiddens: int
    vae_beta: float
    vae_beta_warmup: int
    already_normalized: bool = False
    numeric_stages: bool = False


DATASETS = {
    "zebrafish": DatasetSpec(
        "ZB", "three_interpolation-count_data-hvg.csv", "meta_data.csv", "stage.nice",
        (4, 6, 8), 256, 2, 0.5, 500,
    ),
    "drosophila": DatasetSpec(
        "DR", "three_interpolation-count_data-hvg.csv", "subsample_meta_data.csv", "time",
        (4, 6, 8), 256, 2, 0.25, 500,
    ),
    "schiebinger": DatasetSpec(
        "SC", "three_interpolation-norm_data-hvg.csv", "three_interpolation-meta_data.csv", "day",
        (5, 10, 15), 512, 3, 0.1, 1000, already_normalized=True, numeric_stages=True,
    ),
    "pancreatic": DatasetSpec(
        "PA", "three_interpolation-count_data-hvg.csv", "meta_data.csv", "CellWeek",
        (2, 4, 6), 512, 3, 0.25, 1000,
    ),
}


@dataclass(frozen=True)
class TrajectoryData:
    name: str
    trajectory: tuple[np.ndarray, ...]
    gene_names: tuple[str, ...]
    stage_names: tuple[str, ...]
    test_tps: tuple[int, ...]
    labels: tuple[np.ndarray, ...] | None = None

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def n_timepoints(self) -> int:
        return len(self.trajectory)

    @property
    def train_tps(self) -> tuple[int, ...]:
        return tuple(t for t in range(self.n_timepoints) if t not in self.test_tps)


def _natural_key(value: object) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    ]


def normalize_log1p(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / totals * np.float32(target_sum)).astype(np.float32)


def _read_gmt(path: Path) -> dict[str, set[str]]:
    sets = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            sets[parts[0]] = set(parts[2:])
    return sets


def _load_labels(name: str, data_dir: Path, cells: pd.Index, metadata: pd.DataFrame) -> np.ndarray | None:
    if name == "zebrafish":
        path = data_dir / "URD_Dropseq_Meta.txt"
        if not path.is_file():
            return None
        urd = pd.read_csv(path, sep="\t", skiprows=[1], index_col=0, low_memory=False).loc[cells]
        return urd["Segment"].fillna(-1).astype(int).astype(str).to_numpy()
    if name == "drosophila":
        return metadata["seurat_clusters"].astype(str).to_numpy()
    if name == "pancreatic":
        return metadata["Assigned_cluster"].astype(str).to_numpy()
    path = data_dir / "cell_sets.gmt"
    if not path.is_file():
        return None
    sets = _read_gmt(path)
    labels = pd.Series("Other", index=cells)
    labels[cells.isin(sets["MET"])] = "MET"
    for cell_type in ("IPS", "Stromal", "Neural", "Trophoblast", "Epithelial"):
        labels[cells.isin(sets[cell_type])] = cell_type
    return labels.to_numpy()


def load_dataset(name: str, data_dir: str | Path) -> TrajectoryData:
    spec = DATASETS[name]
    data_dir = Path(data_dir).expanduser().resolve()
    expression_path = data_dir / spec.expression_file
    metadata_path = data_dir / spec.metadata_file
    expression = pd.read_csv(expression_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata = metadata.loc[expression.index]

    keep = ~metadata[spec.stage_column].isna().to_numpy()
    expression, metadata = expression[keep], metadata[keep]
    values = expression.to_numpy(dtype=np.float32)
    if not spec.already_normalized:
        values = normalize_log1p(values)
    stage_values = metadata[spec.stage_column].to_numpy()
    sort_key = float if spec.numeric_stages else _natural_key
    stages = sorted(np.unique(stage_values), key=sort_key)

    labels = _load_labels(name, data_dir, expression.index, metadata)
    return TrajectoryData(
        name=name,
        trajectory=tuple(values[stage_values == stage] for stage in stages),
        gene_names=tuple(str(gene) for gene in expression.columns),
        stage_names=tuple(str(stage) for stage in stages),
        test_tps=spec.test_tps,
        labels=None if labels is None else tuple(labels[stage_values == stage] for stage in stages),
    )


def subsample_trajectory(
    trajectory: Sequence[np.ndarray], max_cells: int | None, seed: int
) -> tuple[np.ndarray, ...]:
    if max_cells is None:
        return tuple(np.asarray(cells, dtype=np.float32) for cells in trajectory)
    rng = np.random.default_rng(seed)
    sampled = []
    for cells in trajectory:
        cells = np.asarray(cells, dtype=np.float32)
        if len(cells) > max_cells:
            cells = cells[rng.choice(len(cells), size=max_cells, replace=False)]
        sampled.append(cells)
    return tuple(sampled)


def build_wfr_dataframe(
    trajectory: Sequence[np.ndarray], timepoints: Sequence[int]
) -> pd.DataFrame:
    n_genes = trajectory[0].shape[1]
    columns = [f"x{i + 1}" for i in range(n_genes)]
    frames = []
    for timepoint in timepoints:
        frame = pd.DataFrame(np.asarray(trajectory[timepoint]), columns=columns)
        frame.insert(0, "samples", float(timepoint))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
