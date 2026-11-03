from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .datasets import DATASETS, TrajectoryData, build_wfr_dataframe, load_dataset, subsample_trajectory
from .ema import EMA
from .inference import generate_trajectories
from .metrics import (
    CellTypeProportions,
    gene_time_correlation_median,
    sinkhorn_ot,
    total_variation,
    weighted_sinkhorn_ot,
)
from .models import FNet
from .refine import REFINE_MODES, refine_decoder
from .train import pretrain
from .vae import PCAAE, VAE, decode_latent, encode_dataframe, train_vae


@dataclass
class ExperimentConfig:
    dataset: str
    data_dir: str
    output_dir: str = "results"
    seed: int = 0
    flow_seed: int | None = None
    device: str = "auto"
    latent: str = "vae"
    latent_dim: int = 10
    vae_hidden_dim: int | None = None
    vae_n_hiddens: int | None = None
    vae_epochs: int = 3000
    vae_batch_size: int = 512
    vae_lr: float = 1e-3
    vae_beta: float | None = None
    vae_beta_warmup: int | None = None
    wfr_delta: float = 2.0
    hidden_dim: int = 256
    n_hiddens: int = 5
    fm_epochs: int = 1500
    fm_batch_size: int = 256
    lr_velocity: float = 5e-4
    lr_growth: float = 5e-3
    eta_min: float = 1e-5
    chunk_size: int = 4096
    ema_decay: float = 0.995
    num_points: int = 2000
    trajectory_bins: int = 100
    refine_modes: tuple[str, ...] = ("none", "both")
    refine_steps: int = 600
    refine_batch_size: int = 512
    refine_lr: float = 1e-4
    refine_blur: float = 0.5
    eval_max_cells: int | None = 3000
    max_cells_per_timepoint: int | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        spec = DATASETS[self.dataset]
        for field in ("vae_hidden_dim", "vae_n_hiddens", "vae_beta", "vae_beta_warmup"):
            if getattr(self, field) is None:
                setattr(self, field, getattr(spec, field))
        if self.flow_seed is None:
            self.flow_seed = self.seed + 1000
        self.refine_modes = tuple(self.refine_modes)

    @property
    def run_dir(self) -> Path:
        name = self.dataset if self.latent == "vae" else f"{self.dataset}_pca"
        return Path(self.output_dir).expanduser().resolve() / name / f"seed_{self.seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    return device


def train_latent_model(config: ExperimentConfig, trajectory, train_frame: pd.DataFrame, device: torch.device):
    set_seed(config.seed)
    n_genes = trajectory[0].shape[1]
    if config.latent == "pca":
        train_cells = train_frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        model = PCAAE.fit(train_cells, config.latent_dim, config.seed).to(device)
    else:
        model = VAE(
            data_dim=n_genes,
            latent_dim=config.latent_dim,
            hidden_dim=config.vae_hidden_dim,
            n_hiddens=config.vae_n_hiddens,
            activation="leakyrelu",
        ).to(device)
        model, _ = train_vae(
            model, train_frame, dim=n_genes, n_epoch=config.vae_epochs,
            batch_size=config.vae_batch_size, lr=config.vae_lr,
            beta=config.vae_beta, beta_warmup_epochs=config.vae_beta_warmup,
            device=device,
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def prepare_latent_data(config: ExperimentConfig, trajectory, train_frame: pd.DataFrame, latent_model, device: torch.device):
    n_genes = trajectory[0].shape[1]
    latent_train = encode_dataframe(latent_model, train_frame, dim=n_genes, use_mean=True, device=device)

    x0 = np.asarray(trajectory[0], dtype=np.float32)
    repeats = int(np.ceil(config.num_points / len(x0)))
    expanded_x0 = np.repeat(x0, repeats, axis=0)[: config.num_points]
    x0_frame = pd.DataFrame(expanded_x0, columns=[f"x{i + 1}" for i in range(n_genes)])
    x0_frame.insert(0, "samples", 0.0)
    latent_x0 = encode_dataframe(latent_model, x0_frame, dim=n_genes, use_mean=False, device=device)
    return latent_train, latent_x0


def train_dynamics(config: ExperimentConfig, latent_train: pd.DataFrame, device: torch.device, show_progress: bool = True):
    model = FNet(
        in_out_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_hiddens=config.n_hiddens,
        activation="leakyrelu",
    ).to(device)
    velocity_optimizer = torch.optim.Adam(model.v_net.parameters(), lr=config.lr_velocity)
    growth_optimizer = torch.optim.Adam(model.g_net.parameters(), lr=config.lr_growth)
    velocity_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        velocity_optimizer, T_max=config.fm_epochs, eta_min=config.eta_min
    )
    growth_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        growth_optimizer, T_max=config.fm_epochs, eta_min=config.eta_min
    )
    ema = EMA(model, decay=config.ema_decay)
    model, *_ = pretrain(
        model, latent_train, velocity_optimizer, growth_optimizer, velocity_scheduler, growth_scheduler,
        config.fm_epochs, config.fm_batch_size, device, config.wfr_delta, True, config.chunk_size, ema,
        show_progress=show_progress,
    )
    ema.apply_shadow()
    return model


def evaluate(config: ExperimentConfig, data: TrajectoryData, trajectory, predictions, weights, latent_model, device) -> list[dict]:
    rng = np.random.default_rng(config.seed)
    correlation = gene_time_correlation_median(trajectory, predictions)
    correlation_weighted = gene_time_correlation_median(trajectory, predictions, list(weights))
    proportions = None
    if data.labels is not None and config.max_cells_per_timepoint is None:
        proportions = CellTypeProportions(
            np.concatenate([trajectory[t] for t in data.train_tps]),
            np.concatenate([data.labels[t] for t in data.train_tps]),
        )
    rows = []
    for t in data.test_tps:
        truth = trajectory[t]
        row = {"time_point": t, "stage": data.stage_names[t]}
        if proportions is not None:
            true_proportions = proportions(truth)
            row["prop_tv_unweighted"] = total_variation(proportions(predictions[t]), true_proportions)
            row["prop_tv_weighted"] = total_variation(proportions(predictions[t], weights[t]), true_proportions)
        if config.eval_max_cells is not None and len(truth) > config.eval_max_cells:
            truth = truth[rng.choice(len(truth), config.eval_max_cells, replace=False)]
        with torch.no_grad():
            encoded = latent_model.encode(torch.as_tensor(truth, device=device))[0]
            reconstruction = latent_model.decode(encoded).cpu().numpy()
        row.update(
            gene_corr_unweighted=correlation,
            gene_corr_weighted=correlation_weighted,
            ot_unweighted=sinkhorn_ot(truth, predictions[t]),
            ot_weighted=weighted_sinkhorn_ot(truth, predictions[t], weights[t]),
            recon_floor=sinkhorn_ot(truth, reconstruction),
        )
        rows.append(row)
    return rows


def save_predictions(path: Path, data: TrajectoryData, predictions, weights) -> None:
    np.savez_compressed(
        path,
        genes=np.asarray(data.gene_names),
        stages=np.asarray(data.stage_names),
        test_tps=np.asarray(data.test_tps),
        pred_test=np.stack([predictions[t] for t in data.test_tps]).astype(np.float32),
        weight_test=np.stack([weights[t] for t in data.test_tps]),
        weight_all=weights,
        mean_unweighted=np.stack([cells.mean(0) for cells in predictions]),
        mean_weighted=np.stack([(cells * (w / w.sum())[:, None]).sum(0) for cells, w in zip(predictions, weights)]),
    )


def run_experiment(config: ExperimentConfig) -> Path:
    device = resolve_device(config.device)
    run_dir = config.run_dir
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and not config.overwrite:
        raise FileExistsError(f"Result already exists: {metrics_path}. Pass --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print(f"Device: {device}")
    data = load_dataset(config.dataset, config.data_dir)
    trajectory = subsample_trajectory(data.trajectory, config.max_cells_per_timepoint, config.seed)
    print(f"Time points ({data.n_timepoints}): {list(data.stage_names)}")
    print(f"Genes: {data.n_genes}; cells per time point: {[len(cells) for cells in trajectory]}")
    print(f"Training time points: {list(data.train_tps)}; held out: {list(data.test_tps)}")

    start = time.perf_counter()
    train_frame = build_wfr_dataframe(trajectory, data.train_tps)
    latent_model = train_latent_model(config, trajectory, train_frame, device)
    latent_name = "vae.pt" if config.latent == "vae" else "pca.pt"
    torch.save(latent_model.state_dict(), run_dir / latent_name)

    set_seed(config.flow_seed)
    latent_train, latent_x0 = prepare_latent_data(config, trajectory, train_frame, latent_model, device)
    dynamics = train_dynamics(config, latent_train, device)
    torch.save(dynamics.state_dict(), run_dir / "flow.pt")

    particles = generate_trajectories(
        latent_x0, dynamics, device, tuple(range(data.n_timepoints)), trajectory_bins=config.trajectory_bins
    )
    np.savez_compressed(run_dir / "particles.npz", **particles)
    latent, weights = particles["latent"], particles["mass"]

    all_rows = []
    prediction_mode = "both" if "both" in config.refine_modes else config.refine_modes[-1]
    for mode in config.refine_modes:
        refined = copy.deepcopy(latent_model)
        refine_decoder(
            refined,
            {t: trajectory[t] for t in data.train_tps},
            {t: latent[t] for t in data.train_tps},
            {t: weights[t] for t in data.train_tps},
            mode=mode,
            steps=config.refine_steps,
            batch_size=config.refine_batch_size,
            lr=config.refine_lr,
            blur=config.refine_blur,
            seed=config.seed,
        )
        predictions = [np.asarray(decode_latent(refined, latent[t], device=device)) for t in range(data.n_timepoints)]
        rows = evaluate(config, data, trajectory, predictions, weights, refined, device)
        all_rows += [{"dataset": config.dataset, "seed": config.seed, "decoder": mode, **row} for row in rows]
        pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
        if mode == prediction_mode:
            if mode != "none":
                torch.save(refined.state_dict(), run_dir / f"{latent_name[:-3]}_refined.pt")
            save_predictions(run_dir / "predictions.npz", data, predictions, weights)

    metrics = pd.DataFrame(all_rows)
    print(metrics.drop(columns=["dataset", "seed"]).to_string(index=False))
    print(f"Finished in {(time.perf_counter() - start) / 60:.1f} min. Results saved to: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scLUMEN on a three-interpolation benchmark.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-dir", required=True, help="Directory holding the dataset files.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-seed", type=int, help="Seed used after the latent model is trained; defaults to seed + 1000.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:1")
    parser.add_argument("--latent", default="vae", choices=["vae", "pca"], help="'pca' runs the representation ablation.")
    parser.add_argument("--vae-epochs", type=int, default=3000)
    parser.add_argument("--fm-epochs", type=int, default=1500)
    parser.add_argument("--num-points", type=int, default=2000, help="Number of particles integrated from the first time point.")
    parser.add_argument("--refine-modes", nargs="+", default=["none", "both"], choices=REFINE_MODES,
                        help="Decoders to evaluate: 'both' is scLUMEN, 'none' is the unrefined model.")
    parser.add_argument("--refine-steps", type=int, default=600)
    parser.add_argument("--refine-lr", type=float, default=1e-4)
    parser.add_argument("--eval-max-cells", type=int, default=3000, help="Cap on ground-truth cells in the Sinkhorn metric.")
    parser.add_argument("--max-cells-per-timepoint", type=int, help="For quick tests only; omit for the benchmark.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run_experiment(ExperimentConfig(**vars(build_parser().parse_args(argv))))
