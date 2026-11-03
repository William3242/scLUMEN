import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

from .models import ODEFunc


def generate_trajectories(dataframe, model, device, timepoints, trajectory_bins=100):
    times = np.asarray(sorted(timepoints), dtype=np.float32)
    start_time = float(times[0])
    initial_rows = dataframe[dataframe["samples"] == start_time]
    initial_array = initial_rows.iloc[:, 1:].to_numpy(
        dtype="float32", copy=True
    )
    initial_state = torch.from_numpy(initial_array).to(device).requires_grad_()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    log_weights = torch.full(
        (initial_state.shape[0], 1),
        -np.log(initial_state.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    augmented_state = (initial_state, log_weights)
    ode_function = ODEFunc(model)

    dense_times = torch.linspace(
        float(times[0]), float(times[-1]), trajectory_bins, device=device
    )
    dense_trajectory, _ = odeint(ode_function, augmented_state, dense_times)
    sample_count = min(100, dense_trajectory.shape[1])
    sample_indices = np.random.choice(
        dense_trajectory.shape[1], size=sample_count, replace=False
    )

    evaluation_times = torch.as_tensor(times, dtype=torch.float32, device=device)
    trajectory, trajectory_log_weights = odeint(
        ode_function, augmented_state, evaluation_times
    )
    return {
        "latent": trajectory.detach().cpu().numpy(),
        "log_mass": trajectory_log_weights.detach().cpu().numpy()[..., 0],
        "mass": torch.softmax(trajectory_log_weights, dim=1).detach().cpu().numpy()[..., 0],
        "dense_trajectory": dense_trajectory[:, sample_indices].detach().cpu().numpy(),
    }
