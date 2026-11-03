import numpy as np
import torch
from tqdm import tqdm

from .uot import compute_uot_plans, get_batch


def pretrain(
    model,
    dataframe,
    velocity_optimizer,
    growth_optimizer,
    velocity_scheduler,
    growth_scheduler,
    n_epoch,
    batch_size,
    device,
    delta,
    use_mini_batch,
    chunk_size,
    ema,
    show_progress=True,
):
    time_labels = dataframe["samples"].to_numpy()
    values = dataframe.iloc[:, 1:].to_numpy()
    train_times = np.unique(time_labels)
    data_by_time = [values[time_labels == time] for time in train_times]

    source_plans, target_plans, sampling_metadata = compute_uot_plans(
        data_by_time,
        train_times,
        delta,
        use_mini_batch,
        chunk_size,
        device,
    )

    velocity_losses = []
    growth_losses = []
    total_losses = []
    progress = tqdm(
        range(n_epoch), desc="Training flow and growth fields...", unit="epoch", disable=not show_progress,
    )

    for _ in progress:
        velocity_optimizer.zero_grad()
        growth_optimizer.zero_grad()
        time, state, target_velocity, target_growth, relative_mass = get_batch(
            data_by_time,
            train_times,
            batch_size,
            source_plans,
            target_plans,
            delta,
            sampling_metadata,
            device,
        )
        predicted_velocity = model.v_net(time, state)
        predicted_growth = model.g_net(time, state)
        velocity_loss = torch.mean(
            (predicted_velocity - target_velocity) ** 2 * relative_mass
        )
        growth_loss = torch.mean(
            (predicted_growth - target_growth) ** 2 * relative_mass
        )
        total_loss = velocity_loss + growth_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite flow/growth loss.")
        total_loss.backward()

        velocity_optimizer.step()
        growth_optimizer.step()
        velocity_scheduler.step()
        growth_scheduler.step()
        ema.update()

        velocity_losses.append(velocity_loss.item())
        growth_losses.append(growth_loss.item())
        total_losses.append(total_loss.item())
        progress.set_postfix(
            {
                "loss": f"{total_loss.item():.6f}",
                "velocity": f"{velocity_loss.item():.6f}",
                "growth": f"{growth_loss.item():.6f}",
            }
        )

    return model, velocity_losses, growth_losses, total_losses
