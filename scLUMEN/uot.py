import numpy as np
import ot
import torch
from tqdm import tqdm


def compute_uot_plans(data_by_time, train_times, delta, use_mini_batch, chunk_size, device):
    source_plans = []
    target_plans = []
    sampling_metadata = []

    for index in tqdm(range(len(train_times) - 1), desc="Computing UOT plans..."):
        source = data_by_time[index]
        target = data_by_time[index + 1]
        source_mass = np.ones(source.shape[0])
        target_mass = np.ones(target.shape[0])

        distances = ot.dist(source, target, metric="euclidean")
        cosine_squared = np.cos(
            np.minimum(distances / (2 * delta), np.pi / 2)
        ) ** 2
        cost = -np.log(np.where(cosine_squared == 0, 1e-10, cosine_squared))

        if use_mini_batch:
            coupling, metadata = _compute_block_coupling(
                cost, source_mass, target_mass, chunk_size, device
            )
        else:
            coupling = _solve_unbalanced_ot(
                source_mass, target_mass, cost, device
            )
            metadata = None

        source_sum = coupling.sum(axis=1)
        target_sum = coupling.sum(axis=0)
        source_plan = (
            source_mass / (source_sum + 1e-12)
        )[:, None] * coupling
        target_plan = (
            target_mass / (target_sum + 1e-12)
        ) * coupling
        source_plans.append(source_plan)
        target_plans.append(target_plan)
        sampling_metadata.append(metadata)

    return source_plans, target_plans, sampling_metadata


def _solve_unbalanced_ot(source_mass, target_mass, cost, device):
    source_tensor = torch.as_tensor(source_mass, device=device)
    target_tensor = torch.as_tensor(target_mass, device=device)
    cost_tensor = torch.as_tensor(cost, device=device)
    return (
        ot.unbalanced.mm_unbalanced(
            source_tensor,
            target_tensor,
            cost_tensor,
            reg_m=[1.0, 1.0],
        )
        .cpu()
        .numpy()
    )


def _compute_block_coupling(cost, source_mass, target_mass, chunk_size, device):
    group_count = source_mass.shape[0] // chunk_size + 1
    source_permutation = np.random.permutation(source_mass.shape[0])
    target_permutation = np.random.permutation(target_mass.shape[0])
    source_groups = np.array_split(source_permutation, group_count)
    target_groups = np.array_split(target_permutation, group_count)
    coupling = np.zeros((source_mass.shape[0], target_mass.shape[0]))
    conditional_subplans = []

    for source_indices, target_indices in zip(source_groups, target_groups):
        sub_coupling = _solve_unbalanced_ot(
            source_mass[source_indices],
            target_mass[target_indices],
            cost[np.ix_(source_indices, target_indices)],
            device,
        )
        coupling[np.ix_(source_indices, target_indices)] = sub_coupling
        row_sums = sub_coupling.sum(axis=1)
        conditional_subplans.append(
            (
                (
                    source_mass[source_indices] / (row_sums + 1e-12)
                )[:, None]
                * sub_coupling
            ).astype(np.float32)
        )

    metadata = {
        "sub_plans": conditional_subplans,
        "source_groups": source_groups,
        "target_groups": target_groups,
    }
    return coupling, metadata


def sample_from_ot_plan(
    plan,
    source,
    target,
    batch_size,
    metadata,
    device,
):
    if metadata is None:
        source_indices, target_indices = _sample_dense_plan(
            plan, batch_size, device
        )
    else:
        source_indices, target_indices = _sample_block_plans(
            metadata, batch_size, device
        )
    return (
        source[source_indices],
        target[target_indices],
        source_indices,
        target_indices,
    )


def _sample_dense_plan(plan, batch_size, device):
    plan_tensor = torch.as_tensor(plan, dtype=torch.float32, device=device)
    row_sums = plan_tensor.sum(axis=1)
    if row_sums.sum() < 1e-9:
        raise RuntimeError("The unbalanced OT plan has zero total mass.")
    source_indices = torch.multinomial(
        row_sums / row_sums.sum(), batch_size, replacement=True
    )
    selected_rows = plan_tensor[source_indices]
    conditional = selected_rows / (row_sums[source_indices, None] + 1e-12)
    target_indices = torch.multinomial(conditional, 1).squeeze(1)
    return source_indices.cpu().numpy(), target_indices.cpu().numpy()


def _sample_block_plans(metadata, batch_size, device):
    subplans = metadata["sub_plans"]
    block_masses = torch.as_tensor(
        [plan.sum() for plan in subplans], dtype=torch.float32, device=device
    )
    if block_masses.sum() < 1e-9:
        raise RuntimeError("The blockwise unbalanced OT plans have zero total mass.")
    sampled_blocks = torch.multinomial(
        block_masses / block_masses.sum(), batch_size, replacement=True
    )
    source_output = torch.empty(batch_size, dtype=torch.int64, device=device)
    target_output = torch.empty(batch_size, dtype=torch.int64, device=device)

    for block_index, count in zip(*torch.unique(sampled_blocks, return_counts=True)):
        block = int(block_index.item())
        subplan = torch.as_tensor(subplans[block], device=device)
        row_sums = subplan.sum(axis=1)
        local_source = torch.multinomial(
            row_sums / row_sums.sum(), int(count.item()), replacement=True
        )
        selected_rows = subplan[local_source]
        conditional = selected_rows / (row_sums[local_source, None] + 1e-12)
        local_target = torch.multinomial(conditional, 1).squeeze(1)
        mask = sampled_blocks == block_index
        source_group = torch.as_tensor(
            metadata["source_groups"][block], device=device
        )
        target_group = torch.as_tensor(
            metadata["target_groups"][block], device=device
        )
        source_output[mask] = source_group[local_source]
        target_output[mask] = target_group[local_target]

    return source_output.cpu().numpy(), target_output.cpu().numpy()


def compute_wfr_path(relative_time, delta_time, source, target, mass0, mass1, delta):
    valid = torch.norm(target - source, dim=1) < torch.pi * delta
    relative_time = relative_time[valid]
    source = source[valid]
    target = target[valid]
    mass0 = mass0[valid]
    mass1 = mass1[valid]

    difference = target - source
    norm = torch.norm(difference, dim=1, keepdim=True)
    direction = difference / (norm + 1e-9)
    tangent = torch.tan(norm / (2 * delta))
    scale = torch.sqrt(mass0 * mass1 / (1 + tangent**2))
    omega = 2 * delta * tangent * scale
    omega_vector = omega * direction

    coefficient_a = mass1 + mass0 - 2 * scale
    coefficient_b = mass0 - scale
    inverse_scale = 2 * delta / (omega + 1e-9)
    state = source + omega_vector * (
        inverse_scale
        * (
            torch.arctan(
                (coefficient_a * relative_time - coefficient_b) * inverse_scale
            )
            - torch.arctan(-coefficient_b * inverse_scale)
        )
    )
    interpolated_mass = (
        coefficient_a * relative_time**2
        - 2 * coefficient_b * relative_time
        + mass0
    )
    mass_derivative = 2 * coefficient_a * relative_time - 2 * coefficient_b
    growth = mass_derivative / interpolated_mass / delta_time
    velocity = omega_vector / interpolated_mass / delta_time
    relative_mass = interpolated_mass / mass0
    return state, growth, velocity, relative_mass, valid


def get_batch(
    data_by_time,
    train_times,
    batch_size,
    source_plans,
    target_plans,
    delta,
    sampling_metadata,
    device,
):
    times = []
    states = []
    velocities = []
    growth_rates = []
    relative_masses = []

    for index in range(len(train_times) - 1):
        source, target, source_indices, target_indices = sample_from_ot_plan(
            source_plans[index],
            data_by_time[index],
            data_by_time[index + 1],
            batch_size,
            sampling_metadata[index],
            device,
        )
        source = torch.as_tensor(source, dtype=torch.float32, device=device)
        target = torch.as_tensor(target, dtype=torch.float32, device=device)
        mass0 = torch.as_tensor(
            source_plans[index][source_indices, target_indices].reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        mass1 = torch.as_tensor(
            target_plans[index][source_indices, target_indices].reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )

        delta_time = train_times[index + 1] - train_times[index]
        relative_time = torch.rand(source.shape[0], 1, device=device)
        state, growth, velocity, relative_mass, valid = compute_wfr_path(
            relative_time, delta_time, source, target, mass0, mass1, delta,
        )
        times.append(delta_time * relative_time[valid] + train_times[index])
        states.append(state)
        velocities.append(velocity)
        growth_rates.append(growth)
        relative_masses.append(relative_mass)

    if not states or sum(len(state) for state in states) == 0:
        raise RuntimeError("No valid conditional paths were sampled; check delta and snapshot support.")
    return (
        torch.cat(times),
        torch.cat(states),
        torch.cat(velocities),
        torch.cat(growth_rates),
        torch.cat(relative_masses),
    )
