import numpy as np
import torch
from geomloss import SamplesLoss

REFINE_MODES = ("none", "static", "dyn", "both")


def refine_decoder(vae, snapshots, latent_particles, particle_weights, mode="both", steps=600, batch_size=512,
                   lr=1e-4, blur=0.5, seed=0):
    if mode == "none" or steps <= 0:
        return vae
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(vae.parameters()).device
    decoder = vae.decoder
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=0.7, debias=True, backend="tensorized")
    times = sorted(snapshots)
    with torch.no_grad():
        posterior_means = {
            t: vae.encode(torch.as_tensor(snapshots[t], dtype=torch.float32, device=device))[0] for t in times
        }
    for _ in range(steps):
        t = times[generator.integers(len(times))]
        cells = snapshots[t]
        observed = torch.as_tensor(
            cells[generator.choice(len(cells), min(batch_size, len(cells)), replace=False)],
            dtype=torch.float32, device=device,
        )
        loss = 0.0
        if mode in ("static", "both"):
            means = posterior_means[t]
            chosen = torch.as_tensor(generator.choice(len(means), min(batch_size, len(means)), replace=False))
            loss = loss + loss_fn(decoder(means[chosen.to(device)]), observed)
        if mode in ("dyn", "both"):
            weights = particle_weights[t] / particle_weights[t].sum()
            chosen = generator.choice(len(weights), min(batch_size, len(weights)), replace=False)
            particle_mass = torch.as_tensor(weights[chosen] / weights[chosen].sum(), dtype=torch.float32, device=device)
            particles = torch.as_tensor(latent_particles[t][chosen], dtype=torch.float32, device=device)
            uniform = torch.full((len(observed),), 1.0 / len(observed), device=device)
            loss = loss + loss_fn(particle_mass, decoder(particles), uniform, observed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return vae
