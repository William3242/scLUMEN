import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm


_ACTIVATIONS = {
    "Tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
}


def _mlp_body(input_dim, hidden_dim, n_hiddens, activation_type):
    layers = [nn.Linear(input_dim, hidden_dim), activation_type()]
    for _ in range(n_hiddens - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_type()])
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim,
        hidden_dim=256,
        n_hiddens=2,
        activation="leakyrelu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        activation_type = _ACTIVATIONS[activation]
        self.enc_body = _mlp_body(
            data_dim, hidden_dim, n_hiddens, activation_type
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        decoder_body = _mlp_body(
            latent_dim, hidden_dim, n_hiddens, activation_type
        )
        self.decoder = nn.Sequential(
            decoder_body, nn.Linear(hidden_dim, data_dim)
        )

    def encode(self, values):
        hidden = self.enc_body(values)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, values):
        mean, log_variance = self.encode(values)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent), mean, log_variance, latent


class PCAAE(nn.Module):
    def __init__(self, data_dim, latent_dim=10):
        super().__init__()
        self.enc = nn.Linear(data_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, data_dim)

    @classmethod
    def fit(cls, train_cells, latent_dim=10, seed=0):
        pca = PCA(latent_dim, random_state=seed).fit(train_cells)
        scale = float(np.sqrt(pca.explained_variance_.mean()))
        components = torch.as_tensor(pca.components_, dtype=torch.float32)
        mean = torch.as_tensor(pca.mean_, dtype=torch.float32)
        model = cls(train_cells.shape[1], latent_dim)
        with torch.no_grad():
            model.enc.weight.copy_(components / scale)
            model.enc.bias.copy_(-(components @ mean) / scale)
            model.decoder.weight.copy_(components.T * scale)
            model.decoder.bias.copy_(mean)
        return model

    def encode(self, values):
        mean = self.enc(values)
        return mean, torch.full_like(mean, -30.0)

    @staticmethod
    def reparameterize(mean, log_variance):
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def decode(self, latent):
        return self.decoder(latent)


def _vae_loss(reconstruction, values, mean, log_variance, beta):
    reconstruction_loss = torch.mean(
        torch.sum((reconstruction - values) ** 2, dim=1)
    )
    kl_loss = -0.5 * torch.mean(
        torch.sum(
            1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1
        )
    )
    return (
        reconstruction_loss + beta * kl_loss,
        reconstruction_loss,
        kl_loss,
    )


def train_vae(
    vae,
    dataframe,
    dim,
    n_epoch=2000,
    batch_size=512,
    lr=1e-3,
    beta=1.0,
    beta_warmup_epochs=0,
    weight_decay=0.0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    show_progress=True,
):
    value_array = dataframe.iloc[:, 1 : dim + 1].to_numpy(
        dtype="float32", copy=True
    )
    values = torch.from_numpy(value_array).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch, eta_min=lr * 1e-2
    )
    history = {"loss": [], "recon": [], "kl": []}
    progress = tqdm(range(n_epoch), desc="Training VAE...", unit="epoch", disable=not show_progress)

    vae.train()
    for epoch in progress:
        indices = torch.randint(0, values.shape[0], (batch_size,), device=device)
        batch = values[indices]
        current_beta = (
            beta
            if beta_warmup_epochs <= 0
            else beta * min(1.0, epoch / beta_warmup_epochs)
        )
        reconstruction, mean, log_variance, _ = vae(batch)
        loss, reconstruction_loss, kl_loss = _vae_loss(
            reconstruction,
            batch,
            mean,
            log_variance,
            current_beta,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        history["recon"].append(reconstruction_loss.item())
        history["kl"].append(kl_loss.item())
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{reconstruction_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
            }
        )

    vae.eval()
    return vae, history


@torch.no_grad()
def encode_dataframe(
    vae,
    dataframe,
    dim,
    use_mean=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    vae.eval()
    value_array = dataframe.iloc[:, 1 : dim + 1].to_numpy(
        dtype="float32", copy=True
    )
    values = torch.from_numpy(value_array).to(device)
    mean, log_variance = vae.encode(values)
    latent = mean if use_mean else vae.reparameterize(mean, log_variance)
    latent = latent.cpu().numpy()
    output = pd.DataFrame(
        latent, columns=[f"z{index}" for index in range(latent.shape[1])]
    )
    output.insert(0, "samples", dataframe["samples"].to_numpy())
    return output


@torch.no_grad()
def decode_latent(
    vae,
    latent,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    vae.eval()
    latent = torch.as_tensor(latent, dtype=torch.float32, device=device)
    return vae.decode(latent).cpu().numpy()
