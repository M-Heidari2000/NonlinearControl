import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from torch.distributions.kl import kl_divergence
from .memory import ReplayBuffer
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
)


def train_backbone(
    config: DictConfig,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        u_dim=train_buffer.u_dim,
        x_dim=config.x_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_input_dim=config.rnn_input_dim,
        min_var=config.min_var
    ).to(device)

    decoder = Decoder(
        x_dim=config.x_dim,
        y_dim=train_buffer.y_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=config.x_dim,
        u_dim=train_buffer.u_dim,
        hidden_dim=config.hidden_dim,
        min_var=config.min_var,
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):
        
        # train
        encoder.train()
        decoder.train()
        dynamics_model.train()

        y, u, _, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = torch.cat((
            torch.zeros((1, config.batch_size, train_buffer.y_dim), device=device),
            einops.rearrange(y, "b l y -> l b y")
        ), dim=0)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        # Initial RNN hidden
        rnn_hidden = torch.zeros((config.batch_size, config.rnn_hidden_dim), device=device)
        posteriors, _ = encoder(rnn_hidden=rnn_hidden, ys=y, us=u)
        # x0:T
        posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
        # reconstruction loss
        y_recon = decoder(einops.rearrange(posterior_samples[1:], "l b x -> (l b) x"))
        y_true = einops.rearrange(y[1:], "l b y -> (l b) y")
        reconstruction_loss = nn.MSELoss()(y_recon, y_true)
        # KL loss
        kl_loss = 0.0
        for t in range(config.chunk_length):
            prior = dynamics_model(x=posterior_samples[t], u=u[t])  # prior at time t+1
            kl_loss += kl_divergence(posteriors[t+1], prior).clamp(min=config.free_nats).mean()
        kl_loss = kl_loss / config.chunk_length

        total_loss = reconstruction_loss + config.kl_beta * kl_loss

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/y reconstruction loss": reconstruction_loss.item(),
            "train/kl loss": kl_loss.item(),
            "global_step": update,
        })
            
        if update % config.test_interval == 0:
            # test
            with torch.no_grad():              
                encoder.eval()
                decoder.eval()
                dynamics_model.eval()

                y, u, _, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = torch.cat((
                    torch.zeros((1, config.batch_size, test_buffer.y_dim), device=device),
                    einops.rearrange(y, "b l y -> l b y")
                ), dim=0)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                # Initial RNN hidden
                rnn_hidden = torch.zeros((config.batch_size, config.rnn_hidden_dim), device=device)
                posteriors, _ = encoder(rnn_hidden=rnn_hidden, ys=y, us=u)
                # x0:T
                posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
                # reconstruction loss
                y_recon = decoder(einops.rearrange(posterior_samples[1:], "l b x -> (l b) x"))
                y_true = einops.rearrange(y[1:], "l b y -> (l b) y")
                reconstruction_loss = nn.MSELoss()(y_recon, y_true)
                # KL loss
                kl_loss = 0.0
                for t in range(config.chunk_length):
                    prior = dynamics_model(x=posterior_samples[t], u=u[t])  # prior at time t+1
                    kl_loss += kl_divergence(posteriors[t+1], prior).clamp(min=config.free_nats).mean()
                kl_loss = kl_loss / config.chunk_length

                total_loss = reconstruction_loss + config.kl_beta * kl_loss

                wandb.log({
                    "test/y reconstruction loss": reconstruction_loss.item(),
                    "test/kl loss": kl_loss.item(),
                    "global_step": update,
                })
                
    return encoder, decoder, dynamics_model
