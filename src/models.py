import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional
from torch.distributions import MultivariateNormal


class Decoder(nn.Module):
    """
        x_t -> y_t
    """
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*x_dim
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, state):
        return self.mlp_layers(state)
    

class Dynamics(nn.Module):
    """
        p(x_t | x_{t-1}, u_{t-1})
    """
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: Optional[int]=None,
        min_std: float=1e-2,
    ):
        super().__init__()

        hidden_dim = (
            hidden_dim if hidden_dim is not None else 2*(x_dim + u_dim)
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, x_dim)
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, x_dim),
            nn.Softplus(),
        )

        self._min_std = min_std

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        hidden = self.mlp_layers(torch.cat([x, u], dim=1))
        mean = self.mean_head(hidden)
        cov = torch.diag_embed(self.std_head(hidden) + self._min_std)
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        return dist
    

class Encoder(nn.Module):
    """
        q(x_t | y1:t, u1:t-1)
    """

    def __init__(
        self,
        y_dim: int,
        u_dim: int,
        x_dim: int,
        rnn_hidden_dim: Optional[int]=128,
        rnn_input_dim: Optional[int]=128,
        min_std: float=1e-2,
    ):
        super().__init__()

        # RNN hidden at time t (ht) summarizes y1:t, u0:t-1
        self.rnn = nn.GRUCell(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
        )

        self.fc_yu = nn.Sequential(
            nn.Linear(y_dim + u_dim, rnn_input_dim),
            nn.ReLU(),
        )

        self.posterior_mean_head = nn.Linear(rnn_hidden_dim, x_dim)
        self.posterior_std_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, x_dim),
            nn.Softplus(),
        )

        self.rnn_hidden_dim = rnn_hidden_dim
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.rnn_input_dim = rnn_input_dim
        self._min_std = min_std

    def forward(
        self,
        rnn_hidden: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
    ):
        """
            Single step update of the rnn hidden
            inputs:
                h: h_{t-1}
                u: u_{t-1}
                y: y_t

            outputs:
                h_t
                q(x_t | y1:t, u0:t-1)
        """

        rnn_input = self.fc_yu(torch.cat((y, u), dim=1))
        rnn_hidden = self.rnn(rnn_input, rnn_hidden)
        mean = self.posterior_mean_head(rnn_hidden)
        cov = torch.diag_embed(self.posterior_std_head(rnn_hidden) + self._min_std)
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

        return dist, rnn_hidden