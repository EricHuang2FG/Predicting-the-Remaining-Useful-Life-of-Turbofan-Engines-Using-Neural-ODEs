import torch
import torch.nn as nn
from torchdiffeq import odeint


class NeuralODE(nn.Module):

    def __init__(self, dimension: int = 64):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, dimension)
        )

        # force some weights at initialization
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, t, theta):
        output = self.linear_stack(theta)
        output = torch.clamp(output, -10.0, 10.0)
        return output


class ODE(nn.Module):
    # still need to tune the parameters of the neural netowrk that is solving the derivatives at each point of the ODE
    def __init__(
        self,
        input_dimension: int = 24,
        hidden_dimension: int = 64,
        encoder_dimension: int = 128,
        regressor_dimension: int = 32,
        dropout: float = 0.2,
        sequence_length: int = 40,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimension * sequence_length, encoder_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dimension, hidden_dimension),
        )

        self.ode = NeuralODE(hidden_dimension)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dimension, regressor_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_dimension, 1),
        )

    def forward(self, x, t_span=torch.tensor([0.0, 1.0])):
        # get initial state
        theta_0 = self.encoder(x)
        theta_final = odeint(self.ode, theta_0, t_span, method="dopri5")[-1]
        prediction = self.regressor(theta_final).squeeze()

        return prediction
