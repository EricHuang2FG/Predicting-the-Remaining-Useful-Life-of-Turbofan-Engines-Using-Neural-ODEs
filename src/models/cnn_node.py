import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.node import NeuralODE


class CNN_ODE(nn.Module):
    def __init__(
        self,
        input_dimension: int = 24,
        cnn_num_kernals: int = 36,
        cnn_kernal_size: int = 3,
        cnn_stride: int = 1,
        cnn_padding: int = 1,
        hidden_dimension: int = 64,
        encoder_dimension: int = 128,
        regressor_dimension: int = 32,
        dropout: float = 0.2,
        sequence_length: int = 40,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dimension,
                out_channels=cnn_num_kernals,
                kernel_size=cnn_kernal_size,
                stride=cnn_stride,
                padding=cnn_padding,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * cnn_num_kernals, encoder_dimension),
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
        cnn_in = x.transpose(1, 2)
        cnn_out = self.cnn(cnn_in)
        theta_0 = self.encoder(cnn_out)
        theta_final = odeint(self.ode, theta_0, t_span, method="dopri5")[-1]
        prediction = self.regressor(theta_final).squeeze()

        return prediction
