import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint

from src.constants import NUM_SETTINGS_AND_SENSOR_READINGS


class NeuralODE(nn.Module):

    def __init__(self, dimension=64):
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

    def __init__(self, input_dimension=24, hidden_dimension=64, sequence_length=40):
        super().__init__()
        # flatten input into one vector
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimension * sequence_length, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, hidden_dimension),
        )

        self.ode = NeuralODE(hidden_dimension)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dimension, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x, t_span=torch.tensor([0.0, 1.0])):
        # get initial state
        theta_0 = self.encoder(x)  # shape: (batch_size, hidden_dimension)
        # model the engine data as ODE and solve it
        theta_final = odeint(self.ode, theta_0, t_span, method="dopri5")[
            -1
        ]  # shape: (batch_size, hidden_dimension)
        # turn it into a number
        prediction = self.regressor(theta_final).squeeze()  # shape: (batch_size)

        return prediction


def train_model(dest_path, input_data, expected_output, epochs=25):
    model = ODE(input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS, hidden_dimension=64)
    loss_function = nn.MSELoss()
    # idk what value of lr is good so picked a random one
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(input_data, expected_output)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            predictions = model(x)
            loss = loss_function(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss / len(dataloader)):.6f}"
        )

    # save the model as a .model file so you don't have to retrain every single time
    torch.save(model.state_dict(), dest_path)
    return model


# rn this function is quite bad and hard coded
# can make it beter in the future
# also have not fully implemented evaluation criteria
def evaluate_model(
    input_data: torch.FloatTensor,
    expected_output: torch.FloatTensor,
    model_path: str = "models/ode.model",
    input_dimension: int = NUM_SETTINGS_AND_SENSOR_READINGS,
    hidden_dimension: int = 64,
    sequence_length: int = 40,
) -> None:
    model: ODE = load_model(
        model_path,
        input_dimension=input_dimension,
        hidden_dimension=hidden_dimension,
        sequence_length=sequence_length,
    )
    model.eval()

    with torch.no_grad():
        predictions: torch.FloatTensor = model(input_data)

    prediction_array: np.ndarray = predictions.numpy()
    ground_truth_array: np.ndarray = expected_output.numpy()

    # clip to be below 100 (the paper does this too)
    prediction_array = np.minimum(100, prediction_array)
    ground_truth_array = np.minimum(100, ground_truth_array)

    # clean out bad values like -1500 (there are about 4 of them out of 100)
    prediction_array_cleaned: list = [
        value if 0 <= value <= 100 else 100 for value in prediction_array
    ]
    prediction_array_cleaned: np.ndarray = np.array(prediction_array_cleaned)

    sort_indicies: np.ndarray = np.argsort(ground_truth_array)
    ground_truth_sorted: np.ndarray = ground_truth_array[sort_indicies]
    prediction_array_cleaned_sorted: np.ndarray = prediction_array_cleaned[
        sort_indicies
    ]

    x = list(range(len(prediction_array)))

    plt.plot(x, ground_truth_sorted, color="red")
    plt.plot(x, prediction_array_cleaned_sorted, color="purple")
    plt.show()

    sum_squared_error: int = 0
    for index, value in enumerate(prediction_array):
        # remove bad values like -1500
        clipped_value = value if abs(value) <= 100 else 100
        sum_squared_error += (clipped_value - ground_truth_array[index]) ** 2
    root_mean_squared_erro_error: float = (
        sum_squared_error / prediction_array.shape[0]
    ) ** 0.5
    print(root_mean_squared_erro_error)

    rmse = np.sqrt(np.mean((prediction_array_cleaned - ground_truth_array) ** 2))

    mape = np.sum(
        np.abs((prediction_array_cleaned - ground_truth_array) / ground_truth_array)
    ) / len(prediction_array_cleaned)

    print(rmse)
    print(mape)


def load_model(
    path: str = "models/ode.model",
    input_dimension: int = NUM_SETTINGS_AND_SENSOR_READINGS,
    hidden_dimension: int = 64,
    sequence_length: int = 40,
) -> ODE:
    model: ODE = ODE(
        input_dimension=input_dimension,
        hidden_dimension=hidden_dimension,
        sequence_length=sequence_length,
    )
    model.load_state_dict(torch.load(path))

    return model
