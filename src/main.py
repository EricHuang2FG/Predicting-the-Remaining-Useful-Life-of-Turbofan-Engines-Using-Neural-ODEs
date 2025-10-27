import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint

NUM_SETTINGS_AND_SENSOR_READINGS: int = 24


# reference paper for this processing:
# https://link.springer.com/article/10.1007/s44196-024-00639-w#data-availability
def preprocess_training_data(
    file_path: str, window_size: int = 40, max_rul: int = 100
) -> tuple[torch.FloatTensor, torch.FloatTensor, StandardScaler]:
    # data has the size (num_lines, 26)
    # column 1: engine number
    # column 2: cycle count
    # column 3 to 5: operational settings
    # column 6 to end: sensor measurements
    data: np.ndarray = np.loadtxt(file_path)

    sliding_window_sequences: list = []
    rul_labels: list = []

    # engine_num = [1, 2, 3, 4, ..., ]
    for engine_num in np.unique(data[:, 0]):
        # get data s.t. its first column is the engine_num
        engine_data: np.ndarray = data[data[:, 0] == engine_num]
        max_life: int = len(engine_data)

        # perform RUL segmentation
        # element-wise subtraction of max_num_cycles - engine_data array
        # then cap every element at max_rul (ref the paper)
        rul: np.ndarray = max_life - engine_data[:, 1]
        rul_segmented: np.ndarray = np.minimum(max_rul, rul)

        # do sliding window processing
        settings_and_sensor_readings: np.ndarray = engine_data[:, 2:26]
        for i in range(len(settings_and_sensor_readings) - window_size + 1):
            sliding_window_sequences.append(
                settings_and_sensor_readings[i : i + window_size]
            )
            rul_labels.append(
                rul_segmented[i + window_size - 1]
            )  # rul at the end of the window

    # shape: (num_sequences, window_size, 24)
    input_data: np.ndarray = np.array(sliding_window_sequences)
    expected_output: np.ndarray = np.array(rul_labels)

    # standardize using Z = (x_i - u) / sigma
    # documentation for this one: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # it does Z = (x_i - u) / sigma for you
    scaler: StandardScaler = StandardScaler()
    orig_shape: tuple = input_data.shape
    input_data_flattened: np.ndarray = input_data.reshape(
        -1, NUM_SETTINGS_AND_SENSOR_READINGS
    )
    input_data_flattened = scaler.fit_transform(input_data_flattened)
    input_data = input_data_flattened.reshape(orig_shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(expected_output), scaler


def preprocess_test_data(
    test_data_path: str, rul_path: str, scaler: StandardScaler, window_size: int = 40
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # similar to preprocess_trainng_data() but runs on test data and real RUL
    # also only takes the last window of each engine test
    test_data: np.ndarray = np.loadtxt(test_data_path)
    rul_data: np.ndarray = np.loadtxt(rul_path)

    test_sequence: list = []

    for engine_num in np.unique(test_data[:, 0]):
        engine_data: np.ndarray = test_data[test_data[:, 0] == engine_num]

        engine_sequence: np.ndarray = engine_data[-window_size:, 2:26]
        engine_sequence_size: int = engine_sequence.shape[0]
        if engine_sequence_size < window_size:
            filler: np.ndarray = np.zeros(
                (window_size - engine_sequence_size, NUM_SETTINGS_AND_SENSOR_READINGS)
            )
            engine_sequence = np.concatenate((filler, engine_sequence), axis=0)

        test_sequence.append(engine_sequence)

    input_data: np.ndarray = np.array(test_sequence)

    # standardize using scaler from TRAIN DATA
    orig_shape: tuple = input_data.shape
    input_data_flattened: np.ndarray = input_data.reshape(
        -1, NUM_SETTINGS_AND_SENSOR_READINGS
    )
    input_data_flattened = scaler.transform(input_data_flattened)
    input_data = input_data_flattened.reshape(orig_shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(rul_data)


class NeuralODE(nn.Module):

    def __init__(self, dimension=64):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, dimension)
        )

        # this just force some weights at initialization
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

    torch.save(model.state_dict(), dest_path)
    return model


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
        if abs(value) <= 100:  # remove bad values like -1500
            sum_squared_error += (value - ground_truth_array[index]) ** 2
    root_mean_squared_erro_error: float = (
        sum_squared_error / prediction_array.shape[0]
    ) ** 0.5
    print(root_mean_squared_erro_error)

    rmse = np.sqrt(np.mean((prediction_array_cleaned - ground_truth_array) ** 2))

    print(rmse)


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


if __name__ == "__main__":
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")
    # model = train_model("models/ode.FD001.v3.model", x, y)
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD001.txt", "CMAPSS/RUL_FD001.txt", scaler
    )
    evaluate_model(x_test, y_test, "models/ode.FD001.v3.model")
